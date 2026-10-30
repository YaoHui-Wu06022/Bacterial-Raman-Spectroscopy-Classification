"""对 init/init_test 做可复现的光谱特征工程与跨文件夹可分性评估。"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from ramanv2.common.arc_data import read_raw_arc_data
from ramanv2.common.naming import build_natural_key, parse_folder_prefix
from ramanv2.core.config import InputConfig
from ramanv2.data.config import DataBuildConfig
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import preprocess_single_spectrum


KNOWN_PEAKS = (620.9, 1001.4, 1031.8, 1583.1, 1602.3)
INTERVAL_COUNT = 12


@dataclass(frozen=True)
class SpectrumRecord:
    """一条光谱及其两层标签和采集文件夹。"""

    path: Path
    domain: str
    level_1: str
    level_2: str
    folder: str


def _load_prefix_map(catalog_path: Path) -> dict[str, str]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {
        prefix.upper(): genus
        for genus, prefixes in payload["genera"].items()
        for prefix in prefixes
    }


def _collect_records(root: Path, prefix_map: dict[str, str]) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    init_root = root / "init"
    for path in sorted(init_root.rglob("*.arc_data"), key=lambda item: build_natural_key(item.as_posix())):
        folder = path.parent.name
        genus = path.parent.parent.name
        prefix = parse_folder_prefix(folder, uppercase_enable=True)
        records.append(SpectrumRecord(path, "init", genus, f"{genus}/{prefix}", folder))
    test_root = root / "init_test"
    for path in sorted(test_root.rglob("*.arc_data"), key=lambda item: build_natural_key(item.as_posix())):
        folder = path.parent.name
        prefix = parse_folder_prefix(folder, uppercase_enable=True)
        genus = prefix_map.get(prefix)
        if genus is None:
            raise ValueError(f"无法映射 init_test 前缀：{prefix} ({path})")
        records.append(SpectrumRecord(path, "init_test", genus, f"{genus}/{prefix}", folder))
    return records


def _segment_derivative(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """按坏段分割计算导数，避免跨越 890–950 cm⁻¹ 间隙。"""
    derivative = np.zeros_like(values, dtype=np.float64)
    step = float(np.median(np.diff(axis)))
    breaks = np.flatnonzero(np.diff(axis) > step * 2.0) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, values.size]
    for start, end in zip(starts, ends):
        if end - start >= 5:
            derivative[start:end] = savgol_filter(
                values[start:end], 15, 3, deriv=1, delta=step, mode="interp"
            )
    return derivative


def _extract_peak_features(values: np.ndarray, axis: np.ndarray, centers: Iterable[float]) -> np.ndarray:
    features: list[float] = []
    step = float(np.median(np.diff(axis)))
    for center in centers:
        mask = np.abs(axis - center) <= 12.0
        if mask.sum() < 5:
            features.extend([0.0, 0.0, 0.0])
            continue
        local = values[mask]
        local_axis = axis[mask]
        peak_index = int(np.argmax(local))
        features.append(float(local[peak_index]))
        features.append(float(np.trapezoid(np.maximum(local, 0.0), local_axis)))
        try:
            widths = peak_widths(local, [peak_index], rel_height=0.5)[0]
            features.append(float(widths[0] * step))
        except (IndexError, ValueError):
            features.append(0.0)
    return np.asarray(features, dtype=np.float64)


def _extract_interval_features(values: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for chunk in np.array_split(values, INTERVAL_COUNT):
        features.extend([float(np.mean(chunk)), float(np.std(chunk)), float(np.percentile(chunk, 90) - np.percentile(chunk, 10))])
    for chunk in np.array_split(derivative, INTERVAL_COUNT):
        features.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return np.asarray(features, dtype=np.float64)


def _preprocess_records(
    records: list[SpectrumRecord],
    input_config: InputConfig,
    build_config: DataBuildConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], np.ndarray, list[float], list[SpectrumRecord]]:
    reference = build_wn_ref(input_config.cut_min, input_config.cut_max, input_config.target_points)
    corrected_values: list[np.ndarray] = []
    snv_values: list[np.ndarray] = []
    smooth_values: list[np.ndarray] = []
    derivative_values: list[np.ndarray] = []
    peak_values: list[np.ndarray] = []
    interval_values: list[np.ndarray] = []
    quality_rows: list[dict[str, object]] = []
    valid_records: list[SpectrumRecord] = []
    axes: list[np.ndarray] = []
    peak_centers: list[float] = list(KNOWN_PEAKS)
    raw_count = len(records)
    for index, record in enumerate(records, 1):
        try:
            axis, intensity, malformed = read_raw_arc_data(record.path)
        except FileNotFoundError:
            quality_rows.append({"path": str(record.path), "domain": record.domain, "level_1": record.level_1, "level_2": record.level_2, "folder": record.folder, "status": "missing_during_analysis"})
            continue
        raw_valid = bool(axis.size and axis.size == intensity.size and np.isfinite(axis).all() and np.isfinite(intensity).all() and np.all(np.diff(axis) > 0))
        output_axis, corrected, _ = preprocess_single_spectrum(
            axis,
            intensity,
            cut_min=input_config.cut_min,
            cut_max=input_config.cut_max,
            reference_wavenumbers=reference,
            bad_bands=input_config.bad_bands,
            baseline_lam=build_config.baseline_lam,
            baseline_asls_p=build_config.baseline_asls_p,
            baseline_max_iter=build_config.baseline_max_iter,
            baseline_fit_min=build_config.baseline_fit_min,
            baseline_fit_max=build_config.baseline_fit_max,
            baseline_method=build_config.baseline_method,
            **build_config.build_cosmic_ray_options("GN"),
        )
        if output_axis is None or corrected is None:
            raise ValueError(f"预处理失败：{record.path}")
        output_axis = np.asarray(output_axis, dtype=np.float64)
        corrected = np.asarray(corrected, dtype=np.float64)
        snv = np.asarray(normalize_spectrum(corrected, "snv"), dtype=np.float64)
        smooth = savgol_filter(corrected, 15, 3, mode="interp")
        smooth_snv = np.asarray(normalize_spectrum(smooth, "snv"), dtype=np.float64)
        derivative = _segment_derivative(smooth_snv, output_axis)
        corrected_values.append(corrected)
        snv_values.append(snv)
        smooth_values.append(smooth_snv)
        derivative_values.append(derivative)
        peak_values.append(_extract_peak_features(smooth_snv, output_axis, peak_centers))
        interval_values.append(_extract_interval_features(smooth_snv, derivative))
        axes.append(output_axis)
        quality_rows.append({
            "path": str(record.path),
            "domain": record.domain,
            "level_1": record.level_1,
            "level_2": record.level_2,
            "folder": record.folder,
            "raw_points": int(axis.size),
            "malformed_lines": int(malformed),
            "raw_finite_monotonic": raw_valid,
            "raw_axis_start": float(axis[0]) if axis.size else None,
            "raw_axis_end": float(axis[-1]) if axis.size else None,
            "processed_points": int(corrected.size),
            "processed_finite": bool(np.isfinite(corrected).all()),
            "status": "ok",
        })
        valid_records.append(record)
        if index % 500 == 0:
            print(f"processed {index}/{raw_count}")
    feature_table = {
        "corrected_full": np.vstack(corrected_values),
        "snv_full": np.vstack(snv_values),
        "smooth_snv": np.vstack(smooth_values),
        "derivative": np.vstack(derivative_values),
        "peak": np.vstack(peak_values),
        "interval": np.vstack(interval_values),
    }
    feature_table["combined"] = np.hstack([feature_table["peak"], feature_table["interval"]])
    return feature_table, quality_rows, np.asarray(axes[0]), peak_centers, valid_records


def _majority_folder_accuracy(groups: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = 0
    total = 0
    for group in np.unique(groups):
        mask = groups == group
        true_values, counts = np.unique(y_true[mask], return_counts=True)
        truth = true_values[int(np.argmax(counts))]
        pred_values, counts = np.unique(y_pred[mask], return_counts=True)
        prediction = pred_values[int(np.argmax(counts))]
        correct += int(truth == prediction)
        total += 1
    return correct / total if total else 0.0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "folder_accuracy": float(_majority_folder_accuracy(groups, y_true, y_pred)),
        "support": int(y_true.size),
        "folder_count": int(np.unique(groups).size),
    }


def _build_model(name: str):
    if name == "nearest_centroid":
        from sklearn.neighbors import NearestCentroid

        return make_pipeline(StandardScaler(), NearestCentroid())
    if name == "linear_svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=42))
    if name == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=20, whiten=True, random_state=42),
            SVC(C=3.0, gamma="scale"),
        )
    raise ValueError(f"未知模型：{name}")


def _nearest_centroid_predict(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    """用训练折统计量完成向量化最近中心预测，避免逐折构建 sklearn 管道。"""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled = (train_features - mean) / scale
    test_scaled = (test_features - mean) / scale
    labels = np.unique(train_labels)
    centroids = np.vstack([train_scaled[train_labels == label].mean(axis=0) for label in labels])
    distances = ((test_scaled[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    return labels[np.argmin(distances, axis=1)]


def _evaluate_nearest_lofo(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """用总和减去留出文件夹的方式快速完成最近中心 LOFO。"""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (features - mean) / scale
    label_values = np.asarray(labels, dtype=object)
    total_sums = {label: scaled[y == label].sum(axis=0) for label in label_values}
    total_counts = {label: int(np.sum(y == label)) for label in label_values}
    predictions = np.empty(y.shape, dtype=object)
    for group in np.unique(groups):
        test_index = np.flatnonzero(groups == group)
        train_labels = y[test_index[0]]
        train_sums = {label: total_sums[label].copy() for label in label_values}
        train_counts = dict(total_counts)
        train_sums[train_labels] -= scaled[test_index].sum(axis=0)
        train_counts[train_labels] -= int(test_index.size)
        available = [label for label in label_values if train_counts[label] > 0]
        centroids = np.vstack([train_sums[label] / train_counts[label] for label in available])
        distances = ((scaled[test_index, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        predictions[test_index] = np.asarray(available, dtype=object)[np.argmin(distances, axis=1)]
    return _metrics(y, predictions, groups, labels), y, predictions


def _evaluate_grouped(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    model_name: str,
    protocol: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if model_name == "nearest_centroid" and protocol == "lofo":
        return _evaluate_nearest_lofo(features, y, groups, labels)
    if protocol == "lofo":
        splitter = LeaveOneGroupOut()
    elif protocol == "group5":
        splitter = GroupKFold(n_splits=min(5, np.unique(groups).size))
    else:
        raise ValueError(protocol)
    predictions = np.empty(y.shape, dtype=object)
    for train_index, test_index in splitter.split(features, y, groups):
        if model_name == "nearest_centroid":
            predictions[test_index] = _nearest_centroid_predict(features[train_index], y[train_index], features[test_index])
        else:
            model = _build_model(model_name)
            model.fit(features[train_index], y[train_index])
            predictions[test_index] = model.predict(features[test_index])
    return _metrics(y, predictions, groups, labels), y, predictions


def _evaluate_external(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    test_groups: np.ndarray,
    labels: list[str],
    model_name: str,
) -> tuple[dict[str, float], np.ndarray]:
    if model_name == "nearest_centroid":
        predictions = _nearest_centroid_predict(train_features, train_y, test_features)
    else:
        model = _build_model(model_name)
        model.fit(train_features, train_y)
        predictions = model.predict(test_features)
    return _metrics(test_y, predictions, test_groups, labels), predictions


def _domain_auc(features: np.ndarray, domains: np.ndarray, groups: np.ndarray) -> float:
    splitter = GroupKFold(n_splits=min(5, np.unique(groups).size))
    probabilities = np.zeros(domains.size, dtype=np.float64)
    for train_index, test_index in splitter.split(features, domains, groups):
        model = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=42))
        model.fit(features[train_index], domains[train_index])
        probabilities[test_index] = model.decision_function(features[test_index])
    return float(roc_auc_score(domains, probabilities))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_pca(features: np.ndarray, records: list[SpectrumRecord], output_path: Path) -> None:
    embedding = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(features))
    domains = np.asarray([record.domain for record in records])
    figure, axis = plt.subplots(figsize=(10, 7))
    for domain, color in (("init", "#1f77b4"), ("init_test", "#d62728")):
        mask = domains == domain
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.35, label=domain, c=color)
    axis.set_title("Current GN init vs init_test PCA (SNV)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.45), max(7, len(labels) * 0.4)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_feature_engineering_analysis(
    project_root: Path | str = Path("."),
    output_dir: Path | str | None = None,
) -> Path:
    """运行当前 init/init_test 特征工程分析并原子发布结果目录。"""
    project_root = Path(project_root).resolve()
    profile_root = project_root / "dataset" / "GN"
    root = profile_root if (profile_root / "init").is_dir() else project_root / "dataset"
    if output_dir is None:
        output_dir = project_root / "output" / ("GN/feature_engineering_current" if root == profile_root else "feature_engineering_current")
    output_dir = Path(output_dir).resolve()
    temp_dir = output_dir.with_name(output_dir.name + "_tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    prefix_map = _load_prefix_map(project_root / "dataset" / ".raman_pipeline" / "profile_catalog.json")
    records = _collect_records(root, prefix_map)
    input_config = InputConfig()
    build_config = DataBuildConfig()
    feature_table, quality_rows, axis, peak_centers, records = _preprocess_records(records, input_config, build_config)
    _write_csv(temp_dir / "feature_quality.csv", quality_rows)

    domains = np.asarray([record.domain for record in records])
    folders = np.asarray([record.folder for record in records])
    domain_groups = np.asarray([f"{record.domain}/{record.folder}" for record in records])
    level_1 = np.asarray([record.level_1 for record in records])
    level_2 = np.asarray([record.level_2 for record in records])
    init_mask = domains == "init"
    test_mask = domains == "init_test"
    feature_names = ["snv_full", "derivative", "peak", "interval", "combined"]
    model_names = ["nearest_centroid", "linear_svm", "rbf_svm"]
    metrics_rows: list[dict[str, object]] = []
    best_external: dict[str, tuple[float, str, str, np.ndarray, np.ndarray]] = {}
    for level_name, labels_array in (("level_1", level_1), ("level_2", level_2)):
        labels = sorted(set(labels_array[init_mask]))
        for feature_name in feature_names:
            features = feature_table[feature_name]
            domain_auc = _domain_auc(features, domains.astype(str) == "init_test", domain_groups)
            for model_name in model_names:
                if model_name == "rbf_svm" and feature_name not in {"peak", "interval", "combined"}:
                    continue
                internal_protocol = "lofo" if model_name == "nearest_centroid" else "group5"
                internal_metrics, _, _ = _evaluate_grouped(features[init_mask], labels_array[init_mask], folders[init_mask], labels, model_name, internal_protocol)
                external_metrics, external_predictions = _evaluate_external(features[init_mask], features[test_mask], labels_array[init_mask], labels_array[test_mask], folders[test_mask], labels, model_name)
                row = {"level": level_name, "feature_group": feature_name, "model": model_name, "internal_protocol": internal_protocol, "domain_auc": domain_auc}
                row.update({f"internal_{key}": value for key, value in internal_metrics.items()})
                row.update({f"external_{key}": value for key, value in external_metrics.items()})
                metrics_rows.append(row)
                score = float(external_metrics["macro_f1"])
                previous = best_external.get(level_name)
                if previous is None or score > previous[0]:
                    best_external[level_name] = (score, feature_name, model_name, labels_array[test_mask], external_predictions)
    _write_csv(temp_dir / "feature_metrics.csv", metrics_rows)
    for level_name, (score, feature_name, model_name, y_true, y_pred) in best_external.items():
        labels = sorted(set(y_true))
        _plot_confusion(y_true, y_pred, labels, temp_dir / f"confusion_{level_name}.png", f"External init_test: {level_name} {feature_name}/{model_name} Macro-F1={score:.3f}")

    _plot_pca(feature_table["snv_full"], records, temp_dir / "pca_domain.png")
    summary = {
        "input": {"init_count": int(init_mask.sum()), "init_test_count": int(test_mask.sum()), "init_folder_count": int(np.unique(folders[init_mask]).size), "init_test_folder_count": int(np.unique(folders[test_mask]).size), "level_1_count": int(np.unique(level_1[init_mask]).size), "level_2_count": int(np.unique(level_2[init_mask]).size), "processed_points": int(feature_table["snv_full"].shape[1]), "known_peak_centers": peak_centers},
        "best_external": {level: {"macro_f1": score, "feature_group": feature, "model": model} for level, (score, feature, model, _, _) in best_external.items()},
        "recommendation": "优先按 init 文件夹分组验证；若线性模型与 RBF 模型都在 init_test 下降，先处理批次/文件夹域偏移，再调模型复杂度。",
    }
    (temp_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# 当前 GN init / init_test 特征工程与可分性分析",
        "",
        f"- init: {summary['input']['init_count']} 条，{summary['input']['init_folder_count']} 个文件夹。",
        f"- init_test: {summary['input']['init_test_count']} 条，{summary['input']['init_test_folder_count']} 个文件夹。",
        f"- level_1 属数量: {summary['input']['level_1_count']}；level_2 种前缀数量: {summary['input']['level_2_count']}。",
        "",
        "## 外部测试最佳组合",
        "",
    ]
    for level, values in summary["best_external"].items():
        report_lines.append(f"- {level}: {values['feature_group']} + {values['model']}，Macro-F1={values['macro_f1']:.4f}。")
    report_lines += [
        "",
        "## 解释",
        "",
        "内部评估按采集文件夹分组；RBF SVM 使用 5 折 GroupKFold，其余模型使用 Leave-One-Folder-Out。init_test 只用于外部评估。波数轴特征仅用于质量和域偏移分析，没有作为分类输入。",
        "",
        "详细指标见 `feature_metrics.csv`，文件夹质量见 `feature_quality.csv`，混淆矩阵和 PCA 图位于同一目录。",
    ]
    (temp_dir / "feature_engineering_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temp_dir.rename(output_dir)
    return output_dir


def main() -> None:
    output = run_feature_engineering_analysis()
    print(output)


if __name__ == "__main__":
    main()
