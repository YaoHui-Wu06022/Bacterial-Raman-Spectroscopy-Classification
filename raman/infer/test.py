from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from raman.data import resolve_dataset_stage
from raman.data.spectrum import build_valid_mask, get_config_bad_bands
from raman.infer.folder import (
    format_prediction_report,
    iter_predict_folders,
    list_arc_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path):
    """把项目相对路径解析为绝对路径"""
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _bundle_root(path):
    """把阶段目录还原到数据集根目录"""
    path = Path(path)
    if path.name in {"train", "train_raw", "test", "init_test"}:
        return path.parent
    return path


def _expected_wavenumbers(config):
    """按模型配置生成严格匹配的波数轴"""
    target_points = int(config.target_points)
    wn = np.linspace(float(config.cut_min), float(config.cut_max), target_points)
    keep_mask = build_valid_mask(wn, get_config_bad_bands(config))
    if keep_mask is not None:
        wn = wn[keep_mask]
    return wn


def _validate_input_length(signal_length, config, source):
    """确认测试谱长度和模型训练配置一致"""
    expected = _expected_wavenumbers(config).shape[0]
    if int(signal_length) != int(expected):
        raise ValueError(
            f"Input length mismatch for {source}: got {signal_length}, expected {expected}. "
            f"请用该模型的 config.yaml 重新从 init_test 生成 test，或在 infer_test.py 中保持 BUILD_TEST_FIRST=True"
        )


def _preprocess_with_config_mask(path, preprocessor, config):
    """按模型 bad_bands 对齐已预处理光谱，再构建模型输入"""
    x = preprocessor(path)
    expected = _expected_wavenumbers(config).shape[0]
    if int(x.shape[-1]) == int(expected):
        return x

    data = np.loadtxt(path, dtype=np.float32)
    data = np.atleast_2d(data)
    if data.shape[1] >= 2:
        keep_mask = build_valid_mask(data[:, 0], get_config_bad_bands(config))
        if keep_mask is not None and int(keep_mask.sum()) == int(expected):
            from raman.data.input import build_model_input

            signal = data[:, 1][keep_mask].astype(np.float32, copy=False)
            aligned = build_model_input(
                signal,
                config=config,
                sg_smooth=preprocessor.sg_smooth,
                sg_d1=preprocessor.sg_d1,
                device=preprocessor.device,
                augment=False,
            )
            return aligned.unsqueeze(0)

    _validate_input_length(x.shape[-1], config, path)
    return x


def _normalize_folder_prefix(name):
    """把 KP02、CITF03 统一成 KP、CITF"""
    match = re.match(r"^([A-Za-z]+)", str(name))
    return match.group(1).upper() if match else str(name).upper()


def _test_folder_prefix(name):
    """把 CS01KP、CS5EC 统一成 KP、EC"""
    text = str(name).strip()
    match = re.match(r"^CS\d*(.+)$", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return _normalize_folder_prefix(text)


def _level_number(level_name):
    """读取 level_1 里的数字"""
    return int(str(level_name).split("_", 1)[1])


def _label_from_parts(parts, level_name):
    """按路径层级推断指定业务层标签"""
    if not parts:
        return None
    level_no = _level_number(level_name)
    if len(parts) < level_no:
        return None
    return "/".join(parts[:level_no])


def _candidate_train_roots(dataset_root):
    """列出可用的训练侧光谱目录，优先使用最终 train"""
    dataset_root = _bundle_root(_resolve_project_path(dataset_root))
    train_root = dataset_root / "train"
    if train_root.is_dir():
        return [train_root]
    train_raw_root = dataset_root / "train_raw"
    return [train_raw_root] if train_raw_root.is_dir() else []


def _iter_labeled_train_files(train_root, level_name):
    """遍历训练侧光谱并带上业务层标签"""
    train_root = Path(train_root)
    for path in sorted(train_root.rglob("*.arc_data")):
        rel = path.relative_to(train_root)
        if len(rel.parts) < 3:
            continue
        label = _label_from_parts(rel.parts[:-1], level_name)
        if label:
            yield path, label, _normalize_folder_prefix(rel.parts[-2])


def _build_expected_lookup(exp_dir, train_roots, level_name):
    """由模型训练清单建立测试简称到各级真实标签的映射"""
    counters: dict[str, Counter] = {}
    train_files = _load_train_file_list(exp_dir)
    if train_files:
        for rel in train_files:
            parts = Path(rel).parts[:-1]
            if not parts:
                continue
            label = _label_from_parts(parts, level_name)
            prefix = _normalize_folder_prefix(parts[-1])
            if label and prefix:
                counters.setdefault(prefix, Counter())[label] += 1

    if counters:
        return {
            prefix: counter.most_common(1)[0][0]
            for prefix, counter in counters.items()
            if counter
        }

    for train_root in train_roots:
        for _, label, prefix in _iter_labeled_train_files(train_root, level_name):
            counters.setdefault(prefix, Counter())[label] += 1
    return {
        prefix: counter.most_common(1)[0][0]
        for prefix, counter in counters.items()
        if counter
    }


def _load_train_file_list(exp_dir):
    """读取模型目录保存的训练文件清单"""
    path = Path(exp_dir) / "train_files.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_train_mean_files(exp_dir, dataset_root, level_name):
    """优先使用模型训练清单，缺失时回退到当前 train_raw"""
    dataset_root = _bundle_root(_resolve_project_path(dataset_root))
    train_files = _load_train_file_list(exp_dir)
    train_root = dataset_root / "train"
    if train_files and train_root.is_dir():
        rows = []
        for rel in train_files:
            path = train_root / rel
            if not path.is_file():
                continue
            parts = Path(rel).parts[:-1]
            label = _label_from_parts(parts, level_name)
            if label:
                rows.append((path, label))
        if rows:
            return rows

    rows = []
    for root in _candidate_train_roots(dataset_root):
        for path, label, _ in _iter_labeled_train_files(root, level_name):
            rows.append((path, label))
    if not rows:
        raise FileNotFoundError(f"No train spectra found under {dataset_root}")
    return rows


def _build_train_mean_bank(exp_dir, dataset_root, level_name, preprocessor, config):
    """构建训练均值谱对照库"""
    from tqdm import tqdm

    signals: dict[str, list[np.ndarray]] = {}
    for path, label in tqdm(
        _resolve_train_mean_files(exp_dir, dataset_root, level_name),
        desc="Building train mean spectra",
        unit="spectrum",
    ):
        x = _preprocess_with_config_mask(path, preprocessor, config)
        signal = x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        signals.setdefault(label, []).append(signal)
    return {
        label: np.mean(np.stack(items, axis=0), axis=0)
        for label, items in signals.items()
    }


def _resolve_test_root(config, override):
    """解析本次独立测试的已处理 test 目录"""
    if override:
        path = Path(override)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    dataset_root = _bundle_root(_resolve_project_path(config.dataset_root))
    return resolve_dataset_stage(
        dataset_root,
        stage="test",
        project_root=PROJECT_ROOT,
        must_exist=True,
    )


def _load_transfer_skip_lookup(manifest_path):
    """读取测试样本迁移 manifest，返回需要跳过的源文件"""
    if not manifest_path:
        return {}
    path = _resolve_project_path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transfer manifest not found: {path}")

    lookup: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            source_folder = (row.get("source_folder") or "").strip()
            source_file = (row.get("source_file") or "").strip()
            if source_folder and source_file:
                lookup.setdefault(source_folder, set()).add(source_file)
    return lookup


def _predict_folder(folder_path, predictor, top_k, skip_lookup=None):
    """预测单个测试文件夹并收集绘图用谱线"""
    from tqdm import tqdm
    from raman.infer.core import predict_tensor

    predictions = []
    signals = []
    skipped = []
    skip_files = (skip_lookup or {}).get(folder_path.name, set())
    for path in tqdm(list_arc_files(folder_path), desc=f"Predicting {folder_path.name}", unit="spectrum"):
        if path.name in skip_files:
            skipped.append(path.name)
            continue
        x = _preprocess_with_config_mask(path, predictor["preprocessor"], predictor["config"])
        results = predict_tensor(x, predictor, top_k=top_k)
        signal = x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        predictions.append(
            {
                "file": path.name,
                "path": path,
                "results": results,
                "top1_label": results[0]["label"],
            }
        )
        signals.append(signal)
    return predictions, np.stack(signals, axis=0) if signals else np.empty((0, 0), dtype=np.float32), skipped


def _folder_summary(folder_name, expected_label, class_names, predictions):
    """汇总单个测试文件夹的多数投票结果"""
    total = len(predictions)
    counter = Counter(item["top1_label"] for item in predictions)
    predicted, majority_count = counter.most_common(1)[0] if counter else ("unknown", 0)
    expected_in_model = expected_label in set(class_names)
    correct_count = (
        sum(1 for item in predictions if item["top1_label"] == expected_label)
        if expected_in_model
        else 0
    )
    correct_ratio = correct_count / total if total else 0.0
    return {
        "folder": folder_name,
        "expected_label": expected_label or "unknown",
        "expected_in_model": bool(expected_in_model),
        "predicted_label": predicted,
        "majority_count": int(majority_count),
        "total_count": int(total),
        "correct_count": int(correct_count),
        "correct_ratio": float(correct_ratio),
        "folder_correct": bool(expected_in_model and predicted == expected_label),
    }


def _plot_spectra(save_path, folder_name, test_signals, wavenumbers, expected_label, predicted_label, train_mean_bank, bad_bands=()):
    """绘制测试均值和训练均值对照图"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_line(ax, y, **kwargs):
        label = kwargs.pop("label", None)
        diffs = np.diff(wavenumbers)
        step = float(np.median(np.abs(diffs))) if diffs.size else 0.0
        gap_idx = np.where(np.abs(diffs) > max(step * 3.0, 1e-6))[0]
        start = 0
        labeled = False
        for idx in gap_idx:
            end = idx + 1
            draw_kwargs = dict(kwargs)
            if label and not labeled:
                draw_kwargs["label"] = label
                labeled = True
            ax.plot(wavenumbers[start:end], y[start:end], **draw_kwargs)
            start = end
        draw_kwargs = dict(kwargs)
        if label and not labeled:
            draw_kwargs["label"] = label
        ax.plot(wavenumbers[start:], y[start:], **draw_kwargs)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (band_min, band_max) in enumerate(bad_bands):
        ax.axvspan(
            band_min,
            band_max,
            color="gray",
            alpha=0.18,
            label="Removed Bad Band" if idx == 0 else None,
        )
    for signal in test_signals:
        plot_line(ax, signal, color="#9ECAE1", alpha=0.38, linewidth=0.9)

    test_mean = test_signals.mean(axis=0)
    plot_line(ax, test_mean, color="#1F77B4", linewidth=2.0, label="Test Mean")

    expected_mean = train_mean_bank.get(expected_label)
    if expected_mean is not None:
        plot_line(ax, expected_mean, color="#E45756", linewidth=2.2, label=f"Train Mean ({expected_label})")

    predicted_mean = train_mean_bank.get(predicted_label)
    if predicted_label != expected_label and predicted_mean is not None:
        plot_line(
            ax,
            predicted_mean,
            color="#F28E2B",
            linewidth=2.0,
            linestyle="--",
            label=f"Predicted Mean ({predicted_label})",
        )

    ax.set_title(f"Spectrum Compare | {folder_name}")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Normalized Intensity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _write_summary(path, rows):
    """写出独立测试文件夹级汇总"""
    total = len(rows)
    correct = sum(1 for row in rows if row["folder_correct"])
    lines = [
        "===== TEST SUMMARY =====",
        "",
        f"Folders        : {total}",
        f"Folder correct : {correct}/{total} ({(correct / total * 100) if total else 0.0:.2f}%)",
        "",
        "\t".join(
            [
                "folder",
                "expected_label",
                "expected_in_model",
                "predicted_label",
                "majority_count",
                "total_count",
                "correct_count",
                "correct_ratio",
                "folder_correct",
            ]
        ),
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row["folder"],
                    row["expected_label"],
                    str(row["expected_in_model"]),
                    row["predicted_label"],
                    str(row["majority_count"]),
                    str(row["total_count"]),
                    str(row["correct_count"]),
                    f"{row['correct_ratio']:.6f}",
                    str(row["folder_correct"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_independent_test(
    exp_dir,
    level,
    test_root=None,
    folder=None,
    top_k=3,
    device=None,
    skip_transferred=False,
    transfer_manifest=None,
):
    """运行独立测试集推理并输出逐谱明细、谱线图和汇总报表"""
    import torch
    from raman.config_io import load_experiment
    from raman.infer.core import load_predictor, normalize_level_name

    level_name = normalize_level_name(level)
    exp_dir = Path(_resolve_project_path(exp_dir))
    config = load_experiment(exp_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_predictor(exp_dir, device, predict_level=level_name)
    class_names = predictor["runtime"].class_names_by_level.get(level_name, [])

    test_root = Path(_resolve_test_root(config, test_root))
    train_roots = _candidate_train_roots(config.dataset_root)
    expected_lookup = _build_expected_lookup(exp_dir, train_roots, level_name)
    train_mean_bank = _build_train_mean_bank(
        exp_dir,
        config.dataset_root,
        level_name,
        predictor["preprocessor"],
        config,
    )
    wavenumbers = _expected_wavenumbers(config)

    out_root = exp_dir / f"test_results_{level_name}"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    folders = iter_predict_folders(test_root, folder)
    if not folders:
        raise FileNotFoundError(f"No test folders found under {test_root}")

    summary_rows = []
    skipped_rows = []
    class_name_set = set(class_names)
    skip_lookup = _load_transfer_skip_lookup(transfer_manifest) if skip_transferred else {}
    for folder_path in folders:
        folder_path = Path(folder_path)
        expected_prefix = _test_folder_prefix(folder_path.name)
        expected_label = expected_lookup.get(expected_prefix)
        if expected_label not in class_name_set:
            print(
                f"[Skip] {folder_path.name}: expected label "
                f"{expected_label or expected_prefix} not in model classes"
            )
            continue

        predictions, signals, skipped = _predict_folder(folder_path, predictor, top_k, skip_lookup=skip_lookup)
        for item in skipped:
            skipped_rows.append(f"{folder_path.name}/{item}")
        if not predictions:
            if skipped:
                print(f"[Skip] {folder_path.name}: all spectra were transferred into training")
            continue

        row = _folder_summary(folder_path.name, expected_label, class_names, predictions)
        summary_rows.append(row)

        folder_out = out_root / folder_path.name
        folder_out.mkdir(parents=True, exist_ok=True)
        report_lines = format_prediction_report(folder_path.name, predictions, row)
        (folder_out / f"{folder_path.name}_file.txt").write_text("".join(report_lines), encoding="utf-8")

        _plot_spectra(
            folder_out / "spectra.png",
            folder_path.name,
            signals,
            wavenumbers,
            row["expected_label"],
            row["predicted_label"],
            train_mean_bank,
            get_config_bad_bands(config),
        )

    _write_summary(out_root / "summary.txt", summary_rows)
    if skipped_rows:
        (out_root / "skipped_transferred_samples.txt").write_text(
            "\n".join(skipped_rows) + "\n",
            encoding="utf-8",
        )
    print(f"[Saved] independent test results -> {out_root}")
    return out_root


def build_parser():
    """构建独立测试参数解析器"""
    parser = argparse.ArgumentParser(description="Run independent Raman test inference")
    parser.add_argument("--exp-dir", required=True, help="模型输出目录")
    parser.add_argument("--level", required=True, help="预测业务层级，例如 level_1、level1 或 1")
    parser.add_argument("--test-root", default=None, help="覆盖默认 test 目录")
    parser.add_argument("--folder", default=None, help="只运行某个测试文件夹")
    parser.add_argument("--top-k", type=int, default=3, help="逐谱输出 top-k 数量")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--skip-transferred", action="store_true", help="跳过已迁移进训练集的测试谱")
    parser.add_argument("--transfer-manifest", default="dataset/初始数据/test_transfer_manifest.csv", help="测试谱迁移 manifest")
    return parser


def main(argv=None):
    """执行独立测试命令"""
    args = build_parser().parse_args(argv)
    import torch

    device = torch.device("cpu") if args.cpu else None
    run_independent_test(
        args.exp_dir,
        args.level,
        test_root=args.test_root,
        folder=args.folder,
        top_k=args.top_k,
        device=device,
        skip_transferred=args.skip_transferred,
        transfer_manifest=args.transfer_manifest,
    )
    return 0
