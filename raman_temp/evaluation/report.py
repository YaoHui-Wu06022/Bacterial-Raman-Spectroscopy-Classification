"""分类评估产物写入。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from raman_temp.common.metrics import compute_classification_metrics


def write_model_report(
    result_dir: Path | str,
    class_names: Sequence[str],
    paths: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
) -> Path:
    """写入模型评估的逐样本结果、指标报告和混淆矩阵。"""
    target_dir = Path(result_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_classification_outputs(
        target_dir,
        class_names,
        paths,
        labels,
        predictions,
        report_name="classification_report.txt",
        matrix_name="confusion_matrix_raw.csv",
        sample_name="val_eval_results.csv",
    )
    return target_dir


def write_baseline_report(
    result_dir: Path | str,
    class_names: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
) -> Path:
    """写入 PCA-SVM 的指标文本和混淆矩阵。"""
    target_dir = Path(result_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_classification_outputs(
        target_dir,
        class_names,
        (),
        labels,
        predictions,
        report_name="metrics.txt",
        matrix_name="confusion_matrix.csv",
        sample_name=None,
    )
    return target_dir


def write_pca_scatter(
    result_dir: Path | str,
    features: np.ndarray | None,
    labels: np.ndarray | None,
    class_names: Sequence[str],
) -> Path | None:
    """在 PCA 至少保留两个成分时写入训练集散点图。"""
    if features is None or labels is None or features.shape[1] < 2:
        return None
    path = Path(result_dir) / "pca_scatter.png"
    figure = plt.figure(figsize=(8, 6))
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if mask.any():
            plt.scatter(features[mask, 0], features[mask, 1], s=12, alpha=0.6, label=class_name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA scatter (train only)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _write_classification_outputs(
    result_dir: Path,
    class_names: Sequence[str],
    paths: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    report_name: str,
    matrix_name: str,
    sample_name: str | None,
) -> None:
    """以统一指标内核生成一个分类结果目录。"""
    label_values = np.asarray(labels, dtype=np.int64)
    prediction_values = np.asarray(predictions, dtype=np.int64)
    if label_values.size == 0:
        raise RuntimeError("评估后没有有效验证样本")
    class_ids = list(range(len(class_names)))
    metrics = compute_classification_metrics(label_values, prediction_values, class_ids)
    report = classification_report(
        label_values,
        prediction_values,
        labels=class_ids,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    (result_dir / report_name).write_text(
        _format_report(report, class_names, metrics), encoding="utf-8"
    )
    matrix = confusion_matrix(label_values, prediction_values, labels=class_ids)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(result_dir / matrix_name)
    _write_confusion_matrix(result_dir / "confusion_matrix.png", matrix, class_names)
    if sample_name is not None:
        pd.DataFrame(
            {"path": list(paths), "label_true": label_values, "label_pred": prediction_values}
        ).to_csv(result_dir / sample_name, index=False)


def _format_report(report: dict, class_names: Sequence[str], metrics: dict[str, float]) -> str:
    """将 sklearn 报告整理为固定的可读文本格式。"""
    lines = [f"{'':<20}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}", ""]
    for name in class_names:
        row = report[name]
        lines.append(
            f"{name:<20}{row['precision'] * 100:>11.4f}%{row['recall'] * 100:>11.4f}%"
            f"{row['f1-score'] * 100:>11.4f}%{int(round(row['support'])):>12d}"
        )
    support = int(sum(report[name]["support"] for name in class_names))
    lines.extend(
        [
            "",
            f"{'summary metric':<20}{'value':>12}{'support':>12}",
            f"{'Accuracy':<20}{metrics['accuracy'] * 100:>11.4f}%{support:>12d}",
            f"{'Macro F1-score':<20}{metrics['macro_f1'] * 100:>11.4f}%{support:>12d}",
            f"{'Macro Recall':<20}{metrics['macro_recall'] * 100:>11.4f}%{support:>12d}",
        ]
    )
    return "\n".join(lines)


def _write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    """写入同时标注行归一化比例和计数的混淆矩阵图。"""
    divisor = matrix.sum(axis=1, keepdims=True).astype(np.float32)
    divisor[divisor == 0] = 1.0
    normalized = matrix.astype(np.float32) / divisor
    annotations = np.array(
        [
            ["0\n(0)" if value == 0 else f"{normalized[row, col] * 100:.1f}%\n({value})" for col, value in enumerate(line)]
            for row, line in enumerate(matrix)
        ],
        dtype=object,
    )
    figure = plt.figure(figsize=(max(7, len(class_names) * 0.55), max(6, len(class_names) * 0.55)))
    axis = sns.heatmap(normalized, cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot=annotations, fmt="", square=True)
    axis.tick_params(axis="x", rotation=45, labelsize=7)
    axis.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
