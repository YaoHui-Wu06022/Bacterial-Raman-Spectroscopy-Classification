"""独立推理的文本、图像和运行清单产物。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ramanv2.spectra.bands import build_valid_mask


def write_file_report(
    output_path: Path | str,
    folder_name: str,
    predictions: list[dict[str, Any]],
    summary: Mapping[str, Any] | None,
) -> None:
    """写入一个文件夹的逐谱 top-k 预测文本。"""
    counts: dict[str, int] = {}
    for item in predictions:
        counts[item["top1_label"]] = counts.get(item["top1_label"], 0) + 1
    lines: list[str] = []
    if summary is not None:
        lines.extend(
            [
                "===== FOLDER SUMMARY =====\n\n",
                f"Expected label      : {summary['expected_label']}\n",
                f"Expected in model   : {summary['expected_in_model']}\n",
                f"Majority prediction : {summary['predicted_label']}\n",
                "Correct spectra     : "
                f"{summary['correct_count']}/{summary['total_count']} "
                f"({summary['correct_ratio'] * 100:.2f}%)\n",
                f"Folder correct      : {summary['folder_correct']}\n",
                "\n===============================================\n\n",
            ]
        )
    lines.append("===== FILE-LEVEL SUMMARY =====\n\n")
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"{label:20s} : {count}\n")
    lines.append("\n===============================================\n\n")
    for item in predictions:
        lines.append(f"########## File: {item['file']} ##########\n")
        top_prediction = item["predictions"][0]
        lines.append(
            f"Top-1 -> {top_prediction['label']} ({top_prediction['probability'] * 100:.2f}%)\n"
        )
        lines.append("Top-k predictions:\n")
        for index, prediction in enumerate(item["predictions"], 1):
            lines.append(
                f"   {index}) {prediction['label']:20s}  {prediction['probability'] * 100:.2f}%\n"
            )
        lines.append("\n===============================================\n\n")
    Path(output_path).write_text("".join(lines), encoding="utf-8")


def write_summary_report(
    output_path: Path | str,
    rows: list[Mapping[str, Any]],
    evaluate_enable: bool,
) -> None:
    """写入所有文件夹的独立推理汇总文本。"""
    correct_count = sum(bool(row["folder_correct"]) for row in rows) if evaluate_enable else 0
    lines = [
        "===== TEST SUMMARY =====",
        "",
        f"Folders        : {len(rows)}",
        (
            "Folder correct : "
            f"{correct_count}/{len(rows)} "
            f"({correct_count / len(rows) * 100 if rows else 0.0:.2f}%)"
            if evaluate_enable
            else "Evaluation     : disabled"
        ),
        "",
    ]
    columns = (
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
        if evaluate_enable
        else ["folder", "predicted_label", "majority_count", "total_count"]
    )
    lines.append("\t".join(columns))
    for row in rows:
        values = [
            f"{float(row[column]):.6f}" if column == "correct_ratio" else str(row[column])
            for column in columns
        ]
        lines.append("\t".join(values))
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_used_runs(
    output_path: Path | str,
    predictor_mode: str,
    target_level: str,
    runs: Mapping[str, Any],
) -> None:
    """写入本次推理可使用的模型 run 清单。"""
    payload = {
        "mode": predictor_mode,
        "target_level": target_level,
        "runs": runs,
    }
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_folder_spectra(
    output_path: Path | str,
    folder_name: str,
    signals: np.ndarray,
    wavenumbers: np.ndarray,
    bad_bands: tuple[tuple[float, float], ...],
    expected_label: str | None,
    predicted_label: str,
    train_mean_bank: Mapping[str, np.ndarray],
) -> None:
    """绘制测试谱均值及可选训练类别均值对照图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for lower, upper in bad_bands:
        axis.axvspan(lower, upper, color="#d9d9d9", alpha=0.35)
    for signal in signals:
        axis.plot(wavenumbers, signal, color="#9ECAE1", alpha=0.38, linewidth=0.9)
    axis.plot(wavenumbers, signals.mean(axis=0), color="#1F77B4", linewidth=2.0, label="Test Mean")
    _plot_train_mean(axis, wavenumbers, train_mean_bank, expected_label, "#E45756", "Train Mean")
    if predicted_label != expected_label:
        _plot_train_mean(axis, wavenumbers, train_mean_bank, predicted_label, "#F28E2B", "Predicted Mean")
    axis.set_title(f"Spectrum Compare | {folder_name}")
    axis.set_xlabel("Wavenumber")
    axis.set_ylabel("Normalized Intensity")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def build_plot_axis(
    cut_min: float,
    cut_max: float,
    point_count: int,
    bad_bands: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """构建与模型输入强度长度匹配的绘图波数轴。"""
    full_axis = np.linspace(cut_min, cut_max, point_count, dtype=np.float32)
    valid_mask = build_valid_mask(full_axis, bad_bands)
    return full_axis if valid_mask is None else full_axis[valid_mask]


def _plot_train_mean(
    axis,
    wavenumbers: np.ndarray,
    train_mean_bank: Mapping[str, np.ndarray],
    label: str | None,
    color: str,
    title: str,
) -> None:
    """按类别存在性绘制一条训练均值对照曲线。"""
    if label is None:
        return
    values = train_mean_bank.get(label)
    if values is not None:
        axis.plot(
            wavenumbers,
            values,
            color=color,
            linewidth=2.0,
            label=f"{title} ({label})",
        )
