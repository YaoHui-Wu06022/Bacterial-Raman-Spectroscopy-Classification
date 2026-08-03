"""analysis 图、CSV、日志与多任务聚合产物写入。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_task_reports(
    figure_dir: Path,
    tag: str,
    class_names: tuple[str, ...],
    channel: np.ndarray,
    band: np.ndarray,
    mean_spectra: np.ndarray,
    layer_scores: dict[str, float],
    row_norm: str,
) -> None:
    """写入一个模型任务的通道、层和类别波段产物。"""
    channel_names = ("main", "smooth", "d1")[: len(channel)]
    save_bar_plot(figure_dir / f"channel_importance_IG_{tag}.png", channel_names, channel, "Input Channel Importance")
    save_bar_plot(figure_dir / f"layer_attribution_{tag}.png", tuple(layer_scores), tuple(layer_scores.values()), "Layer Attribution")
    write_band_csv(figure_dir / f"band_importance_per_class_{tag}.csv", class_names, band)
    save_band_heatmap(
        figure_dir / f"band_importance_heatmap_{tag}.png",
        class_names,
        band,
        mean_spectra,
        row_norm,
    )


def write_aggregate_reports(summaries, figure_dir: Path, row_norm: str) -> None:
    """按分析样本量聚合多个 parent 模型的归因结果。"""
    if len(summaries) <= 1:
        return
    weights = np.asarray([item["weight"] for item in summaries], dtype=float)
    channel = sum(item["channel"] * weight for item, weight in zip(summaries, weights)) / weights.sum()
    save_bar_plot(
        figure_dir / "channel_importance_IG_aggregate.png",
        ("main", "smooth", "d1")[: len(channel)],
        channel,
        "Input Channel Importance",
    )
    class_ids = sorted({class_id for item in summaries for class_id in item["class_ids"]})
    id_to_row = {class_id: row for row, class_id in enumerate(class_ids)}
    band = np.zeros((len(class_ids), summaries[0]["band"].shape[1]), dtype=float)
    counts = np.zeros(len(class_ids), dtype=np.int64)
    mean_spectra = np.zeros_like(band)
    mean_counts = np.zeros(len(class_ids), dtype=np.int64)
    name_by_id = {
        class_id: name
        for item in summaries
        for class_id, name in zip(item["class_ids"], item["class_names"])
    }
    for item in summaries:
        for local_row, class_id in enumerate(item["class_ids"]):
            row = id_to_row[class_id]
            count = item["counts"][local_row]
            band[row] += item["band"][local_row] * count
            counts[row] += count
            mean_count = item["mean_counts"][local_row]
            mean_spectra[row] += item["mean"][local_row] * mean_count
            mean_counts[row] += mean_count
    band /= np.maximum(counts[:, None], 1)
    mean_spectra /= np.maximum(mean_counts[:, None], 1)
    names = tuple(name_by_id[class_id] for class_id in class_ids)
    write_band_csv(figure_dir / "band_importance_per_class_aggregate.csv", names, band)
    save_band_heatmap(
        figure_dir / "band_importance_heatmap_aggregate.png",
        names,
        band,
        mean_spectra,
        row_norm,
    )
    layers: dict[str, float] = {}
    for item, weight in zip(summaries, weights):
        for name, value in item["layer"].items():
            layers[name] = layers.get(name, 0.0) + value * weight
    total = sum(layers.values())
    if total:
        layers = {name: value / total for name, value in layers.items()}
    save_bar_plot(
        figure_dir / "layer_attribution_aggregate.png",
        tuple(layers),
        tuple(layers.values()),
        "Layer Attribution",
    )


def save_bar_plot(output_path: Path, names, values, title: str) -> None:
    """保存一个简单的类别或层级柱状图。"""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(names, values)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def write_band_csv(output_path: Path, class_names, values: np.ndarray) -> None:
    """以类别、波段索引、归因值三列保存逐点 IG 结果。"""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "band_index", "importance"])
        for name, row in zip(class_names, values):
            writer.writerows((name, index, float(value)) for index, value in enumerate(row))


def save_band_heatmap(
    output_path: Path,
    class_names,
    importance: np.ndarray,
    mean_spectra: np.ndarray,
    row_norm: str,
) -> None:
    """绘制按类别排列的平均谱与归因强度图。"""
    values = importance.copy()
    if row_norm == "max":
        values /= np.maximum(values.max(axis=1, keepdims=True), 1e-8)
    elif row_norm == "sum":
        values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
    figure, axes = plt.subplots(
        len(class_names),
        1,
        figsize=(12, max(3, len(class_names) * 1.4)),
        squeeze=False,
    )
    for row, axis in enumerate(axes[:, 0]):
        spectrum = mean_spectra[row]
        normalized = (spectrum - spectrum.min()) / max(float(spectrum.max() - spectrum.min()), 1e-8)
        axis.plot(normalized, color="black", linewidth=0.8)
        axis.fill_between(
            np.arange(len(normalized)),
            0,
            normalized,
            color=plt.cm.Blues(float(values[row].mean())),
            alpha=0.8,
        )
        axis.set_ylabel(class_names[row], fontsize=7)
    axes[-1, 0].set_xlabel("Band Index")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
