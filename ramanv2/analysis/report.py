"""analysis 图、CSV、日志与多任务聚合产物写入。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from ramanv2.common.plotting import shorten_class_names
from ramanv2.spectra.axis import estimate_gap_indices
from ramanv2.spectra.bands import normalize_bad_bands


def save_task_reports(
    figure_dir: Path,
    class_names: tuple[str, ...],
    channel: np.ndarray,
    band: np.ndarray,
    mean_spectra: np.ndarray,
    layer_scores: dict[str, float],
    channel_names: tuple[str, ...],
    wavenumbers: np.ndarray,
    bad_bands,
    row_norm: str,
    separate_class_plots_enable: bool,
) -> None:
    """写入一个模型任务的通道、层和类别波段产物。"""
    save_channel_importance_plot(
        figure_dir / "channel_importance_IG.png",
        channel_names,
        channel,
        aggregate_enable=True,
    )
    save_layer_attribution_plot(
        figure_dir / "layer_importance.png",
        tuple(layer_scores),
        tuple(layer_scores.values()),
        aggregate_enable=False,
    )
    write_band_csv(
        figure_dir / "band_importance_per_class.csv",
        class_names,
        band,
        wavenumbers,
        row_norm,
    )
    save_band_heatmap(
        figure_dir / "band_importance_heatmap.png",
        class_names,
        band,
        mean_spectra,
        row_norm,
    )
    if separate_class_plots_enable:
        save_class_band_heatmaps(
            figure_dir,
            "band_importance_heatmap",
            class_names,
            band,
            mean_spectra,
            wavenumbers,
            bad_bands,
            row_norm,
        )


def write_aggregate_reports(
    summaries,
    figure_dir: Path,
    channel_names: tuple[str, ...],
    wavenumbers: np.ndarray,
    bad_bands,
    row_norm: str,
    separate_class_plots_enable: bool,
) -> None:
    """按分析样本量聚合多个 parent 模型的归因结果。"""
    if len(summaries) <= 1:
        return
    weights = np.asarray([item["weight"] for item in summaries], dtype=float)
    channel = sum(item["channel"] * weight for item, weight in zip(summaries, weights)) / weights.sum()
    save_channel_importance_plot(
        figure_dir / "channel_importance_IG_aggregate.png",
        channel_names,
        channel,
        aggregate_enable=True,
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
    write_band_csv(
        figure_dir / "band_importance_per_class_aggregate.csv",
        names,
        band,
        wavenumbers,
        row_norm,
    )
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
    if separate_class_plots_enable:
        save_class_band_heatmaps(
            figure_dir,
            "band_importance_heatmap_aggregate",
            names,
            band,
            mean_spectra,
            wavenumbers,
            bad_bands,
            row_norm,
        )
    save_layer_attribution_plot(
        figure_dir / "layer_importance_aggregate.png",
        tuple(layers),
        tuple(layers.values()),
        aggregate_enable=True,
    )


def save_channel_importance_plot(
    output_path: Path,
    channel_names: tuple[str, ...],
    values: np.ndarray,
    aggregate_enable: bool,
) -> None:
    """保存输入通道相对归因柱状图。"""
    figure, axis = plt.subplots(figsize=(7, 4))
    names = channel_names[: len(values)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    axis.bar(names, values, color=colors)
    title = (
        "Input Channel Contribution (Aggregated)"
        if aggregate_enable
        else "Input Channel Contribution (Integrated Gradients)"
    )
    axis.set_title(title)
    axis.set_ylabel("Relative Importance")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_layer_attribution_plot(
    output_path: Path,
    names: tuple[str, ...],
    values: tuple[float, ...],
    aggregate_enable: bool,
) -> None:
    """保存 activation × gradient 的层归因柱状图。"""
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, values)
    ylabel = (
        "Layer Importance (Aggregated |A x G|)"
        if aggregate_enable
        else "Layer Importance (Normalized |A × G|)"
    )
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=60)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def write_band_csv(
    output_path: Path,
    class_names,
    values: np.ndarray,
    wavenumbers: np.ndarray,
    row_norm: str,
) -> None:
    """以类别、波段索引、波数和归因值保存逐点 IG 结果。"""
    normalized = normalize_band_importance(values, row_norm)
    if len(wavenumbers) != normalized.shape[1]:
        wavenumbers = np.arange(normalized.shape[1])
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "band_index", "wavenumber", "importance"])
        for name, row in zip(class_names, normalized):
            writer.writerows(
                (name, index, float(wavenumbers[index]), float(value))
                for index, value in enumerate(row)
            )


def normalize_band_importance(values: np.ndarray, row_norm: str) -> np.ndarray:
    """按每个类别行归一化波段归因，用于图和 CSV 的一致输出。"""
    normalized = values.copy()
    if row_norm == "max":
        normalized /= np.maximum(normalized.max(axis=1, keepdims=True), 1e-8)
    elif row_norm == "sum":
        normalized /= np.maximum(normalized.sum(axis=1, keepdims=True), 1e-8)
    elif row_norm != "none":
        raise ValueError(f"未知的 row_norm：{row_norm}")
    return normalized


def save_band_heatmap(
    output_path: Path,
    class_names,
    importance: np.ndarray,
    mean_spectra: np.ndarray,
    row_norm: str,
) -> None:
    """保留一张按类别排列的整体波段归因概览图。"""
    values = normalize_band_importance(importance, row_norm)
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


def save_class_band_heatmaps(
    figure_dir: Path,
    file_prefix: str,
    class_names,
    importance: np.ndarray,
    mean_spectra: np.ndarray,
    wavenumbers: np.ndarray,
    bad_bands,
    row_norm: str,
) -> None:
    """按类别保存带波数轴、坏区和颜色归因填充的单谱图。"""
    values = normalize_band_importance(importance, row_norm)
    if len(wavenumbers) != values.shape[1]:
        wavenumbers = np.arange(values.shape[1])
        bad_bands = ()
    for class_name, row, spectrum in zip(class_names, values, mean_spectra):
        output_path = figure_dir / f"{file_prefix}__{_build_class_plot_name(class_name)}.png"
        save_class_band_heatmap(
            output_path,
            class_name,
            row,
            spectrum,
            wavenumbers,
            bad_bands,
        )


def save_class_band_heatmap(
    output_path: Path,
    class_name: str,
    importance: np.ndarray,
    mean_spectrum: np.ndarray,
    wavenumbers: np.ndarray,
    bad_bands,
) -> None:
    """保存单类别的平均光谱和 Integrated Gradients 波段填充图。"""
    if len(importance) != len(mean_spectrum) or len(importance) != len(wavenumbers):
        raise ValueError("波段归因、平均光谱与波数轴长度必须一致")
    display_name = shorten_class_names((class_name,))[0]
    normalized_spectrum = _normalize_spectrum(mean_spectrum)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    _add_bad_band_spans(axis, bad_bands)
    _add_colored_band_fill(axis, wavenumbers, normalized_spectrum, importance)
    _plot_spectrum_segments(axis, wavenumbers, normalized_spectrum)
    axis.set_xlim(wavenumbers[0], wavenumbers[-1])
    axis.set_ylim(-0.05, 0.85)
    tick_indices = np.linspace(0, len(wavenumbers) - 1, 6, dtype=int)
    axis.set_xticks(wavenumbers[tick_indices])
    axis.set_xticklabels([f"{wavenumbers[index]:.0f}" for index in tick_indices])
    axis.set_xlabel("Wavenumber")
    axis.set_title(f"Band Importance - {display_name}")
    color_map = plt.get_cmap("Blues")
    scalar_map = plt.cm.ScalarMappable(
        cmap=color_map,
        norm=plt.Normalize(vmin=0.0, vmax=1.0),
    )
    scalar_map.set_array([])
    colorbar = figure.colorbar(
        scalar_map,
        ax=axis,
        fraction=0.018,
        pad=0.025,
        aspect=35,
        label="Importance (normalized)",
    )
    colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.075, right=0.94, top=0.90, bottom=0.16)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _build_class_plot_name(class_name: str) -> str:
    """将层级类别名转为稳定的单图文件名片段。"""
    parts = str(class_name).replace("\\", "/").split("/")
    return "__".join(part for part in parts if part)


def _normalize_spectrum(values: np.ndarray) -> np.ndarray:
    """将单条平均光谱压缩到零至一的绘图范围。"""
    lower = float(np.min(values))
    upper = float(np.max(values))
    return np.zeros_like(values) if upper - lower < 1e-8 else (values - lower) / (upper - lower)


def _add_bad_band_spans(axis, bad_bands) -> None:
    """在单类别归因图中以灰色区域标记被排除的坏波段。"""
    for lower, upper in normalize_bad_bands(bad_bands):
        axis.axvspan(lower, upper, color="gray", alpha=0.15, zorder=0)


def _add_colored_band_fill(
    axis,
    wavenumbers: np.ndarray,
    spectrum: np.ndarray,
    importance: np.ndarray,
) -> None:
    """按相邻波段的归因强度为光谱下方区域逐段着色。"""
    gap_indices = set(estimate_gap_indices(wavenumbers))
    color_map = plt.get_cmap("Blues")
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    polygons = []
    colors = []
    for index in range(len(wavenumbers) - 1):
        if index in gap_indices:
            continue
        polygons.append(
            [
                (wavenumbers[index], 0.0),
                (wavenumbers[index], spectrum[index] * 0.8),
                (wavenumbers[index + 1], spectrum[index + 1] * 0.8),
                (wavenumbers[index + 1], 0.0),
            ]
        )
        colors.append(color_map(color_norm((importance[index] + importance[index + 1]) * 0.5)))
    axis.add_collection(
        PolyCollection(polygons, facecolors=colors, edgecolors="none", alpha=0.9)
    )


def _plot_spectrum_segments(axis, wavenumbers: np.ndarray, spectrum: np.ndarray) -> None:
    """在有效波段分段绘制平均光谱，避免跨越坏区连线。"""
    start = 0
    for gap_index in estimate_gap_indices(wavenumbers):
        end = gap_index + 1
        if end - start > 1:
            axis.plot(wavenumbers[start:end], spectrum[start:end] * 0.8, color="#1f1f1f", linewidth=1.0)
        start = end
    if len(wavenumbers) - start > 1:
        axis.plot(wavenumbers[start:], spectrum[start:] * 0.8, color="#1f1f1f", linewidth=1.0)
