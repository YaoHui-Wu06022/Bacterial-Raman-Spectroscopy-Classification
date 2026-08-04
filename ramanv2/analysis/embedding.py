"""Train/Val embedding 收集与 UMAP 输出。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


def save_train_val_umap(
    model: torch.nn.Module,
    train_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    validation_labels: torch.Tensor,
    class_names: tuple[str, ...],
    output_path: Path,
    neighbors: int,
    min_dist: float,
    seed: int,
) -> None:
    """提取 Train/Val embedding 并保存联合 UMAP 散点图。"""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP 分析需要安装 umap-learn") from exc
    with torch.no_grad():
        _train_logits, train_embeddings = model(train_inputs, return_embedding_enable=True)
        _validation_logits, validation_embeddings = model(
            validation_inputs,
            return_embedding_enable=True,
        )
    arrays = [train_embeddings.cpu().numpy(), validation_embeddings.cpu().numpy()]
    labels = np.concatenate(
        [
            train_labels.detach().cpu().numpy(),
            validation_labels.detach().cpu().numpy(),
        ]
    )
    splits = np.asarray([0] * len(arrays[0]) + [1] * len(arrays[1]))
    points = umap.UMAP(
        n_neighbors=min(neighbors, max(2, len(splits) - 1)),
        min_dist=min_dist,
        random_state=seed,
    ).fit_transform(np.vstack(arrays))
    _save_split_umap(points, labels, splits, class_names, output_path)


def _save_split_umap(
    points: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """按 Train/Val 双面板保存类别着色的 UMAP 图。"""
    class_count = len(class_names)
    color_map = _resolve_umap_color_map(class_count)
    boundaries = np.arange(class_count + 1) - 0.5
    color_norm = colors.BoundaryNorm(boundaries, color_map.N)
    scalar_map = ScalarMappable(cmap=color_map, norm=color_norm)
    scalar_map.set_array([])
    x_limits, y_limits = _resolve_umap_limits(points)
    legend_columns, legend_font_size, right_margin = _resolve_umap_legend_layout(class_count)
    legend_rows = int(np.ceil(max(class_count, 1) / legend_columns))
    figure_height = max(6.0, min(14.0, 0.32 * legend_rows + 1.8))
    figure, axes = plt.subplots(1, 2, figsize=(14, figure_height), sharex=True, sharey=True)
    for axis, title, split_id in zip(axes, ("Train", "Val"), (0, 1)):
        mask = splits == split_id
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            c=labels[mask],
            cmap=color_map,
            norm=color_norm,
            marker="o",
            s=18,
            alpha=0.85,
            edgecolors="none",
        )
        axis.set_title(title)
        axis.set_xlabel("UMAP-1")
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
    axes[0].set_ylabel("UMAP-2")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=scalar_map.to_rgba(class_id),
            markeredgecolor="none",
            markersize=6,
            label=class_name,
        )
        for class_id, class_name in enumerate(class_names)
    ]
    figure.subplots_adjust(left=0.055, right=right_margin, bottom=0.10, top=0.90, wspace=0.08)
    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(right_margin + 0.01, 0.5),
        ncol=legend_columns,
        fontsize=legend_font_size,
        frameon=False,
    )
    figure.savefig(output_path, dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _resolve_umap_color_map(class_count: int):
    """按类别数选择离散 UMAP 颜色表。"""
    if class_count <= 10:
        return plt.get_cmap("tab10")
    if class_count <= 20:
        return plt.get_cmap("tab20")
    return plt.get_cmap("hsv")


def _resolve_umap_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """为两个 UMAP 面板计算一致的坐标范围。"""
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_padding = max((x_max - x_min) * 0.05, 1e-3)
    y_padding = max((y_max - y_min) * 0.05, 1e-3)
    return (x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding)


def _resolve_umap_legend_layout(class_count: int) -> tuple[int, int, float]:
    """按类别数确定右侧 UMAP 图例的列数、字号与留白。"""
    if class_count <= 12:
        return 1, 8, 0.84
    if class_count <= 30:
        return 2, 7, 0.78
    return 3, 6, 0.70
