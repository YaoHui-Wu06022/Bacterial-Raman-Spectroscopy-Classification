"""跨模块复用的绘图布局辅助函数。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CHINESE_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "Noto Sans SC",
    "SimHei",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)

# 审图与幻灯片图共用的画布、配色和尺寸参数。
PLOT_BACKGROUND = "#FFFFFF"
PLOT_GRID_COLOR = "#F7F7F7"
PLOT_SPINE_COLOR = "#5F6368"
PLOT_SPINE_WIDTH = 2.0
PLOT_TICK_SIZE = 20.0
PLOT_LINE_COLOR = "#175DCC"
PLOT_BASELINE_COLOR = "#E07824"
PLOT_SMOOTH_COLOR = "#2F8F46"
PLOT_ZERO_COLOR = "#91979E"
PLOT_FAINT_COLOR = "#A8B0BA"
BAD_BAND_ALPHA = 0.35

FIGURE_SIZE_PREFIX = (12.0, 12.0)
FIGURE_DPI_PREFIX = 160
FIGURE_SIZE_LEAF = (10.0, 5.0)
FIGURE_DPI_LEAF = 300
FIGURE_DPI_SUMMARY = 200
FIGURE_SIZE_DECK = (12.0, 8.0)
FIGURE_DPI_DECK = 120

CATEGORICAL_PALETTE = (
    "#d60000",
    "#8c3bff",
    "#018700",
    "#00acc6",
    "#e6a500",
    "#ff7ed1",
    "#6b004f",
    "#573b00",
    "#005659",
    "#15e18c",
    "#0000dd",
    "#a17569",
    "#bcb6ff",
    "#bf03b8",
    "#645472",
    "#790000",
    "#0774d8",
    "#729a7c",
    "#ff7752",
    "#004b00",
    "#8e7b01",
    "#f2007b",
    "#8eba00",
    "#a57bb8",
    "#5901a3",
    "#e2afaf",
    "#a03a52",
    "#a1c8c8",
    "#9e4b00",
    "#546744",
    "#bac389",
    "#5e7b87",
)


def configure_matplotlib_fonts() -> str | None:
    """选择可用中文字体，避免导出的 Matplotlib 图片出现方框。"""
    from matplotlib import font_manager, rcParams

    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in CHINESE_FONT_CANDIDATES if name in available), None)
    if selected is not None:
        rcParams["font.family"] = [selected]
        rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    return selected


def apply_canvas(
    axis,
    *,
    facecolor: str = PLOT_BACKGROUND,
    grid_color: str | None = PLOT_GRID_COLOR,
    grid_alpha: float = 1.0,
    grid_linewidth: float = 0.8,
    spine_color: str = PLOT_SPINE_COLOR,
    spine_width: float = PLOT_SPINE_WIDTH,
    tick_size: float | None = PLOT_TICK_SIZE,
) -> None:
    """给单个子图应用统一画布：底色、网格、边框和刻度字号。"""
    axis.set_facecolor(facecolor)
    if grid_color is not None:
        axis.grid(True, color=grid_color, alpha=grid_alpha, linewidth=grid_linewidth)
    for spine in axis.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(spine_width)
    if tick_size is not None:
        axis.tick_params(labelsize=tick_size, width=2, length=6)


def save_figure(figure, path, *, dpi: float | None = None) -> None:
    """统一保存节奏：紧凑布局、固定底色，并关闭画布。"""
    from matplotlib import pyplot as plt

    figure.tight_layout()
    figure.savefig(path, dpi=dpi, facecolor=PLOT_BACKGROUND)
    plt.close(figure)


def break_bad_bands(
    axis: np.ndarray,
    values: np.ndarray,
    bad_bands,
) -> tuple[np.ndarray, np.ndarray]:
    """把坏段内部和两侧边界置为 NaN，保证任何轴密度下都不会跨坏区连线。"""
    merged_axis = np.asarray(axis, dtype=np.float64)
    merged_values = np.asarray(values, dtype=np.float64).copy()
    bands = [band for band in (bad_bands or ())]
    if not bands:
        return merged_axis, merged_values
    mask = np.ones(merged_axis.shape, dtype=bool)
    edges = []
    for lower, upper in bands:
        mask &= ~((merged_axis >= lower) & (merged_axis <= upper))
        edges.extend((float(lower), float(upper)))
    merged_values[~mask] = np.nan
    merged_axis = np.union1d(merged_axis, np.asarray(edges, dtype=np.float64))
    reindexed = np.full(merged_axis.shape, np.nan, dtype=np.float64)
    reindexed[np.searchsorted(merged_axis, axis)] = merged_values
    return merged_axis, reindexed


def plot_with_bad_band_breaks(axis, x, y, bad_bands, **kwargs) -> None:
    """绘制坏区两侧断开的曲线，避免跨越无效区连线。"""
    broken_x, broken_y = break_bad_bands(x, y, bad_bands)
    axis.plot(broken_x, broken_y, **kwargs)


def add_bad_band_spans(
    axis,
    bad_bands,
    *,
    alpha: float = BAD_BAND_ALPHA,
    label: str | None = None,
    zorder: int | None = None,
) -> None:
    """在图中标记已经规范化的坏波段。"""
    for index, (lower, upper) in enumerate(bad_bands):
        options = {"color": "gray", "alpha": alpha}
        if label is not None and index == 0:
            options["label"] = label
        if zorder is not None:
            options["zorder"] = zorder
        axis.axvspan(lower, upper, **options)


def shorten_class_names(class_names: Sequence[str]) -> list[str]:
    """提取层级类别路径的末级名称，用于紧凑显示坐标轴标签。"""
    return [_shorten_class_name(name) for name in class_names]


def _shorten_class_name(class_name: str) -> str:
    """将 Windows 或 POSIX 风格的层级路径压缩为末级名称。"""
    text = str(class_name).replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    return parts[-1] if parts else text


def resolve_confusion_matrix_figsize(class_names: Sequence[str]) -> tuple[float, float]:
    """按类别数和标签长度计算混淆矩阵的合适画布尺寸。"""
    class_count = max(len(class_names), 1)
    max_name_length = max((len(str(name)) for name in class_names), default=0)
    cell_size = 0.62
    label_padding = min(max_name_length, 24) * 0.06
    width = 2.3 + class_count * cell_size + label_padding
    height = 2.3 + class_count * cell_size
    return min(max(width, 6.0), 38.0), min(max(height, 5.6), 38.0)


def resolve_confusion_matrix_left_margin(class_names: Sequence[str]) -> float:
    """按纵轴标签长度计算左侧留白，避免标签被图片裁切。"""
    max_name_length = max((len(str(name)) for name in class_names), default=0)
    margin = 0.115 + min(max_name_length, 28) * 0.006
    return min(max(margin, 0.18), 0.34)


def resolve_confusion_matrix_font_sizes(class_count: int) -> tuple[int, int]:
    """按类别数返回单元格标注和坐标轴标签的字号。"""
    if class_count <= 12:
        return 11, 12
    if class_count <= 24:
        return 9, 11
    if class_count <= 36:
        return 8, 10
    return 7, 9
