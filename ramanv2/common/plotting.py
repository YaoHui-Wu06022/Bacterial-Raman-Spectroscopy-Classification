"""跨模块复用的绘图布局辅助函数。"""

from __future__ import annotations

from collections.abc import Sequence


CHINESE_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "Noto Sans SC",
    "SimHei",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
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


def add_bad_band_spans(
    axis,
    bad_bands,
    *,
    alpha: float = 0.2,
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
