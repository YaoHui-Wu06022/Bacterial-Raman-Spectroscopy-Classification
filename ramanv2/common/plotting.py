"""跨模块复用的绘图布局辅助函数。"""

from __future__ import annotations

from collections.abc import Sequence


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
