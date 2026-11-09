"""UI 页面共用的数值与路径展示格式。"""

from __future__ import annotations

from pathlib import Path

from ramanv2.common.paths import safe_relative_to


def format_ratio(value: object, fallback: str) -> str:
    """把 0–1 比例格式化为百分比；空值或非数值返回 fallback。"""
    if value is None:
        return fallback
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return fallback


def progress_ratio(done: int, total: int) -> float:
    """把处理进度换算为 0–1 比例；总量未知时视为完成。"""
    if total <= 0:
        return 1.0
    return min(max(done / total, 0.0), 1.0)


def relative_label(path: Path | str, base: Path | str) -> str:
    """优先显示相对路径；跨盘或越界时回退为绝对路径。"""
    relative = safe_relative_to(path, base)
    if relative is None:
        return str(Path(path).resolve())
    return str(relative)
