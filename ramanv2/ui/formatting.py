"""UI 页面共用的数值展示格式。"""

from __future__ import annotations


def format_ratio(value: object, fallback: str) -> str:
    """把 0–1 比例格式化为百分比；空值或非数值返回 fallback。"""
    if value is None:
        return fallback
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return fallback


__all__ = ["format_ratio"]
