"""波数轴构造、步长估计与绘图断点识别。"""

from __future__ import annotations

import numpy as np

from .bands import build_valid_mask


def build_wn_ref(cut_min, cut_max, target_points):
    """生成裁剪和插值共同使用的均匀波数轴。"""
    return np.linspace(cut_min, cut_max, target_points)


def expected_wavenumbers(config):
    """按配置生成严格匹配模型有效输入长度的波数轴。"""
    axis = np.linspace(float(config.cut_min), float(config.cut_max), int(config.target_points))
    mask = build_valid_mask(axis, config.bad_bands)
    return axis[mask] if mask is not None else axis


def estimate_gap_indices(wavenumbers, gap_factor: float = 1.5) -> list[int]:
    """返回相邻步长明显增大前的索引，供绘图在缺失波段处断线。"""
    values = np.asarray(wavenumbers, dtype=np.float32)
    if values.size < 2:
        return []
    steps = np.diff(values)
    positive = steps[steps > 0]
    if positive.size == 0:
        return []
    median_step = float(np.median(positive))
    if median_step <= 0:
        return []
    return np.where(steps > median_step * float(gap_factor))[0].tolist()
