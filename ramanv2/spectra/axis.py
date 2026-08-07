"""波数轴构造、步长估计与绘图断点识别。"""

from __future__ import annotations

import numpy as np

from .bands import build_valid_mask, get_config_bad_bands


def build_wn_ref(cut_min, cut_max, target_points):
    """生成裁剪和插值共同使用的均匀波数轴。"""
    return np.linspace(cut_min, cut_max, target_points)


def median_step_cm(wavenumbers) -> float:
    """估计波数轴中有限、非零相邻步长的中位数。"""
    values = np.asarray(wavenumbers, dtype=np.float32)
    if values.size < 2:
        return 1.0
    differences = np.abs(np.diff(values))
    differences = differences[np.isfinite(differences) & (differences > 1e-8)]
    return float(np.median(differences)) if differences.size else 1.0


def build_wavenumber_axis(length: int, config):
    """按输入配置构造与模型输入长度对齐的有效波数轴。"""
    bad_bands = get_config_bad_bands(config)
    if hasattr(config, "cut_min") and hasattr(config, "cut_max"):
        if hasattr(config, "target_points"):
            try:
                target_points = int(config.target_points)
            except (TypeError, ValueError):
                target_points = None
            if target_points:
                full_axis = np.linspace(config.cut_min, config.cut_max, target_points)
                mask = build_valid_mask(full_axis, bad_bands)
                if mask is not None:
                    full_axis = full_axis[mask]
                if full_axis.shape[0] == length:
                    return full_axis
        if hasattr(config, "step_cm"):
            return config.cut_min + config.step_cm * np.arange(length)
        return np.linspace(config.cut_min, config.cut_max, length)
    return np.arange(length)


def expected_wavenumbers(config):
    """按配置生成严格匹配模型有效输入长度的波数轴。"""
    axis = np.linspace(float(config.cut_min), float(config.cut_max), int(config.target_points))
    mask = build_valid_mask(axis, get_config_bad_bands(config))
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
