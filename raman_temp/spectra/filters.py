"""一维窗口滤波与 Savitzky-Golay 系数计算。"""

from __future__ import annotations

import math

import numpy as np


def odd_window_points(points, min_points: int = 3) -> int:
    """将窗口点数规范为不小于最小值的奇数。"""
    value = max(int(min_points), int(round(float(points))))
    return value + 1 if value % 2 == 0 else value


def median_filter_1d(values, window):
    """使用边缘复制的一维中值滤波，并保持原数组长度。"""
    array = np.asarray(values)
    window = odd_window_points(window)
    if array.size < window:
        return array.copy()
    padding = window // 2
    padded = np.pad(array, pad_width=padding, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1).astype(np.float32, copy=False)


def _validate_sg_params(window_length, polyorder, deriv) -> tuple[int, int, int]:
    """校验 Savitzky-Golay 窗口、阶数和导数阶数。"""
    if window_length is None:
        raise ValueError("window_length 不能为空")
    window_length = int(window_length)
    polyorder = int(polyorder)
    deriv = int(deriv)
    if window_length <= 0:
        raise ValueError(f"window_length 必须大于 0，当前为 {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length 必须是奇数，当前为 {window_length}")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length 必须大于 polyorder，当前 window_length={window_length}, polyorder={polyorder}"
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(
            f"deriv 必须落在 [0, polyorder]，当前 deriv={deriv}, polyorder={polyorder}"
        )
    return window_length, polyorder, deriv


def sg_coeff(window_length, polyorder, deriv):
    """生成一组 Savitzky-Golay 平滑或导数卷积核系数。"""
    window_length, polyorder, deriv = _validate_sg_params(
        window_length, polyorder, deriv
    )
    half = (window_length - 1) // 2
    positions = np.arange(-half, half + 1, dtype=np.float32)
    design = np.vander(positions, N=polyorder + 1, increasing=True)
    coefficients = np.linalg.pinv(design)[deriv] * math.factorial(deriv)
    coefficients = coefficients.astype(np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError(
            f"sg_coeff 生成了非有限值：window_length={window_length}, polyorder={polyorder}, deriv={deriv}"
        )
    return coefficients
