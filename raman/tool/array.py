"""数组窗口、区间和鲁棒尺度工具"""

import numpy as np


def median_filter_1d(x, window):
    """用边缘复制的一维 median 估计局部正常谱形"""
    window = int(window)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    if x.size < window:
        return x.copy()

    pad = window // 2
    padded = np.pad(x, pad_width=pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1).astype(np.float32, copy=False)


def odd_window_points(points, min_points=3):
    """把窗口点数规整为不小于 min_points 的奇数"""
    points = max(int(min_points), int(round(float(points))))
    if points % 2 == 0:
        points += 1
    return points


def nonnegative_points(points):
    """把点数参数规整为非负整数"""
    return max(0, int(round(float(points))))


def iter_true_segments(mask):
    """按顺序产出布尔 mask 中连续 True 区间，区间右端开口"""
    start = None
    for idx, enabled in enumerate(mask):
        if enabled and start is None:
            start = idx
        elif not enabled and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask)


def contiguous_regions(mask):
    """返回布尔 mask 中连续 True 区间列表"""
    return list(iter_true_segments(np.asarray(mask, dtype=bool)))


def robust_finite_scale(values):
    """用有限值的 MAD/标准差估计鲁棒尺度，异常或空输入返回 0"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    center = float(np.median(values))
    centered_abs = np.abs(values - center)
    scale = 1.4826 * float(np.median(centered_abs))
    if scale <= 1e-8:
        nonzero_abs = centered_abs[centered_abs > 1e-8]
        if nonzero_abs.size > 0:
            scale = 1.4826 * float(np.median(nonzero_abs))
    if scale <= 1e-8:
        scale = float(np.std(values))
    return max(scale, 0.0)


def robust_mad_scale(values, floor=1e-8):
    """返回一组数值的 MAD 鲁棒尺度"""
    scale = robust_finite_scale(values)
    return max(float(scale), float(floor))


def moving_average(values, window):
    """对一维数组做奇数窗口滑动平均"""
    values = np.asarray(values, dtype=np.float32)
    window = odd_window_points(window)
    if values.size < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def robust_wave_stats(spectra, min_scale=0.05, floor_fraction=0.25):
    """逐波数计算中位谱和鲁棒尺度"""
    spectra = np.asarray(spectra, dtype=np.float32)
    center = np.median(spectra, axis=0)
    mad = np.median(np.abs(spectra - center), axis=0)
    scale = 1.4826 * mad
    if np.any(scale > 1e-8):
        floor = max(
            float(np.median(scale[scale > 1e-8])) * float(floor_fraction),
            float(min_scale),
        )
    else:
        floor = float(min_scale)
    return center, np.maximum(scale, floor)
