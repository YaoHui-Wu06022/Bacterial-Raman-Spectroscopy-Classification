"""数组窗口与布尔区间工具。"""

import numpy as np


def odd_window_points(points, min_points=3):
    """将窗口点数规范为不小于 ``min_points`` 的奇数。"""
    points = max(int(min_points), int(round(float(points))))
    return points + 1 if points % 2 == 0 else points


def median_filter_1d(values, window):
    """使用边缘复制的一维中值滤波，保留输入长度。"""
    values = np.asarray(values)
    window = odd_window_points(window)
    if values.size < window:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, pad_width=padding, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1).astype(np.float32, copy=False)


def iter_true_segments(mask):
    """按顺序产生布尔掩码中连续 ``True`` 的左闭右开索引区间。"""
    start = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def contiguous_regions(mask):
    """返回布尔掩码中连续 ``True`` 的索引区间列表。"""
    return list(iter_true_segments(np.asarray(mask, dtype=bool)))
