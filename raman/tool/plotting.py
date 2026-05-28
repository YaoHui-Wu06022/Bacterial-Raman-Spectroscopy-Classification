"""通用绘图辅助函数"""

import numpy as np

from raman.tool.array import contiguous_regions
from raman.tool.spectrum import median_step_cm


def insert_nan_gaps(wn, *values):
    """在波数轴明显断开的地方插入 NaN，避免绘图跨坏段连线"""
    wn = np.asarray(wn, dtype=np.float32)
    value_arrays = [np.asarray(value, dtype=np.float32) for value in values]
    if wn.size < 2:
        return (wn, *value_arrays)

    diffs = np.diff(wn)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        return (wn, *value_arrays)

    normal_step = float(np.median(positive))
    gap_mask = diffs > normal_step * 3.0
    if not gap_mask.any():
        return (wn, *value_arrays)

    wn_out = []
    values_out = [[] for _ in value_arrays]
    for idx in range(wn.size):
        wn_out.append(float(wn[idx]))
        for out, arr in zip(values_out, value_arrays):
            out.append(float(arr[idx]))
        if idx < wn.size - 1 and gap_mask[idx]:
            wn_out.append(float((wn[idx] + wn[idx + 1]) / 2.0))
            for out in values_out:
                out.append(np.nan)

    return (
        np.asarray(wn_out, dtype=np.float32),
        *[np.asarray(out, dtype=np.float32) for out in values_out],
    )


def add_bad_band_spans(ax, bad_bands, alpha=0.14, label=None):
    """在图中标出坏段区间"""
    labeled = False
    for band_min, band_max in bad_bands:
        span_label = label if label and not labeled else None
        ax.axvspan(band_min, band_max, color="gray", alpha=alpha, label=span_label)
        labeled = True


def keep_mask_without_bad_bands(wn, bad_bands):
    """构造绘图时避开坏段和大间隔的掩码"""
    wn = np.asarray(wn, dtype=np.float32)
    keep = np.ones_like(wn, dtype=bool)
    for band_min, band_max in bad_bands:
        keep &= ~((wn >= band_min) & (wn <= band_max))
    if wn.size >= 2:
        step = median_step_cm(wn)
        gap_breaks = np.where(np.abs(np.diff(wn)) > step * 1.8)[0]
        keep[gap_breaks + 1] = False
    return keep


def plot_segments_without_bad_bands(ax, wn, values, bad_bands, show_bad_bands=True, **kwargs):
    """绘制断线光谱，避免坏段灰色区域内出现连线"""
    wn = np.asarray(wn, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    keep = keep_mask_without_bad_bands(wn, bad_bands)
    label = kwargs.pop("label", None)
    labeled = False
    for start, end in contiguous_regions(keep):
        if end - start >= 2:
            line_label = label if label and not labeled else None
            ax.plot(wn[start:end], values[start:end], label=line_label, **kwargs)
            labeled = True
    if show_bad_bands:
        add_bad_band_spans(ax, bad_bands)


def fill_between_segments_without_bad_bands(ax, wn, lower, upper, bad_bands, **kwargs):
    """分段绘制区间填充，避免跨坏段连起来"""
    wn = np.asarray(wn, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    keep = keep_mask_without_bad_bands(wn, bad_bands)
    label = kwargs.pop("label", None)
    labeled = False
    for start, end in contiguous_regions(keep):
        if end - start >= 2:
            band_label = label if label and not labeled else None
            ax.fill_between(wn[start:end], lower[start:end], upper[start:end], label=band_label, **kwargs)
            labeled = True
    add_bad_band_spans(ax, bad_bands)


def shorten_class_name(name):
    """坐标轴只显示最后一级类别名，避免层级路径太长"""
    text = str(name).replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    return parts[-1] if parts else text


def shorten_class_names(class_names):
    """批量压缩类别显示名"""
    return [shorten_class_name(name) for name in class_names]


def auto_confusion_matrix_figsize(class_names):
    """按类别数等比例放大图片，让每个格子的物理尺寸尽量一致"""
    num_classes = max(len(class_names), 1)
    max_label_len = max((len(str(name)) for name in class_names), default=0)

    cell_size = 0.62
    label_pad = min(max_label_len, 24) * 0.06
    width = 2.3 + num_classes * cell_size + label_pad
    height = 2.3 + num_classes * cell_size
    return (
        min(max(width, 6.0), 38.0),
        min(max(height, 5.6), 38.0),
    )


def auto_confusion_matrix_left_margin(class_names):
    """按 y 轴类别名长度增加左侧留白，避免较长属名贴边或被裁切"""
    max_label_len = max((len(str(name)) for name in class_names), default=0)
    margin = 0.115 + min(max_label_len, 28) * 0.006
    return min(max(margin, 0.18), 0.34)


def auto_confusion_matrix_font_sizes(num_classes):
    """按类别数缩放标注和坐标字号"""
    if num_classes <= 12:
        return 11, 12
    if num_classes <= 24:
        return 9, 11
    if num_classes <= 36:
        return 8, 10
    return 7, 9
