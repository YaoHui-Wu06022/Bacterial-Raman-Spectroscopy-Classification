"""波数轴、坏波段与绘图断点工具。"""

import numpy as np


def build_wn_ref(cut_min, cut_max, target_points):
    """生成裁剪后插值使用的统一波数轴。"""
    return np.linspace(cut_min, cut_max, target_points)


def normalize_bad_bands(bad_bands):
    """过滤非法坏波段配置，并统一为升序浮点区间元组。"""
    if not bad_bands:
        return ()
    normalized = []
    for band in bad_bands:
        if not isinstance(band, (tuple, list)) or len(band) != 2:
            continue
        lower, upper = band
        if lower is None or upper is None:
            continue
        lower, upper = float(lower), float(upper)
        normalized.append((min(lower, upper), max(lower, upper)))
    return tuple(normalized)


def build_valid_mask(wn, bad_bands):
    """构造有效波段掩码；未配置坏波段时返回 ``None``。"""
    bad_bands = normalize_bad_bands(bad_bands)
    if not bad_bands:
        return None
    mask = np.ones_like(wn, dtype=bool)
    for lower, upper in bad_bands:
        mask &= ~((wn >= lower) & (wn <= upper))
    return mask


def get_config_bad_bands(config):
    """兼容读取配置对象中的 ``BAD_BANDS`` 或 ``bad_bands``。"""
    if hasattr(config, "BAD_BANDS"):
        return normalize_bad_bands(config.BAD_BANDS)
    if hasattr(config, "bad_bands"):
        return normalize_bad_bands(config.bad_bands)
    return ()


def median_step_cm(wn):
    """估计波数轴的中位有效步长。"""
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size < 2:
        return 1.0
    diffs = np.abs(np.diff(wn))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-8)]
    return float(np.median(diffs)) if diffs.size else 1.0


def build_wavenumber_axis(length, config):
    """按预处理配置构造与模型输入长度对齐的波数轴。"""
    bad_bands = get_config_bad_bands(config)
    if hasattr(config, "cut_min") and hasattr(config, "cut_max"):
        if hasattr(config, "target_points"):
            try:
                target_points = int(config.target_points)
            except (TypeError, ValueError):
                target_points = None
            if target_points:
                wn_full = np.linspace(config.cut_min, config.cut_max, target_points)
                mask = build_valid_mask(wn_full, bad_bands)
                if mask is not None:
                    wn_full = wn_full[mask]
                if wn_full.shape[0] == length:
                    return wn_full
        if hasattr(config, "delta"):
            return config.cut_min + config.delta * np.arange(length)
        return np.linspace(config.cut_min, config.cut_max, length)
    return np.arange(length)


def expected_wavenumbers(config):
    """按配置生成严格匹配模型输入长度的有效波数轴。"""
    wn = np.linspace(float(config.cut_min), float(config.cut_max), int(config.target_points))
    mask = build_valid_mask(wn, get_config_bad_bands(config))
    return wn[mask] if mask is not None else wn


def estimate_gap_indices(wavenumbers, gap_factor=1.5):
    """根据相邻步长突变返回绘图断线前的索引。"""
    wavenumbers = np.asarray(wavenumbers, dtype=np.float32)
    if wavenumbers.size < 2:
        return []
    steps = np.diff(wavenumbers)
    positive = steps[steps > 0]
    if positive.size == 0:
        return []
    step = float(np.median(positive))
    return np.where(steps > step * float(gap_factor))[0].tolist() if step > 0 else []
