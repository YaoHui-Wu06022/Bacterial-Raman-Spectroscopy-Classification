"""波数轴和坏段 mask 工具"""

import numpy as np


def build_wn_ref(cut_min, cut_max, target_points):
    """生成裁剪后统一插值使用的波数坐标"""
    return np.linspace(cut_min, cut_max, target_points)


def normalize_bad_bands(bad_bands):
    """规范化坏波段配置，过滤非法项并统一转成浮点区间"""
    if not bad_bands:
        return ()

    normalized = []
    for band in bad_bands:
        if band is None:
            continue
        if not isinstance(band, (tuple, list)) or len(band) != 2:
            continue

        band_min, band_max = band
        if band_min is None or band_max is None:
            continue

        band_min = float(band_min)
        band_max = float(band_max)
        if band_max < band_min:
            band_min, band_max = band_max, band_min
        normalized.append((band_min, band_max))

    return tuple(normalized)


def build_valid_mask(wn, bad_bands):
    """根据坏波段区间构造有效波段掩码；未配置坏波段时返回 None"""
    bad_bands = normalize_bad_bands(bad_bands)
    if not bad_bands:
        return None
    valid_mask = np.ones_like(wn, dtype=bool)
    for band_min, band_max in bad_bands:
        valid_mask &= ~((wn >= band_min) & (wn <= band_max))
    return valid_mask


def get_config_bad_bands(config):
    """从配置对象中读取坏波段设置"""
    if hasattr(config, "BAD_BANDS"):
        return normalize_bad_bands(config.BAD_BANDS)
    if hasattr(config, "bad_bands"):
        return normalize_bad_bands(config.bad_bands)
    return ()


def median_step_cm(wn):
    """估计波数轴的中位步长"""
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size < 2:
        return 1.0
    diffs = np.abs(np.diff(wn))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-8)]
    return float(np.median(diffs)) if diffs.size else 1.0


def region_width_cm(wn, start, end):
    """计算索引区间对应的波数宽度"""
    if end <= start:
        return 0.0
    step = median_step_cm(wn)
    return float(abs(wn[end - 1] - wn[start]) + step)


def build_wavenumber_axis(length, config):
    """根据预处理配置构造和模型输入长度对齐的波数轴"""
    bad_bands = get_config_bad_bands(config)
    if hasattr(config, "cut_min") and hasattr(config, "cut_max"):
        if hasattr(config, "target_points"):
            try:
                target_points = int(config.target_points)
            except Exception:
                target_points = None
            if target_points:
                wn_full = np.linspace(config.cut_min, config.cut_max, target_points)
                keep_mask = build_valid_mask(wn_full, bad_bands)
                if keep_mask is not None:
                    wn_full = wn_full[keep_mask]
                if wn_full.shape[0] == length:
                    return wn_full
        if hasattr(config, "delta"):
            return config.cut_min + config.delta * np.arange(length)
        return np.linspace(config.cut_min, config.cut_max, length)
    return np.arange(length)


def expected_wavenumbers(config):
    """按输入配置生成严格匹配模型输入长度的波数轴"""
    target_points = int(config.target_points)
    wn = np.linspace(float(config.cut_min), float(config.cut_max), target_points)
    keep_mask = build_valid_mask(wn, get_config_bad_bands(config))
    if keep_mask is not None:
        wn = wn[keep_mask]
    return wn


def output_wavenumbers(config):
    """返回应用坏段遮罩后的输出波数轴"""
    wn = config.build_wn_ref()
    keep = build_valid_mask(wn, getattr(config, "bad_bands", ()))
    return wn[keep] if keep is not None else wn


def contiguous_index_ranges(wn, gap_factor=1.8):
    """按波数轴大间隔拆成连续索引段"""
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size == 0:
        return []
    step = median_step_cm(wn)
    cuts = np.where(np.abs(np.diff(wn)) > step * float(gap_factor))[0] + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [wn.size]])
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def estimate_gap_indices(wavenumbers, gap_factor=1.5):
    """通过相邻步长突变估计绘图断线位置"""
    wavenumbers = np.asarray(wavenumbers, dtype=np.float32)
    if wavenumbers.size < 2:
        return []
    diffs = np.diff(wavenumbers)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return []
    step = float(np.median(positive))
    if step <= 0:
        return []
    return np.where(np.diff(wavenumbers) > step * float(gap_factor))[0].tolist()
