"""坏波段配置与有效点掩码。"""

from __future__ import annotations

import numpy as np


def normalize_bad_bands(bad_bands) -> tuple[tuple[float, float], ...]:
    """过滤非法配置，并将每段坏波数区间规范为升序浮点元组。"""
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


def build_valid_mask(wavenumbers, bad_bands):
    """构造排除坏波段的布尔掩码；未配置时返回 ``None``。"""
    normalized = normalize_bad_bands(bad_bands)
    if not normalized:
        return None
    mask = np.ones_like(wavenumbers, dtype=bool)
    for lower, upper in normalized:
        mask &= ~((wavenumbers >= lower) & (wavenumbers <= upper))
    return mask


def get_config_bad_bands(config) -> tuple[tuple[float, float], ...]:
    """兼容读取配置对象中的 ``BAD_BANDS`` 或 ``bad_bands`` 字段。"""
    if hasattr(config, "BAD_BANDS"):
        return normalize_bad_bands(config.BAD_BANDS)
    if hasattr(config, "bad_bands"):
        return normalize_bad_bands(config.bad_bands)
    return ()
