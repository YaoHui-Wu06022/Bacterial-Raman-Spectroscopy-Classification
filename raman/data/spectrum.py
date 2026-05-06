import numpy as np

EPS = 1e-8


def read_arc_data(path):
    """读取两列文本光谱文件，返回波数和强度数组"""
    wn, sp = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                wn.append(float(parts[0]))
                sp.append(float(parts[1]))
            except Exception:
                continue
    return np.array(wn), np.array(sp)


def load_arc_intensity(path):
    """读取单个 .arc_data 文件的强度列"""
    data = np.loadtxt(path, dtype=np.float32)
    data = np.atleast_2d(data)
    return data[:, 1].astype(np.float32, copy=False)


def write_arc_data(path, wn, sp, fmt="%.8f"):
    """把一条光谱写回两列文本格式"""
    arr = np.column_stack([wn, sp])
    np.savetxt(path, arr, fmt=[fmt, fmt])


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

        normalized.append((float(band_min), float(band_max)))

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


def snv(data, eps=EPS):
    """对单条或多条光谱做 SNV 标准化"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        mean = data.mean()
        std = max(data.std(), eps)
        return (data - mean) / std
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + eps)


def minmax_normalize(data, eps=EPS):
    """对单条或多条光谱做 Min-Max 归一化"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        min_value = np.min(data)
        max_value = np.max(data)
        denom = max(max_value - min_value, eps)
        return (data - min_value) / denom
    min_value = np.min(data, axis=1, keepdims=True)
    max_value = np.max(data, axis=1, keepdims=True)
    denom = np.maximum(max_value - min_value, eps)
    return (data - min_value) / denom
