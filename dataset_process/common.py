import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy import sparse
from scipy.sparse.linalg import spsolve

EPS = 1e-8


def read_arc_data(path):
    """读取两列文本光谱文件，返回波数和强度数组。"""
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


def build_wn_ref(cut_min, cut_max, target_points):
    """生成裁剪后统一插值使用的波数坐标。"""
    return np.linspace(cut_min, cut_max, target_points)


def normalize_bad_bands(bad_bands):
    """规范化坏波段配置，过滤非法项并统一转成浮点区间。"""
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
    """根据坏波段区间构造有效波段掩码；未配置坏波段时返回 None。"""
    bad_bands = normalize_bad_bands(bad_bands)
    if not bad_bands:
        return None
    valid_mask = np.ones_like(wn, dtype=bool)
    for band_min, band_max in bad_bands:
        valid_mask &= ~((wn >= band_min) & (wn <= band_max))
    return valid_mask


def asls_baseline(spectrum, lam=1e5, p=0.01, niter=10, valid_mask=None):
    """使用 AsLS 估计基线，可选择跳过坏波段对应位置。"""
    length = len(spectrum)
    # 构造二阶差分矩阵
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    weights = np.ones(length)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0

    for _ in range(niter):
        matrix_w = sparse.diags(weights, 0)
        matrix_z = (matrix_w + lam * (D.T @ D)).tocsc() # W + λD^T D
        baseline = spsolve(matrix_z, weights * spectrum)    # 解 z
        weights = np.where(spectrum > baseline, p, 1 - p)
        if valid_mask is not None:
            weights[~valid_mask] = 0.0

    return baseline


def snv(data, eps=EPS):
    """对单条或多条光谱做 SNV 标准化。"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        mean = data.mean()
        std = max(data.std(), eps)
        return (data - mean) / std
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + eps)


def minmax_normalize(data, eps=EPS):
    """对单条或多条光谱做 Min-Max 归一化。"""
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


def normalize_for_plot(spectra, method):
    """按指定方法对绘图用光谱做归一化。"""
    method = method.lower()
    if method == "minmax":
        return minmax_normalize(spectra)
    if method == "snv":
        return snv(spectra)
    raise ValueError(f"Unknown norm method: {method}")


def preprocess_single_spectrum(
    wn,
    sp,
    cut_min,
    cut_max,
    wn_ref,
    bad_bands,
    asls_lam,
    asls_p,
    asls_max_iter,
):
    """对单条光谱执行基线校正、裁剪，并让坏波段同时退出插值与最终输出。"""
    bad_bands = normalize_bad_bands(bad_bands)
    valid_mask = build_valid_mask(wn, bad_bands)

    baseline = asls_baseline(
        sp,
        lam=asls_lam,
        p=asls_p,
        niter=asls_max_iter,
        valid_mask=valid_mask,
    )
    sp_bc = sp - baseline

    mask_cut = (wn >= cut_min) & (wn <= cut_max)
    wn_cut = wn[mask_cut]
    sp_cut = sp_bc[mask_cut]

    if wn_cut.size < 10:
        return None, None

    if bad_bands:
        src_keep_mask = build_valid_mask(wn_cut, bad_bands)
        wn_cut = wn_cut[src_keep_mask]
        sp_cut = sp_cut[src_keep_mask]

        target_keep_mask = build_valid_mask(wn_ref, bad_bands)
        wn_ref = wn_ref[target_keep_mask]

    if wn_cut.size < 10 or wn_ref.size == 0:
        return None, None

    sp_interp = np.interp(wn_ref, wn_cut, sp_cut)

    return wn_ref, sp_interp


def save_mean_plot(wn, spectra, out_path, norm_method, bad_bands, title):
    """保存一组光谱的均值谱图，并在图上标出坏波段区域。"""
    bad_bands = normalize_bad_bands(bad_bands)
    spectra_norm = normalize_for_plot(spectra, norm_method)
    mean_spec = np.mean(spectra_norm, axis=0)
    std_spec = np.std(spectra_norm, axis=0)

    plt.figure(figsize=(10, 5))
    std_proxy = Patch(facecolor="C0", alpha=0.3, label="+/-1 std range")

    if bad_bands:
        for band_min, band_max in bad_bands:
            plt.axvspan(
                band_min,
                band_max,
                color="gray",
                alpha=0.2,
                label="CCD-affected region"
                if (band_min, band_max) == bad_bands[0]
                else None,
            )

    plt.plot(wn, mean_spec, label=f"Mean spectrum {norm_method}")
    plt.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, alpha=0.3)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel(f"{norm_method} intensity")
    plt.xlim([wn.min(), wn.max()])
    plt.legend(handles=[std_proxy] + plt.gca().get_legend_handles_labels()[0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
