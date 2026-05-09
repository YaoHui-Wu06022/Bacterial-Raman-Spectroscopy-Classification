import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy import sparse
from scipy.sparse.linalg import spsolve

from raman.data.spectrum import (
    build_valid_mask,
    minmax_normalize,
    normalize_bad_bands,
    snv,
)


def asls_baseline(spectrum, lam=1e5, p=0.01, niter=10, valid_mask=None):
    """使用 AsLS 估计基线，可选择跳过坏波段对应位置"""
    length = len(spectrum)
    # 构造二阶差分矩阵
    diff = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    weights = np.ones(length)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0

    for _ in range(niter):
        matrix_w = sparse.diags(weights, 0)
        matrix_b = (matrix_w + lam * (diff.T @ diff)).tocsc()
        baseline = spsolve(matrix_b, weights * spectrum)
        weights = np.where(spectrum > baseline, p, 1 - p)
        if valid_mask is not None:
            weights[~valid_mask] = 0.0

    return baseline


def normalize_for_plot(spectra, method):
    """按指定方法对绘图用光谱做归一化"""
    method = method.lower()
    if method == "minmax":
        return minmax_normalize(spectra)
    if method == "snv":
        return snv(spectra)
    raise ValueError(f"Unknown norm method: {method}")


def _median_filter_1d(x, window):
    """用边缘复制的局部中值估计正常谱形"""
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


def remove_cosmic_rays(
    sp,
    window=7,
    threshold=8.0,
    max_iter=2,
    valid_mask=None,
):
    """
    保守去除宇宙射线尖峰

    只检测相对局部中值异常抬高的窄峰，坏段位置不参与判断
    """
    cleaned = np.asarray(sp, dtype=np.float32).copy()
    if cleaned.size < 3 or max_iter <= 0:
        return cleaned, 0

    if valid_mask is None:
        valid_mask = np.ones(cleaned.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != cleaned.shape:
            valid_mask = np.ones(cleaned.shape, dtype=bool)

    replaced_mask = np.zeros(cleaned.shape, dtype=bool)
    for _ in range(int(max_iter)):
        local_median = _median_filter_1d(cleaned, window)
        residual = cleaned - local_median
        residual_valid = residual[valid_mask]
        residual_valid = residual_valid[np.isfinite(residual_valid)]
        if residual_valid.size == 0:
            break

        center = np.median(residual_valid)
        mad = np.median(np.abs(residual_valid - center))
        scale = 1.4826 * mad
        if scale <= 1e-8:
            scale = float(np.std(residual_valid))
        if scale <= 1e-8:
            break

        spike_mask = valid_mask & (residual > float(threshold) * scale)
        if not spike_mask.any():
            break

        replaced_mask |= spike_mask
        cleaned[spike_mask] = local_median[spike_mask]

    return cleaned, int(replaced_mask.sum())


def remove_group_cosmic_rays(
    spectra,
    threshold=12.0,
    min_samples=3,
):
    """
    用同一类别内的统计量兜底去除残留尖峰

    单谱局部中值适合抓单点尖峰；组内统计更适合抓只出现在少数样本中的窄段异常
    """
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[0] < int(min_samples):
        return spectra, 0

    center = np.median(spectra, axis=0)
    residual = spectra - center
    mad = np.median(np.abs(residual), axis=0)
    scale = 1.4826 * mad

    valid_scale = scale[np.isfinite(scale) & (scale > 1e-6)]
    if valid_scale.size == 0:
        fallback_scale = float(np.std(residual))
    else:
        fallback_scale = float(np.median(valid_scale))
    fallback_scale = max(fallback_scale, 1e-6)
    scale = np.where(scale > 1e-6, scale, fallback_scale)

    spike_mask = residual > float(threshold) * scale
    if not spike_mask.any():
        return spectra, 0

    cleaned = spectra.copy()
    cleaned[spike_mask] = np.broadcast_to(center, spectra.shape)[spike_mask]
    return cleaned, int(spike_mask.sum())


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
    cosmic_ray_remove=False,
    cosmic_ray_window=7,
    cosmic_ray_threshold=8.0,
    cosmic_ray_max_iter=2,
):
    """对单条光谱执行宇宙射线去除、基线校正、裁剪和插值"""
    bad_bands = normalize_bad_bands(bad_bands)
    valid_mask = build_valid_mask(wn, bad_bands)
    sp_clean = np.asarray(sp, dtype=np.float32)
    cosmic_replaced = 0

    if cosmic_ray_remove:
        sp_clean, cosmic_replaced = remove_cosmic_rays(
            sp_clean,
            window=cosmic_ray_window,
            threshold=cosmic_ray_threshold,
            max_iter=cosmic_ray_max_iter,
            valid_mask=valid_mask,
        )

    baseline = asls_baseline(
        sp_clean,
        lam=asls_lam,
        p=asls_p,
        niter=asls_max_iter,
        valid_mask=valid_mask,
    )
    sp_bc = sp_clean - baseline

    mask_cut = (wn >= cut_min) & (wn <= cut_max)
    wn_cut = wn[mask_cut]
    sp_cut = sp_bc[mask_cut]

    if wn_cut.size < 10:
        return None, None, cosmic_replaced

    if bad_bands:
        src_keep_mask = build_valid_mask(wn_cut, bad_bands)
        wn_cut = wn_cut[src_keep_mask]
        sp_cut = sp_cut[src_keep_mask]

        target_keep_mask = build_valid_mask(wn_ref, bad_bands)
        wn_ref = wn_ref[target_keep_mask]

    if wn_cut.size < 10 or wn_ref.size == 0:
        return None, None, cosmic_replaced

    sp_interp = np.interp(wn_ref, wn_cut, sp_cut)

    return wn_ref, sp_interp, cosmic_replaced


def save_mean_plot(wn, spectra, out_path, norm_method, bad_bands, title):
    """保存一组光谱的均值谱图，并在图上标出坏波段区域"""
    bad_bands = normalize_bad_bands(bad_bands)
    spectra_norm = normalize_for_plot(spectra, norm_method)
    mean_spec = np.mean(spectra_norm, axis=0)
    q10_spec = np.quantile(spectra_norm, 0.10, axis=0)
    q90_spec = np.quantile(spectra_norm, 0.90, axis=0)

    plt.figure(figsize=(10, 5))
    std_proxy = Patch(facecolor="C0", alpha=0.3, label="q10-q90 range")

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
    plt.fill_between(wn, q10_spec, q90_spec, alpha=0.3)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel(f"{norm_method} intensity")
    plt.xlim([wn.min(), wn.max()])
    plt.legend(handles=[std_proxy] + plt.gca().get_legend_handles_labels()[0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
