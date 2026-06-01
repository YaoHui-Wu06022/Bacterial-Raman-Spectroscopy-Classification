"""离线预处理链路

集中放置从原始光谱到清洗后光谱的离线处理逻辑：
基线估计、宇宙射线处理、单谱预处理和均值图输出
"""

# ===== 基线估计 =====
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy import sparse
from scipy.sparse.linalg import spsolve

from raman.data.input import normalize_spectrum
from raman.tool.array import (
    median_filter_1d,
    odd_window_points,
)
from raman.tool.plotting import insert_nan_gaps
from raman.tool.spectrum import build_valid_mask, normalize_bad_bands


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
        if valid_mask is not None:
            next_weights = np.zeros(length)
            next_weights[valid_mask] = np.where(spectrum[valid_mask] > baseline[valid_mask], p, 1 - p)
            weights = next_weights
        else:
            weights = np.where(spectrum > baseline, p, 1 - p)

    return baseline


def arpls_baseline(spectrum, lam=1e5, niter=15, valid_mask=None):
    """使用 arPLS 估计基线，对正向 Raman 峰更少施加权重"""
    y = np.asarray(spectrum, dtype=np.float64)
    length = len(y)
    diff = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    penalty = lam * (diff.T @ diff)
    weights = np.ones(length, dtype=np.float64)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0

    for _ in range(int(niter)):
        matrix_w = sparse.diags(weights, 0)
        baseline = spsolve((matrix_w + penalty).tocsc(), weights * y)
        residual = y - baseline
        negative = residual[residual < 0]
        if negative.size < 2:
            break

        mean_neg = float(np.mean(negative))
        std_neg = float(np.std(negative))
        if std_neg <= 1e-12:
            break

        logits = 2.0 * (residual - (2.0 * std_neg - mean_neg)) / std_neg
        logits = np.clip(logits, -60.0, 60.0)
        next_weights = 1.0 / (1.0 + np.exp(logits))
        if valid_mask is not None:
            next_weights[~valid_mask] = 0.0

        if np.linalg.norm(next_weights - weights) / max(np.linalg.norm(weights), 1e-12) < 1e-3:
            weights = next_weights
            break
        weights = next_weights

    return baseline.astype(np.float32, copy=False)


def airpls_baseline(spectrum, lam=1e5, niter=15, valid_mask=None):
    """使用 airPLS 估计基线，迭代提高负残差区域权重"""
    y = np.asarray(spectrum, dtype=np.float64)
    length = len(y)
    diff = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    penalty = lam * (diff.T @ diff)
    weights = np.ones(length, dtype=np.float64)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0
        stop_scale = float(np.sum(np.abs(y[valid_mask])))
    else:
        stop_scale = float(np.sum(np.abs(y)))
    stop_scale = max(stop_scale, 1e-12)

    for iteration in range(1, int(niter) + 1):
        matrix_w = sparse.diags(weights, 0)
        baseline = spsolve((matrix_w + penalty).tocsc(), weights * y)
        residual = y - baseline
        negative_mask = residual < 0
        if valid_mask is not None:
            negative_mask &= valid_mask

        negative_sum = float(np.sum(np.abs(residual[negative_mask])))
        if negative_sum <= 1e-3 * stop_scale:
            break

        next_weights = np.zeros(length, dtype=np.float64)
        next_weights[negative_mask] = np.exp(
            np.clip(
                iteration * np.abs(residual[negative_mask]) / max(negative_sum, 1e-12),
                -60.0,
                60.0,
            )
        )
        if next_weights[negative_mask].size > 0:
            edge_weight = float(next_weights[negative_mask].max())
            next_weights[0] = edge_weight
            next_weights[-1] = edge_weight
        if valid_mask is not None:
            next_weights[~valid_mask] = 0.0
        weights = next_weights

    return baseline.astype(np.float32, copy=False)


def estimate_baseline(
    spectrum,
    method="asls",
    lam=1e5,
    p=0.01,
    niter=15,
    valid_mask=None,
):
    """按配置选择基线估计算法"""
    method = str(method).lower()
    if method == "asls":
        return asls_baseline(
            spectrum,
            lam=lam,
            p=p,
            niter=niter,
            valid_mask=valid_mask,
        )
    if method == "arpls":
        return arpls_baseline(
            spectrum,
            lam=lam,
            niter=niter,
            valid_mask=valid_mask,
        )
    if method == "airpls":
        return airpls_baseline(
            spectrum,
            lam=lam,
            niter=niter,
            valid_mask=valid_mask,
        )
    raise ValueError(f"Unknown baseline method: {method}")


# ===== 宇宙射线处理 =====

@dataclass(frozen=True)
class CosmicRayStats:
    """记录单谱宇宙射线替换点数"""

    cosmic_ray: int = 0

    def __int__(self):
        return int(self.cosmic_ray)


def _residual_z_score(residual, valid_mask):
    residual_valid = residual[valid_mask]
    residual_valid = residual_valid[np.isfinite(residual_valid)]
    if residual_valid.size == 0:
        return None

    center = np.median(residual_valid)
    centered_abs = np.abs(residual_valid - center)
    mad = np.median(centered_abs)
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(residual_valid))
    if scale <= 1e-8:
        nonzero_abs = centered_abs[centered_abs > 1e-8]
        if nonzero_abs.size > 0:
            scale = 1.4826 * float(np.median(nonzero_abs))
    if scale <= 1e-8:
        return None

    return (residual - center) / scale


def remove_cosmic_rays(
    sp,
    window_points=7,
    threshold=7.0,
    max_iter=2,
    valid_mask=None,
):
    """用局部 median/MAD 去除正向宇宙射线尖峰"""
    cleaned = np.asarray(sp, dtype=np.float32).copy()
    if cleaned.size < 3 or max_iter <= 0:
        return cleaned, CosmicRayStats()

    if valid_mask is None:
        valid_mask = np.ones(cleaned.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != cleaned.shape:
            valid_mask = np.ones(cleaned.shape, dtype=bool)

    replaced_mask = np.zeros(cleaned.shape, dtype=bool)
    cosmic_ray_window = odd_window_points(window_points)
    for _ in range(int(max_iter)):
        local_median = median_filter_1d(cleaned, cosmic_ray_window)
        residual = cleaned - local_median
        z_score = _residual_z_score(residual, valid_mask)
        if z_score is None:
            break

        spike_mask = valid_mask & (z_score > float(threshold))
        if not spike_mask.any():
            break

        replaced_mask |= spike_mask
        cleaned[spike_mask] = local_median[spike_mask]

    return cleaned, CosmicRayStats(cosmic_ray=int(replaced_mask.sum()))


# ===== 单谱离线预处理 =====
def preprocess_single_spectrum(
    wn,
    sp,
    cut_min,
    cut_max,
    wn_ref,
    bad_bands,
    baseline_lam,
    baseline_asls_p,
    baseline_max_iter,
    baseline_fit_min=None,
    baseline_fit_max=None,
    baseline_method="asls",
    cosmic_ray_remove=False,
    cosmic_ray_window_points=7,
    cosmic_ray_threshold=7.0,
    cosmic_ray_max_iter=2,
):
    """对单条光谱执行宇宙射线去除、基线校正、裁剪和插值"""
    bad_bands = normalize_bad_bands(bad_bands)
    sp_clean = np.asarray(sp, dtype=np.float32)
    cosmic_replaced = CosmicRayStats()

    if cosmic_ray_remove:
        # 宇宙射线先按整条原始谱处理，避免坏段边界附近的尖峰因 mask 断开而漏修。
        # 坏段 mask 仍用于后续基线估计、裁切删除和绘图遮挡。
        sp_clean, cosmic_replaced = remove_cosmic_rays(
            sp_clean,
            window_points=cosmic_ray_window_points,
            threshold=cosmic_ray_threshold,
            max_iter=cosmic_ray_max_iter,
            valid_mask=None,
        )

    fit_min = cut_min if baseline_fit_min is None else float(baseline_fit_min)
    fit_max = cut_max if baseline_fit_max is None else float(baseline_fit_max)
    baseline_fit_mask = (wn >= fit_min) & (wn <= fit_max)
    wn_fit = wn[baseline_fit_mask]
    sp_fit = sp_clean[baseline_fit_mask]
    valid_fit_mask = build_valid_mask(wn_fit, bad_bands)

    if wn_fit.size < 10:
        return None, None, cosmic_replaced

    baseline = estimate_baseline(
        sp_fit,
        method=baseline_method,
        lam=baseline_lam,
        p=baseline_asls_p,
        niter=baseline_max_iter,
        valid_mask=valid_fit_mask,
    )
    sp_bc_fit = sp_fit - baseline

    mask_cut = (wn_fit >= cut_min) & (wn_fit <= cut_max)
    wn_cut = wn_fit[mask_cut]
    sp_cut = sp_bc_fit[mask_cut]

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


# ===== 预处理结果绘图 =====
def prepare_mean_plot_data(wn, spectra, norm_method, bad_bands):
    """计算均值谱绘图需要的标准化统计量"""
    bad_bands = normalize_bad_bands(bad_bands)
    spectra_norm = normalize_spectrum(spectra, norm_method)
    mean_spec = np.mean(spectra_norm, axis=0)
    q10_spec = np.quantile(spectra_norm, 0.10, axis=0)
    q90_spec = np.quantile(spectra_norm, 0.90, axis=0)
    keep_mask = build_valid_mask(wn, bad_bands)
    if keep_mask is not None and keep_mask.shape == mean_spec.shape:
        mean_spec = np.where(keep_mask, mean_spec, np.nan)
        q10_spec = np.where(keep_mask, q10_spec, np.nan)
        q90_spec = np.where(keep_mask, q90_spec, np.nan)
    wn_plot, mean_plot, q10_plot, q90_plot = insert_nan_gaps(
        wn,
        mean_spec,
        q10_spec,
        q90_spec,
    )
    return bad_bands, wn_plot, mean_plot, q10_plot, q90_plot


def _add_bad_band_spans(ax, bad_bands, with_label=False):
    if bad_bands:
        for band_min, band_max in bad_bands:
            ax.axvspan(
                band_min,
                band_max,
                color="gray",
                alpha=0.2,
                label="CCD-affected region"
                if with_label and (band_min, band_max) == bad_bands[0]
                else None,
            )


def save_mean_plot(wn, spectra, out_path, norm_method, bad_bands, title):
    """保存一组光谱的均值谱图，并在图上标出坏波段区域"""
    bad_bands, wn_plot, mean_plot, q10_plot, q90_plot = prepare_mean_plot_data(
        wn,
        spectra,
        norm_method,
        bad_bands,
    )

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    _add_bad_band_spans(ax, bad_bands, with_label=True)
    plt.plot(wn_plot, mean_plot, label=f"Mean spectrum {norm_method}")
    plt.fill_between(wn_plot, q10_plot, q90_plot, alpha=0.3)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized intensity")
    plt.xlim([wn.min(), wn.max()])
    handles = [Patch(facecolor="C0", alpha=0.3, label="q10-q90 range")]
    handles += plt.gca().get_legend_handles_labels()[0]
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_mean_summary_plot(groups, out_path, norm_method, bad_bands):
    """按层级将多个类别均值谱错位排列为一张长图"""
    if not groups:
        return

    row_count = len(groups)
    plot_rows = []
    # Colorcet glasbey_dark 前 32 色，适合白底上的多类别细线
    colors = (
        "#d60000",
        "#8c3bff",
        "#018700",
        "#00acc6",
        "#e6a500",
        "#ff7ed1",
        "#6b004f",
        "#573b00",
        "#005659",
        "#15e18c",
        "#0000dd",
        "#a17569",
        "#bcb6ff",
        "#bf03b8",
        "#645472",
        "#790000",
        "#0774d8",
        "#729a7c",
        "#ff7752",
        "#004b00",
        "#8e7b01",
        "#f2007b",
        "#8eba00",
        "#a57bb8",
        "#5901a3",
        "#e2afaf",
        "#a03a52",
        "#a1c8c8",
        "#9e4b00",
        "#546744",
        "#bac389",
        "#5e7b87",
    )
    for group in groups:
        normalized_bands, wn_plot, mean_plot, _, _ = prepare_mean_plot_data(
            group["wn"],
            group["spectra"],
            norm_method,
            bad_bands,
        )
        plot_rows.append((group["label"], wn_plot, mean_plot, normalized_bands))

    finite_values = np.concatenate([curve[np.isfinite(curve)] for _, _, curve, _ in plot_rows])
    span = float(np.percentile(finite_values, 95) - np.percentile(finite_values, 5))
    offset_step = max(span * 1.05, 1e-6)

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.62 * row_count + 1.4)))
    _add_bad_band_spans(ax, plot_rows[0][3])
    for idx, (label, wn_plot, mean_plot, _) in enumerate(plot_rows):
        offset = (row_count - idx - 1) * offset_step
        ax.plot(wn_plot, mean_plot + offset, color=colors[idx % len(colors)], linewidth=1.0)
        ax.text(
            -0.01,
            offset,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_xlim([groups[0]["wn"].min(), groups[0]["wn"].max()])
    ax.tick_params(axis="y", which="both", labelleft=False, left=False)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

