from dataclasses import dataclass

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


@dataclass(frozen=True)
class CosmicRayStats:
    """记录单谱宇宙射线最终归属替换点数，int(stats) 返回唯一替换点数"""

    narrow: int = 0
    peak: int = 0

    @property
    def total(self):
        return int(self.narrow) + int(self.peak)

    def __int__(self):
        return self.total


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

    for iteration in range(1, int(niter) + 1):
        matrix_w = sparse.diags(weights, 0)
        baseline = spsolve((matrix_w + penalty).tocsc(), weights * y)
        residual = y - baseline
        negative_mask = residual < 0
        if valid_mask is not None:
            negative_mask &= valid_mask

        negative_sum = float(np.sum(np.abs(residual[negative_mask])))
        if negative_sum <= 1e-3 * float(np.sum(np.abs(y))):
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


def _odd_window_points(points, min_points=3):
    """把点数窗口规整成局部 median 需要的奇数点数"""
    points = max(int(min_points), int(round(float(points))))
    if points % 2 == 0:
        points += 1
    return points


def _pad_points(points):
    """把边缘扩展点数规整成非负整数"""
    return max(0, int(round(float(points))))


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
        nonzero_abs = centered_abs[centered_abs > 1e-8]
        if nonzero_abs.size > 0:
            scale = 1.4826 * float(np.median(nonzero_abs))
    if scale <= 1e-8:
        scale = float(np.std(residual_valid))
    if scale <= 1e-8:
        return None

    return (residual - center) / scale


def _iter_true_segments(mask):
    start = None
    for idx, enabled in enumerate(mask):
        if enabled and start is None:
            start = idx
        elif not enabled and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask)


def _collect_replacement_anchors(valid_mask, anchor_bad_mask, start, end, count):
    left = []
    idx = start - 1
    while idx >= 0 and len(left) < count:
        if valid_mask[idx] and not anchor_bad_mask[idx]:
            left.append(idx)
        idx -= 1

    right = []
    idx = end
    while idx < valid_mask.size and len(right) < count:
        if valid_mask[idx] and not anchor_bad_mask[idx]:
            right.append(idx)
        idx += 1

    return (
        np.asarray(left[::-1], dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


def _linear_segment_replacement(cleaned, fallback, start, end, left_indices, right_indices):
    if left_indices.size > 0 and right_indices.size > 0:
        left = int(left_indices[-1])
        right = int(right_indices[0])
        if right > left:
            return np.interp(
                np.arange(start, end, dtype=np.float32),
                np.array([left, right], dtype=np.float32),
                np.array([cleaned[left], cleaned[right]], dtype=np.float32),
            )
    return fallback[start:end]


def _robust_local_scale(values):
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


def _damped_segment_replacement(cleaned, fallback, start, end, left_indices, right_indices):
    trend = np.asarray(fallback[start:end], dtype=np.float64)
    original = np.asarray(cleaned[start:end], dtype=np.float64)
    if trend.size == 0 or not np.all(np.isfinite(trend)):
        return None

    left_values = cleaned[left_indices].astype(np.float64)
    right_values = cleaned[right_indices].astype(np.float64)
    anchor_values = np.concatenate([left_values, right_values])
    anchor_values = anchor_values[np.isfinite(anchor_values)]
    if anchor_values.size < 2:
        return None

    local_min = float(np.min(anchor_values))
    local_max = float(np.max(anchor_values))
    local_range = max(local_max - local_min, 0.0)
    diff_parts = []
    if left_values.size >= 2:
        diff_parts.append(np.diff(left_values))
    if right_values.size >= 2:
        diff_parts.append(np.diff(right_values))
    diff_scale = _robust_local_scale(np.concatenate(diff_parts)) if diff_parts else 0.0
    detail_limit = max(
        min(3.0 * diff_scale, 0.25 * local_range) if diff_scale > 0.0 else 0.0,
        0.01 * float(np.median(np.abs(anchor_values))),
        1e-6,
    )

    # 保留原段相对趋势的形态；边缘梯度越异常、相对趋势幅度越大，压缩越强。
    detail = original - trend
    segment_left = left_values[-1] if left_values.size > 0 and np.isfinite(left_values[-1]) else original[0]
    segment_right = right_values[0] if right_values.size > 0 and np.isfinite(right_values[0]) else original[-1]
    extended = np.concatenate([[segment_left], original, [segment_right]])
    diffs = np.abs(np.diff(extended))
    point_gradient = np.maximum(diffs[:-1], diffs[1:])

    anchor_level = float(np.median(np.abs(anchor_values)))
    gradient_limit = max(
        3.0 * diff_scale if diff_scale > 0.0 else 0.0,
        0.10 * local_range,
        0.01 * anchor_level,
        1e-6,
    )
    positive_limit = max(detail_limit, 1e-6)
    negative_limit = max(2.0 * detail_limit, 1e-6)

    gradient_scale = np.minimum(1.0, gradient_limit / np.maximum(point_gradient, 1e-8))
    amplitude_limit = np.where(detail >= 0.0, positive_limit, negative_limit)
    amplitude_scale = np.minimum(1.0, amplitude_limit / np.maximum(np.abs(detail), 1e-8))
    detail_scale = np.minimum(gradient_scale, amplitude_scale)

    replacement = trend + detail * detail_scale
    replacement = np.clip(
        replacement,
        local_min - 2.0 * detail_limit,
        local_max + detail_limit,
    )
    return replacement.astype(np.float32, copy=False)


def _replace_segment(cleaned, fallback, start, end, valid_mask, anchor_bad_mask=None):
    segment_valid = valid_mask[start:end]
    if not segment_valid.any():
        return

    if anchor_bad_mask is None or anchor_bad_mask.shape != cleaned.shape:
        anchor_bad_mask = np.zeros(cleaned.shape, dtype=bool)

    # peak 段保留原段相对趋势的形态，只压低异常起伏；失败时回退线性插值。
    left_indices, right_indices = _collect_replacement_anchors(
        valid_mask,
        anchor_bad_mask,
        start,
        end,
        count=5,
    )
    replacement = _damped_segment_replacement(cleaned, fallback, start, end, left_indices, right_indices)
    if replacement is None:
        replacement = _linear_segment_replacement(
            cleaned,
            fallback,
            start,
            end,
            left_indices,
            right_indices,
        )

    segment = cleaned[start:end].copy()
    segment[segment_valid] = replacement[segment_valid]
    cleaned[start:end] = segment


def _merge_segments_with_small_gaps(mask, max_gap_points=1):
    segments = list(_iter_true_segments(mask))
    if not segments:
        return []

    merged = [list(segments[0])]
    max_gap_points = max(0, int(max_gap_points))
    for start, end in segments[1:]:
        last = merged[-1]
        if start - last[1] <= max_gap_points:
            last[1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _remove_peak_segments(
    cleaned,
    valid_mask,
    peak_window_points,
    core_z,
    expand_z,
    expand_gap_points,
    max_width_points,
    mean_z_min,
    pad_points,
):
    window = _odd_window_points(peak_window_points)
    fallback = _median_filter_1d(cleaned, window)
    residual = cleaned - fallback
    z_score = _residual_z_score(residual, valid_mask)
    if z_score is None:
        return 0, np.zeros(cleaned.shape, dtype=bool)

    core_mask = valid_mask & (residual > 0) & (z_score >= float(core_z))
    if not core_mask.any():
        return 0, np.zeros(cleaned.shape, dtype=bool)

    expand_mask = valid_mask & (residual > 0) & (z_score >= float(expand_z))
    if not expand_mask.any():
        return 0, np.zeros(cleaned.shape, dtype=bool)

    pad_points = _pad_points(pad_points)
    expand_gap_points = _pad_points(expand_gap_points)
    replaced_mask = np.zeros(cleaned.shape, dtype=bool)

    for start, end in _merge_segments_with_small_gaps(expand_mask, max_gap_points=expand_gap_points):
        if not core_mask[start:end].any():
            continue

        segment_valid = valid_mask[start:end]
        if not segment_valid.any():
            continue

        valid_indices = np.where(segment_valid)[0] + start
        first_idx = int(valid_indices[0])
        last_idx = int(valid_indices[-1])
        width_points = last_idx - first_idx + 1
        if width_points > int(max_width_points):
            continue

        z_segment = z_score[start:end][segment_valid]
        mean_z = float(np.mean(np.maximum(z_segment, 0.0)))
        if mean_z < float(mean_z_min):
            continue

        replace_start = max(0, first_idx - pad_points)
        replace_end = min(cleaned.size, last_idx + pad_points + 1)
        segment_replace_mask = np.zeros(cleaned.shape, dtype=bool)
        segment_replace_mask[replace_start:replace_end] = True
        segment_replace_mask &= valid_mask

        _replace_segment(cleaned, fallback, replace_start, replace_end, valid_mask, expand_mask)
        replaced_mask |= segment_replace_mask

    return int(replaced_mask.sum()), replaced_mask


def remove_cosmic_rays(
    sp,
    window_points=7,
    threshold=7.0,
    max_iter=2,
    valid_mask=None,
    peak_prominence_z=9.0,
    peak_window_points=21,
    peak_expand_z=3.0,
    peak_expand_gap_points=2,
    peak_width_max_points=21,
    peak_mean_z_min=3.0,
    peak_pad_points=2,
):
    """
    去除宇宙射线尖峰和短正异常段

    narrow 处理极端点；peak 在 narrow 后的局部 median 正残差上处理短异常段。
    """
    cleaned = np.asarray(sp, dtype=np.float32).copy()
    if cleaned.size < 3 or max_iter <= 0:
        return cleaned, CosmicRayStats()

    if valid_mask is None:
        valid_mask = np.ones(cleaned.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != cleaned.shape:
            valid_mask = np.ones(cleaned.shape, dtype=bool)

    narrow_mask = np.zeros(cleaned.shape, dtype=bool)
    narrow_window = _odd_window_points(window_points)
    for _ in range(int(max_iter)):
        local_median = _median_filter_1d(cleaned, narrow_window)
        residual = cleaned - local_median
        z_score = _residual_z_score(residual, valid_mask)
        if z_score is None:
            break

        spike_mask = valid_mask & (z_score > float(threshold))
        if not spike_mask.any():
            break

        narrow_mask |= spike_mask
        cleaned[spike_mask] = local_median[spike_mask]

    _, peak_mask = _remove_peak_segments(
        cleaned,
        valid_mask,
        peak_window_points=peak_window_points,
        core_z=peak_prominence_z,
        expand_z=peak_expand_z,
        expand_gap_points=peak_expand_gap_points,
        max_width_points=peak_width_max_points,
        mean_z_min=peak_mean_z_min,
        pad_points=peak_pad_points,
    )

    narrow_final_mask = narrow_mask & ~peak_mask

    return cleaned, CosmicRayStats(
        narrow=int(narrow_final_mask.sum()),
        peak=int(peak_mask.sum()),
    )



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
    cosmic_ray_peak_prominence_z=9.0,
    cosmic_ray_peak_window_points=21,
    cosmic_ray_peak_expand_z=3.0,
    cosmic_ray_peak_expand_gap_points=2,
    cosmic_ray_peak_width_max_points=21,
    cosmic_ray_peak_mean_z_min=3.0,
    cosmic_ray_peak_pad_points=2,
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
            peak_prominence_z=cosmic_ray_peak_prominence_z,
            peak_window_points=cosmic_ray_peak_window_points,
            peak_expand_z=cosmic_ray_peak_expand_z,
            peak_expand_gap_points=cosmic_ray_peak_expand_gap_points,
            peak_width_max_points=cosmic_ray_peak_width_max_points,
            peak_mean_z_min=cosmic_ray_peak_mean_z_min,
            peak_pad_points=cosmic_ray_peak_pad_points,
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


def _insert_nan_gaps(wn, *values):
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


def save_mean_plot(wn, spectra, out_path, norm_method, bad_bands, title):
    """保存一组光谱的均值谱图，并在图上标出坏波段区域"""
    bad_bands = normalize_bad_bands(bad_bands)
    spectra_norm = normalize_for_plot(spectra, norm_method)
    mean_spec = np.mean(spectra_norm, axis=0)
    q10_spec = np.quantile(spectra_norm, 0.10, axis=0)
    q90_spec = np.quantile(spectra_norm, 0.90, axis=0)
    keep_mask = build_valid_mask(wn, bad_bands)
    if keep_mask is not None and keep_mask.shape == mean_spec.shape:
        mean_spec = np.where(keep_mask, mean_spec, np.nan)
        q10_spec = np.where(keep_mask, q10_spec, np.nan)
        q90_spec = np.where(keep_mask, q90_spec, np.nan)
    wn_plot, mean_plot, q10_plot, q90_plot = _insert_nan_gaps(
        wn,
        mean_spec,
        q10_spec,
        q90_spec,
    )

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

    plt.plot(wn_plot, mean_plot, label=f"Mean spectrum {norm_method}")
    plt.fill_between(wn_plot, q10_plot, q90_plot, alpha=0.3)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel(f"{norm_method} intensity")
    plt.xlim([wn.min(), wn.max()])
    plt.legend(handles=[std_proxy] + plt.gca().get_legend_handles_labels()[0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
