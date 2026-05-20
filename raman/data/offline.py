from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy import sparse
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.sparse.linalg import spsolve

from raman.data.spectrum import (
    build_valid_mask,
    minmax_normalize,
    normalize_bad_bands,
    snv,
)


@dataclass(frozen=True)
class CosmicRayStats:
    """记录单谱宇宙射线清理数量，int(stats) 返回总替换点数"""

    narrow: int = 0
    peak: int = 0
    residual: int = 0

    @property
    def total(self):
        return int(self.narrow) + int(self.peak) + int(self.residual)

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


def _median_step_cm(wn):
    """估计当前波数轴的采样步长"""
    if wn is None:
        return 1.0
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size < 2:
        return 1.0
    diffs = np.diff(wn)
    diffs = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-8)])
    if diffs.size == 0:
        return 1.0
    return max(float(np.median(diffs)), 1e-6)


def _cm_to_odd_window_points(width_cm, wn, min_points=3):
    """把 cm^-1 窗口换算成局部滤波需要的奇数点数"""
    step = _median_step_cm(wn)
    points = max(int(min_points), int(round(float(width_cm) / step)))
    if points % 2 == 0:
        points += 1
    return points


def _cm_to_pad_points(width_cm, wn):
    """把 cm^-1 扩展宽度换算成点数"""
    step = _median_step_cm(wn)
    return max(0, int(round(float(width_cm) / step)))


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


def _replace_segment(cleaned, fallback, start, end, valid_mask):
    segment_valid = valid_mask[start:end]
    if not segment_valid.any():
        return

    left = start - 1
    while left >= 0 and not valid_mask[left]:
        left -= 1

    right = end
    while right < cleaned.size and not valid_mask[right]:
        right += 1

    if left >= 0 and right < cleaned.size and right > left:
        x = np.arange(start, end, dtype=np.float32)
        replacement = np.interp(
            x,
            np.array([left, right], dtype=np.float32),
            np.array([cleaned[left], cleaned[right]], dtype=np.float32),
        )
    else:
        replacement = fallback[start:end]

    segment = cleaned[start:end].copy()
    segment[segment_valid] = replacement[segment_valid]
    cleaned[start:end] = segment


def _merge_intervals(intervals):
    intervals = list(intervals)
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _expand_mask(mask, pad_points, valid_mask=None):
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(mask.shape, dtype=bool)
    pad_points = max(0, int(pad_points))
    for start, end in _iter_true_segments(mask):
        result[max(0, start - pad_points): min(mask.size, end + pad_points)] = True
    if valid_mask is not None:
        result &= np.asarray(valid_mask, dtype=bool)
    return result


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


def _estimate_noise_from_diff(cleaned, valid_mask):
    valid_values = cleaned[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size < 3:
        return 0.0

    diff = np.diff(valid_values)
    if diff.size == 0:
        return 0.0

    mad = np.median(np.abs(diff - np.median(diff)))
    scale = 1.4826 * mad / np.sqrt(2.0)
    if scale <= 1e-8:
        scale = float(np.std(diff) / np.sqrt(2.0))
    return max(scale, 0.0)


def _remove_peak_morphology_spikes(
    cleaned,
    valid_mask,
    prominence_z,
    width_max_cm,
    ratio_z_per_cm,
    pad_cm,
    rel_height,
    wn=None,
):
    empty_mask = np.zeros(cleaned.shape, dtype=bool)
    noise = _estimate_noise_from_diff(cleaned, valid_mask)
    if noise <= 1e-8:
        return 0, empty_mask

    peaks, _ = find_peaks(cleaned)
    if peaks.size == 0:
        return 0, empty_mask

    peaks = peaks[valid_mask[peaks]]
    if peaks.size == 0:
        return 0, empty_mask

    prominences = peak_prominences(cleaned, peaks)[0]
    widths, _, left_ips, right_ips = peak_widths(
        cleaned,
        peaks,
        rel_height=float(rel_height),
    )

    step = _median_step_cm(wn)

    width_cm = widths * step
    prominence_score = prominences / noise
    ratio = prominence_score / np.maximum(width_cm, step)

    selected = (
        (prominence_score >= float(prominence_z))
        & (width_cm <= float(width_max_cm))
        & (ratio >= float(ratio_z_per_cm))
    )
    if not selected.any():
        return 0, empty_mask

    fallback = _median_filter_1d(cleaned, 7)
    pad_points = _cm_to_pad_points(pad_cm, wn)
    intervals = []
    for left_ip, right_ip in zip(left_ips[selected], right_ips[selected]):
        start = max(0, int(np.floor(left_ip)) - pad_points)
        end = min(cleaned.size, int(np.ceil(right_ip)) + pad_points + 1)
        if start < end:
            intervals.append((start, end))

    replaced_mask = np.zeros(cleaned.shape, dtype=bool)
    for start, end in _merge_intervals(intervals):
        segment_mask = np.zeros(cleaned.shape, dtype=bool)
        segment_mask[start:end] = True
        segment_mask &= valid_mask
        if not segment_mask.any():
            continue

        start_idx = int(np.where(segment_mask)[0][0])
        end_idx = int(np.where(segment_mask)[0][-1]) + 1
        _replace_segment(cleaned, fallback, start_idx, end_idx, valid_mask)
        replaced_mask[start_idx:end_idx] |= valid_mask[start_idx:end_idx]

    return int(replaced_mask.sum()), replaced_mask


def _remove_anchored_residual_plateaus(
    cleaned,
    valid_mask,
    anchor_mask,
    window_cm,
    threshold_z,
    min_width_cm,
    max_width_cm,
    anchor_pad_cm,
    replace_pad_cm,
    min_area_z_cm,
    max_points_fraction,
    wn=None,
):
    if not anchor_mask.any():
        return 0, np.zeros(cleaned.shape, dtype=bool)

    step = _median_step_cm(wn)
    window = _cm_to_odd_window_points(window_cm, wn)
    fallback = _median_filter_1d(cleaned, window)
    residual = cleaned - fallback
    z_score = _residual_z_score(residual, valid_mask)
    if z_score is None:
        return 0, np.zeros(cleaned.shape, dtype=bool)

    anchor_pad_points = _cm_to_pad_points(anchor_pad_cm, wn)
    anchor_region = _expand_mask(anchor_mask, anchor_pad_points, valid_mask)
    candidate_mask = (
        valid_mask
        & anchor_region
        & (residual > 0)
        & (z_score >= float(threshold_z))
    )
    if not candidate_mask.any():
        return 0, np.zeros(cleaned.shape, dtype=bool)

    replace_pad_points = _cm_to_pad_points(replace_pad_cm, wn)
    max_points = max(1, int(np.floor(valid_mask.sum() * float(max_points_fraction))))
    replaced_mask = np.zeros(cleaned.shape, dtype=bool)

    for start, end in _merge_segments_with_small_gaps(candidate_mask, max_gap_points=1):
        segment_valid = valid_mask[start:end]
        if not segment_valid.any():
            continue

        valid_indices = np.where(segment_valid)[0] + start
        first_idx = int(valid_indices[0])
        last_idx = int(valid_indices[-1])
        anchor_start = first_idx
        while anchor_start > 0 and anchor_region[anchor_start - 1]:
            anchor_start -= 1

        anchor_end = last_idx + 1
        while anchor_end < cleaned.size and anchor_region[anchor_end]:
            anchor_end += 1

        component_anchor_mask = anchor_mask[anchor_start:anchor_end] & valid_mask[anchor_start:anchor_end]
        if int(component_anchor_mask.sum()) < 2:
            continue

        width_cm = abs(float(wn[last_idx] - wn[first_idx])) + step if wn is not None else (last_idx - first_idx + 1) * step
        if width_cm < float(min_width_cm) or width_cm > float(max_width_cm):
            continue

        z_segment = z_score[start:end][segment_valid]
        peak_z = float(np.max(z_segment))
        area_z_cm = float(np.sum(np.maximum(z_segment, 0.0)) * step)
        if peak_z < float(threshold_z) or area_z_cm < float(min_area_z_cm):
            continue

        replace_start = max(0, anchor_start - replace_pad_points)
        replace_end = min(cleaned.size, anchor_end + replace_pad_points)
        segment_replace_mask = np.zeros(cleaned.shape, dtype=bool)
        segment_replace_mask[replace_start:replace_end] = True
        segment_replace_mask &= valid_mask

        new_points = segment_replace_mask & ~replaced_mask
        if int(replaced_mask.sum() + new_points.sum()) > max_points:
            continue

        _replace_segment(cleaned, fallback, replace_start, replace_end, valid_mask)
        replaced_mask |= segment_replace_mask

    return int(replaced_mask.sum()), replaced_mask


def remove_cosmic_rays(
    sp,
    window_cm=10.0,
    threshold=8.0,
    max_iter=2,
    valid_mask=None,
    wn=None,
    peak_prominence_z=8.0,
    peak_width_max_cm=10.0,
    peak_ratio_z_per_cm=4.0,
    peak_pad_cm=2.0,
    peak_rel_height=0.5,
    residual_threshold_z=10.0,
    residual_max_points_fraction=0.03,
):
    """
    去除宇宙射线尖峰

    只检测相对局部中值异常抬高的正向异常峰；调用方可通过 valid_mask 限制检测范围
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
    narrow_window = _cm_to_odd_window_points(window_cm, wn)
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

    peak_replaced, peak_mask = _remove_peak_morphology_spikes(
        cleaned,
        valid_mask,
        prominence_z=peak_prominence_z,
        width_max_cm=peak_width_max_cm,
        ratio_z_per_cm=peak_ratio_z_per_cm,
        pad_cm=peak_pad_cm,
        rel_height=peak_rel_height,
        wn=wn,
    )

    residual_min_width_cm = float(peak_pad_cm)
    residual_window_cm = float(peak_width_max_cm) * 2
    residual_max_width_cm = float(peak_width_max_cm) * 2
    residual_anchor_pad_cm = float(peak_pad_cm) * 3.0
    residual_replace_pad_cm = float(peak_pad_cm)
    residual_min_area_z_cm = float(residual_threshold_z) * residual_min_width_cm

    residual_replaced, _ = _remove_anchored_residual_plateaus(
        cleaned,
        valid_mask,
        anchor_mask=(narrow_mask | peak_mask),
        window_cm=residual_window_cm,
        threshold_z=residual_threshold_z,
        min_width_cm=residual_min_width_cm,
        max_width_cm=residual_max_width_cm,
        anchor_pad_cm=residual_anchor_pad_cm,
        replace_pad_cm=residual_replace_pad_cm,
        min_area_z_cm=residual_min_area_z_cm,
        max_points_fraction=residual_max_points_fraction,
        wn=wn,
    )

    return cleaned, CosmicRayStats(
        narrow=int(narrow_mask.sum()),
        peak=int(peak_replaced),
        residual=int(residual_replaced),
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
    cosmic_ray_window_cm=10.0,
    cosmic_ray_threshold=8.0,
    cosmic_ray_max_iter=2,
    cosmic_ray_peak_prominence_z=8.0,
    cosmic_ray_peak_width_max_cm=10.0,
    cosmic_ray_peak_ratio_z_per_cm=4.0,
    cosmic_ray_peak_pad_cm=2.0,
    cosmic_ray_peak_rel_height=0.5,
    cosmic_ray_residual_threshold_z=10.0,
    cosmic_ray_residual_max_points_fraction=0.03,
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
            window_cm=cosmic_ray_window_cm,
            threshold=cosmic_ray_threshold,
            max_iter=cosmic_ray_max_iter,
            valid_mask=None,
            wn=wn,
            peak_prominence_z=cosmic_ray_peak_prominence_z,
            peak_width_max_cm=cosmic_ray_peak_width_max_cm,
            peak_ratio_z_per_cm=cosmic_ray_peak_ratio_z_per_cm,
            peak_pad_cm=cosmic_ray_peak_pad_cm,
            peak_rel_height=cosmic_ray_peak_rel_height,
            residual_threshold_z=cosmic_ray_residual_threshold_z,
            residual_max_points_fraction=cosmic_ray_residual_max_points_fraction,
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
