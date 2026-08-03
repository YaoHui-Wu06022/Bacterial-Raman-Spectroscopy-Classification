"""离线单谱清洗的基础算子。"
此模块只处理数值数组，不负责数据集目录遍历、日志写入或绘图。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .bands import build_valid_mask, normalize_bad_bands
from .filters import median_filter_1d, odd_window_points


def asls_baseline(spectrum, lam: float = 1e5, p: float = 0.01, niter: int = 10, valid_mask=None):
    """使用 AsLS 估计基线；坏波段可通过掩码完全排除。"""
    length = len(spectrum)
    difference = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    weights = np.ones(length)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0

    for _ in range(niter):
        weight_matrix = sparse.diags(weights, 0)
        baseline = spsolve((weight_matrix + lam * (difference.T @ difference)).tocsc(), weights * spectrum)
        if valid_mask is None:
            weights = np.where(spectrum > baseline, p, 1 - p)
        else:
            next_weights = np.zeros(length)
            next_weights[valid_mask] = np.where(
                spectrum[valid_mask] > baseline[valid_mask], p, 1 - p
            )
            weights = next_weights
    return baseline


def arpls_baseline(spectrum, lam: float = 1e5, niter: int = 15, valid_mask=None):
    """使用 arPLS 估计基线，对正向拉曼峰保留较低权重。"""
    values = np.asarray(spectrum, dtype=np.float64)
    length = len(values)
    difference = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    penalty = lam * (difference.T @ difference)
    weights = np.ones(length, dtype=np.float64)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0

    for _ in range(int(niter)):
        baseline = spsolve((sparse.diags(weights, 0) + penalty).tocsc(), weights * values)
        residual = values - baseline
        negative = residual[residual < 0]
        if negative.size < 2:
            break
        mean_negative = float(np.mean(negative))
        std_negative = float(np.std(negative))
        if std_negative <= 1e-12:
            break
        logits = 2.0 * (residual - (2.0 * std_negative - mean_negative)) / std_negative
        next_weights = 1.0 / (1.0 + np.exp(np.clip(logits, -60.0, 60.0)))
        if valid_mask is not None:
            next_weights[~valid_mask] = 0.0
        if np.linalg.norm(next_weights - weights) / max(np.linalg.norm(weights), 1e-12) < 1e-3:
            weights = next_weights
            break
        weights = next_weights
    return baseline.astype(np.float32, copy=False)


def airpls_baseline(spectrum, lam: float = 1e5, niter: int = 15, valid_mask=None):
    """使用 airPLS 估计基线，迭代提高负残差位置的拟合权重。"""
    values = np.asarray(spectrum, dtype=np.float64)
    length = len(values)
    difference = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
    penalty = lam * (difference.T @ difference)
    weights = np.ones(length, dtype=np.float64)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0
        stop_scale = float(np.sum(np.abs(values[valid_mask])))
    else:
        stop_scale = float(np.sum(np.abs(values)))
    stop_scale = max(stop_scale, 1e-12)

    for iteration in range(1, int(niter) + 1):
        baseline = spsolve((sparse.diags(weights, 0) + penalty).tocsc(), weights * values)
        residual = values - baseline
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
    method: str = "asls",
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 15,
    valid_mask=None,
):
    """按方法名选择基线估计算法。"""
    method = str(method).lower()
    if method == "asls":
        return asls_baseline(spectrum, lam=lam, p=p, niter=niter, valid_mask=valid_mask)
    if method == "arpls":
        return arpls_baseline(spectrum, lam=lam, niter=niter, valid_mask=valid_mask)
    if method == "airpls":
        return airpls_baseline(spectrum, lam=lam, niter=niter, valid_mask=valid_mask)
    raise ValueError(f"Unknown baseline method: {method}")


@dataclass(frozen=True)
class CosmicRayStats:
    """记录一条光谱被局部中值替换的宇宙射线尖峰点数。"""

    cosmic_ray: int = 0

    def __int__(self) -> int:
        return int(self.cosmic_ray)


def _residual_z_score(residual, valid_mask):
    """使用 MAD 估计残差尺度，生成稳健 z 分数。"""
    valid_values = residual[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        return None
    center = np.median(valid_values)
    centered_abs = np.abs(valid_values - center)
    scale = 1.4826 * np.median(centered_abs)
    if scale <= 1e-8:
        scale = float(np.std(valid_values))
    if scale <= 1e-8:
        nonzero = centered_abs[centered_abs > 1e-8]
        if nonzero.size:
            scale = 1.4826 * float(np.median(nonzero))
    if scale <= 1e-8:
        return None
    return (residual - center) / scale


def remove_cosmic_rays(
    spectrum,
    window_points: int = 7,
    threshold: float = 7.0,
    max_iter: int = 2,
    valid_mask=None,
):
    """使用局部 median/MAD 替换正向宇宙射线尖峰。"""
    cleaned = np.asarray(spectrum, dtype=np.float32).copy()
    if cleaned.size < 3 or max_iter <= 0:
        return cleaned, CosmicRayStats()
    if valid_mask is None:
        valid_mask = np.ones(cleaned.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != cleaned.shape:
            valid_mask = np.ones(cleaned.shape, dtype=bool)

    replaced_mask = np.zeros(cleaned.shape, dtype=bool)
    window = odd_window_points(window_points)
    for _ in range(int(max_iter)):
        local_median = median_filter_1d(cleaned, window)
        z_score = _residual_z_score(cleaned - local_median, valid_mask)
        if z_score is None:
            break
        spike_mask = valid_mask & (z_score > float(threshold))
        if not spike_mask.any():
            break
        replaced_mask |= spike_mask
        cleaned[spike_mask] = local_median[spike_mask]
    return cleaned, CosmicRayStats(cosmic_ray=int(replaced_mask.sum()))


def preprocess_single_spectrum(
    wavenumbers,
    spectrum,
    cut_min,
    cut_max,
    reference_wavenumbers,
    bad_bands,
    baseline_lam,
    baseline_asls_p,
    baseline_max_iter,
    baseline_fit_min=None,
    baseline_fit_max=None,
    baseline_method: str = "asls",
    cosmic_ray_enable: bool = False,
    cosmic_ray_window_points: int = 7,
    cosmic_ray_threshold: float = 7.0,
    cosmic_ray_max_iter: int = 2,
):
    """完成单谱尖峰修复、基线校正、裁剪、坏段删除和插值。"""
    normalized_bands = normalize_bad_bands(bad_bands)
    cleaned = np.asarray(spectrum, dtype=np.float32)
    cosmic_stats = CosmicRayStats()
    if cosmic_ray_enable:
        cleaned, cosmic_stats = remove_cosmic_rays(
            cleaned,
            window_points=cosmic_ray_window_points,
            threshold=cosmic_ray_threshold,
            max_iter=cosmic_ray_max_iter,
        )

    fit_min = cut_min if baseline_fit_min is None else float(baseline_fit_min)
    fit_max = cut_max if baseline_fit_max is None else float(baseline_fit_max)
    fit_mask = (wavenumbers >= fit_min) & (wavenumbers <= fit_max)
    fitted_wavenumbers = wavenumbers[fit_mask]
    fitted_spectrum = cleaned[fit_mask]
    valid_fit_mask = build_valid_mask(fitted_wavenumbers, normalized_bands)
    if fitted_wavenumbers.size < 10:
        return None, None, cosmic_stats

    baseline = estimate_baseline(
        fitted_spectrum,
        method=baseline_method,
        lam=baseline_lam,
        p=baseline_asls_p,
        niter=baseline_max_iter,
        valid_mask=valid_fit_mask,
    )
    corrected = fitted_spectrum - baseline
    crop_mask = (fitted_wavenumbers >= cut_min) & (fitted_wavenumbers <= cut_max)
    cropped_wavenumbers = fitted_wavenumbers[crop_mask]
    cropped_spectrum = corrected[crop_mask]
    target_axis = reference_wavenumbers

    if normalized_bands:
        source_mask = build_valid_mask(cropped_wavenumbers, normalized_bands)
        cropped_wavenumbers = cropped_wavenumbers[source_mask]
        cropped_spectrum = cropped_spectrum[source_mask]
        target_mask = build_valid_mask(target_axis, normalized_bands)
        target_axis = target_axis[target_mask]
    if cropped_wavenumbers.size < 10 or target_axis.size == 0:
        return None, None, cosmic_stats
    return target_axis, np.interp(target_axis, cropped_wavenumbers, cropped_spectrum), cosmic_stats
