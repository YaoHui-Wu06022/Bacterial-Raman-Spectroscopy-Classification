"""离线单谱清洗的基础算子。

此模块只处理数值数组，不负责数据集目录遍历、日志写入或绘图。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

from .bad_bands import build_valid_mask, normalize_bad_bands


# 基线拟合前只压制单点噪声，避免 airPLS 重加权在相近谱之间跳到不同不动点。
BASELINE_INPUT_SMOOTH_WINDOW = 5
# 多项式基线的固定阶数，配合非对称迭代加权抑制峰位被拉高。
BASELINE_POLY_DEGREE = 5


@lru_cache(maxsize=32)
def _build_baseline_penalty(length: int, lam: float):
    """缓存相同谱长和惩罚系数对应的二阶差分稀疏矩阵。"""
    difference = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, 1, 2],
        shape=(length - 2, length),
        dtype=np.float64,
    )
    return (float(lam) * (difference.T @ difference)).tocsc()


def asls_baseline(spectrum, lam: float = 1e5, p: float = 0.01, niter: int = 10, valid_mask=None):
    """使用 AsLS 估计基线；坏波段可通过掩码完全排除。"""
    length = len(spectrum)
    penalty = _build_baseline_penalty(length, float(lam))
    weights = np.ones(length)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        weights[~valid_mask] = 0.0

    for _ in range(niter):
        weight_matrix = sparse.diags(weights, 0)
        baseline = spsolve((weight_matrix + penalty).tocsc(), weights * spectrum)
        if valid_mask is None:
            weights = np.where(spectrum > baseline, p, 1 - p)
        else:
            next_weights = np.zeros(length)
            next_weights[valid_mask] = np.where(
                spectrum[valid_mask] > baseline[valid_mask], p, 1 - p
            )
            weights = next_weights
    return baseline


def airpls_baseline(spectrum, lam: float = 1e5, niter: int = 15, valid_mask=None):
    """使用 airPLS 估计基线，迭代提高负残差位置的拟合权重。"""
    values = np.asarray(spectrum, dtype=np.float64)
    length = len(values)
    penalty = _build_baseline_penalty(length, float(lam))
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
    if method == "airpls":
        return airpls_baseline(spectrum, lam=lam, niter=niter, valid_mask=valid_mask)
    if method in {"poly", "polynomial"}:
        return polynomial_baseline(spectrum, p=p, niter=niter, valid_mask=valid_mask)
    raise ValueError(f"Unknown baseline method: {method}")


def polynomial_baseline(
    spectrum,
    degree: int = BASELINE_POLY_DEGREE,
    p: float = 0.01,
    niter: int = 15,
    valid_mask=None,
):
    """使用非对称迭代加权的多项式拟合估计基线。

    每轮只给高于当前曲线的点降权（权重 p），低于曲线的点保持 1-p，
    使多项式贴合下包络而不是被峰位拉起；坏段通过 valid_mask 完全排除。
    """
    values = np.asarray(spectrum, dtype=np.float64)
    length = len(values)
    if length <= degree:
        raise ValueError(f"谱长 {length} 不足多项式阶数 {degree} + 1")
    positions = np.linspace(-1.0, 1.0, length)
    weights = np.ones(length, dtype=np.float64)
    fit_mask = np.ones(length, dtype=bool)
    if valid_mask is not None:
        fit_mask = np.asarray(valid_mask, dtype=bool)
        weights[~fit_mask] = 0.0
    baseline = np.zeros(length, dtype=np.float64)
    for _ in range(int(niter)):
        coefficients = np.polyfit(
            positions[fit_mask],
            values[fit_mask],
            int(degree),
            w=np.sqrt(weights[fit_mask]),
        )
        baseline = np.polyval(coefficients, positions)
        residual = values - baseline
        next_weights = np.where(residual > 0.0, float(p), 1.0 - float(p))
        weights = np.where(fit_mask, next_weights, 0.0)
    return baseline.astype(np.float32, copy=False)


def smooth_baseline_input(spectrum):
    """给基线估计提供轻度平滑输入；后续相减仍使用原始拟合段。"""
    values = np.asarray(spectrum, dtype=np.float64)
    if values.size < BASELINE_INPUT_SMOOTH_WINDOW:
        return values
    return savgol_filter(
        values,
        BASELINE_INPUT_SMOOTH_WINDOW,
        3,
        mode="interp",
    )


def interpolate_valid_bands(wavenumbers, spectrum, target_axis, bad_bands):
    """分有效波段插值，边界取本段端点值，禁止跨坏段借用另一侧采样点。"""
    bands = normalize_bad_bands(bad_bands)
    if not bands:
        return np.interp(target_axis, wavenumbers, spectrum)
    wavenumbers = np.asarray(wavenumbers)
    spectrum = np.asarray(spectrum)
    target_axis = np.asarray(target_axis)
    source_mask = build_valid_mask(wavenumbers, bands)
    if not build_valid_mask(target_axis, bands).all():
        raise ValueError("插值目标轴不能包含坏波段")
    upper_edges = sorted(upper for _, upper in bands)
    source_groups = np.searchsorted(upper_edges, wavenumbers)
    target_groups = np.searchsorted(upper_edges, target_axis)
    result = np.empty(len(target_axis), dtype=np.float64)
    for group in np.unique(target_groups):
        source = source_mask & (source_groups == group)
        target = target_groups == group
        if not source.any():
            raise ValueError("插值目标波段没有有效源采样点")
        result[target] = np.interp(target_axis[target], wavenumbers[source], spectrum[source])
    return result


def fit_baseline_spectrum(
    wavenumbers,
    spectrum,
    *,
    baseline_fit_min: float,
    baseline_fit_max: float,
    bad_bands,
    baseline_lam: float,
    baseline_asls_p: float,
    baseline_max_iter: int,
    baseline_method: str = "asls",
):
    """完成校验与基线估计，返回拟合区间上的校正谱与基线。"""
    wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float32)
    if wavenumbers.ndim != 1 or spectrum.ndim != 1:
        raise ValueError(
            f"光谱输入必须是一维数组：wavenumbers.ndim={wavenumbers.ndim}, "
            f"spectrum.ndim={spectrum.ndim}"
        )
    if wavenumbers.size != spectrum.size:
        raise ValueError(
            f"波数与强度长度不一致：wavenumbers={wavenumbers.size}, "
            f"spectrum={spectrum.size}"
        )
    normalized_bands = normalize_bad_bands(bad_bands)
    fit_mask = (wavenumbers >= baseline_fit_min) & (wavenumbers <= baseline_fit_max)
    fitted_wavenumbers = wavenumbers[fit_mask]
    fitted_spectrum = spectrum[fit_mask]
    valid_fit_mask = build_valid_mask(fitted_wavenumbers, normalized_bands)
    if fitted_wavenumbers.size < 10:
        return (
            fitted_wavenumbers,
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    baseline = estimate_baseline(
        smooth_baseline_input(fitted_spectrum),
        method=baseline_method,
        lam=baseline_lam,
        p=baseline_asls_p,
        niter=baseline_max_iter,
        valid_mask=valid_fit_mask,
    )
    return fitted_wavenumbers, fitted_spectrum - baseline, baseline


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
):
    """完成单谱基线校正、裁剪、坏段删除和插值。"""
    normalized_bands = normalize_bad_bands(bad_bands)
    fit_min = cut_min if baseline_fit_min is None else float(baseline_fit_min)
    fit_max = cut_max if baseline_fit_max is None else float(baseline_fit_max)
    fitted_wavenumbers, corrected, _baseline = fit_baseline_spectrum(
        wavenumbers,
        spectrum,
        baseline_fit_min=fit_min,
        baseline_fit_max=fit_max,
        bad_bands=normalized_bands,
        baseline_lam=baseline_lam,
        baseline_asls_p=baseline_asls_p,
        baseline_max_iter=baseline_max_iter,
        baseline_method=baseline_method,
    )
    if fitted_wavenumbers.size < 10:
        return None, None
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
        return None, None
    interpolated = interpolate_valid_bands(
        cropped_wavenumbers, cropped_spectrum, target_axis, normalized_bands,
    )
    return target_axis, interpolated
