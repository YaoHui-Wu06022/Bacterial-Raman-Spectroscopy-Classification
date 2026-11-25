"""小球谱参考峰拟合与日期级仿射参数计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


PRIMARY_PEAKS = (620.9, 1001.4, 1602.3)
AUXILIARY_PEAKS = (1031.8, 1583.1)
PRIMARY_STD = (0.69, 0.54, 0.73)
MIN_SNR = 8.0
# 1583.1 与相邻主峰重叠，单独放宽该辅助峰的信噪比门槛。
OVERLAP_MIN_SNR = 6.5
INITIAL_RADIUS = 45.0
REFINED_RADIUS = 18.0
ISOLATED_PRIMARY_RADIUS = 12.0
AUXILIARY_RADIUS = 10.0
OVERLAP_RADIUS = 17.0


@dataclass(frozen=True)
class PeakFit:
    """保存一个参考峰的拟合中心和质量状态。"""

    target: float
    center: float | None
    rmse: float | None
    snr: float | None
    status: str


def evaluate_pseudo_voigt(
    x: np.ndarray,
    baseline: float,
    slope: float,
    amplitude: float,
    center: float,
    fwhm: float,
    lorentz_fraction: float,
) -> np.ndarray:
    """计算带一次线性基线的 pseudo-Voigt 峰形。"""
    gaussian = np.exp(-4.0 * np.log(2.0) * np.square((x - center) / fwhm))
    lorentz = 1.0 / (1.0 + 4.0 * np.square((x - center) / fwhm))
    profile = (1.0 - lorentz_fraction) * gaussian + lorentz_fraction * lorentz
    return baseline + slope * (x - center) + amplitude * profile


def evaluate_double_pseudo_voigt(
    x: np.ndarray,
    baseline: float,
    slope: float,
    first_amplitude: float,
    first_center: float,
    first_fwhm: float,
    first_lorentz_fraction: float,
    second_amplitude: float,
    second_center: float,
    second_fwhm: float,
    second_lorentz_fraction: float,
) -> np.ndarray:
    """计算共享线性基线的双 pseudo-Voigt 峰形。"""
    first = evaluate_pseudo_voigt(
        x,
        baseline,
        slope,
        first_amplitude,
        first_center,
        first_fwhm,
        first_lorentz_fraction,
    )
    second = evaluate_pseudo_voigt(
        x,
        0.0,
        0.0,
        second_amplitude,
        second_center,
        second_fwhm,
        second_lorentz_fraction,
    )
    return first + second


def _finite_sorted_arrays(wavenumbers: np.ndarray, intensities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(wavenumbers) & np.isfinite(intensities)
    x = np.asarray(wavenumbers[mask], dtype=np.float64)
    y = np.asarray(intensities[mask], dtype=np.float64)
    order = np.argsort(x)
    return x[order], y[order]


def _is_saturated_peak(intensities: np.ndarray) -> bool:
    if intensities.size < 3:
        return False
    maximum = float(np.max(intensities))
    at_peak = np.isclose(intensities, maximum, rtol=0.0, atol=max(abs(maximum) * 1e-10, 1e-12))
    longest_run = 0
    current_run = 0
    for value in at_peak:
        current_run = current_run + 1 if value else 0
        longest_run = max(longest_run, current_run)
    return longest_run >= 3


def _fit_pseudo_voigt_peak(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    target: float,
    center_hint: float,
    radius: float,
) -> PeakFit:
    x_all, y_all = _finite_sorted_arrays(wavenumbers, intensities)
    mask = np.abs(x_all - center_hint) <= radius
    x, y = x_all[mask], y_all[mask]
    if x.size < 9:
        return PeakFit(target, None, None, None, "window_too_small")
    if _is_saturated_peak(y):
        return PeakFit(target, None, None, None, "saturated")

    edge_count = max(2, x.size // 6)
    edge_x = np.concatenate((x[:edge_count], x[-edge_count:]))
    edge_y = np.concatenate((y[:edge_count], y[-edge_count:]))
    slope, baseline = np.polyfit(edge_x - center_hint, edge_y, 1)
    residual = y - (baseline + slope * (x - center_hint))
    center_initial = float(x[np.argmax(residual)])
    amplitude_initial = max(float(np.max(residual)), np.finfo(float).eps)
    value_range = max(float(np.ptp(y)), amplitude_initial, 1.0)
    step = max(float(np.median(np.diff(x))), np.finfo(float).eps)
    slope_limit = max(value_range / radius * 5.0, 1.0)
    lower = [float(np.min(y) - 2.0 * value_range), -slope_limit, 0.0, float(x.min()), step, 0.0]
    upper = [float(np.max(y) + 2.0 * value_range), slope_limit, 10.0 * value_range, float(x.max()), 2.0 * radius, 1.0]
    initial = [baseline, slope, amplitude_initial, center_initial, min(max(6.0, step * 2.0), radius), 0.5]
    try:
        parameters, _ = curve_fit(
            evaluate_pseudo_voigt,
            x,
            y,
            p0=initial,
            bounds=(lower, upper),
            maxfev=30000,
        )
    except (RuntimeError, ValueError):
        return PeakFit(target, None, None, None, "fit_failed")

    fit_residual = y - evaluate_pseudo_voigt(x, *parameters)
    noise = float(np.std(fit_residual, ddof=1)) if fit_residual.size > 1 else 0.0
    snr = float(parameters[2] / max(noise, np.finfo(float).eps))
    status = "ok" if snr >= MIN_SNR else "low_snr"
    return PeakFit(
        target,
        float(parameters[3]),
        float(np.sqrt(np.mean(np.square(fit_residual)))),
        snr,
        status,
    )


def _fit_overlapping_auxiliary_peak(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    auxiliary_center_hint: float,
    primary_center_hint: float,
) -> PeakFit:
    """同时拟合 1583.1 和 1602.3，避免相邻主峰干扰。"""
    target = AUXILIARY_PEAKS[1]
    x_all, y_all = _finite_sorted_arrays(wavenumbers, intensities)
    lower_edge = auxiliary_center_hint - OVERLAP_RADIUS
    upper_edge = primary_center_hint + OVERLAP_RADIUS
    mask = (x_all >= lower_edge) & (x_all <= upper_edge)
    x, y = x_all[mask], y_all[mask]
    if x.size < 12:
        return PeakFit(target, None, None, None, "window_too_small")
    if _is_saturated_peak(y):
        return PeakFit(target, None, None, None, "saturated")

    edge_count = max(2, x.size // 6)
    edge_x = np.concatenate((x[:edge_count], x[-edge_count:]))
    edge_y = np.concatenate((y[:edge_count], y[-edge_count:]))
    slope, baseline = np.polyfit(edge_x - auxiliary_center_hint, edge_y, 1)
    residual = y - (baseline + slope * (x - auxiliary_center_hint))
    amplitude = max(float(np.max(residual)), np.finfo(float).eps)
    value_range = max(float(np.ptp(y)), amplitude, 1.0)
    step = max(float(np.median(np.diff(x))), np.finfo(float).eps)
    slope_limit = max(value_range / max(upper_edge - lower_edge, step) * 5.0, 1.0)
    initial = [
        baseline,
        slope,
        amplitude * 0.25,
        auxiliary_center_hint,
        max(6.0, step * 2.0),
        0.5,
        amplitude,
        primary_center_hint,
        max(6.0, step * 2.0),
        0.5,
    ]
    lower = [
        float(np.min(y) - 2.0 * value_range),
        -slope_limit,
        0.0,
        auxiliary_center_hint - AUXILIARY_RADIUS,
        step,
        0.0,
        0.0,
        primary_center_hint - AUXILIARY_RADIUS,
        step,
        0.0,
    ]
    upper = [
        float(np.max(y) + 2.0 * value_range),
        slope_limit,
        10.0 * value_range,
        auxiliary_center_hint + AUXILIARY_RADIUS,
        2.0 * OVERLAP_RADIUS,
        1.0,
        10.0 * value_range,
        primary_center_hint + AUXILIARY_RADIUS,
        2.0 * OVERLAP_RADIUS,
        1.0,
    ]
    try:
        parameters, _ = curve_fit(
            evaluate_double_pseudo_voigt,
            x,
            y,
            p0=initial,
            bounds=(lower, upper),
            maxfev=50000,
        )
    except (RuntimeError, ValueError):
        return PeakFit(target, None, None, None, "fit_failed")

    fit_residual = y - evaluate_double_pseudo_voigt(x, *parameters)
    noise = float(np.std(fit_residual, ddof=1)) if fit_residual.size > 1 else 0.0
    snr = float(parameters[2] / max(noise, np.finfo(float).eps))
    status = "ok" if snr >= OVERLAP_MIN_SNR else "low_snr"
    return PeakFit(
        target,
        float(parameters[3]),
        float(np.sqrt(np.mean(np.square(fit_residual)))),
        snr,
        status,
    )


def _fit_primary(wavenumbers: np.ndarray, intensities: np.ndarray, hints: dict[float, float], radius: float) -> dict[float, PeakFit]:
    return {
        target: _fit_pseudo_voigt_peak(wavenumbers, intensities, target, hints[target], radius)
        for target in PRIMARY_PEAKS
    }


def has_valid_primary(fits: dict[float, PeakFit]) -> bool:
    return all(fits[target].status == "ok" for target in PRIMARY_PEAKS)


def fit_date_affine(primary_fits: list[dict[float, PeakFit]]) -> tuple[float, float, dict[float, float]] | None:
    valid = [fits for fits in primary_fits if has_valid_primary(fits)]
    if not valid:
        return None
    raw_centers = np.asarray(
        [np.median([fits[target].center for fits in valid]) for target in PRIMARY_PEAKS],
        dtype=np.float64,
    )
    weights = 1.0 / np.asarray(PRIMARY_STD, dtype=np.float64)
    scale, offset = np.polyfit(raw_centers, PRIMARY_PEAKS, 1, w=weights)
    residuals = {
        target: float(scale * center + offset - target)
        for target, center in zip(PRIMARY_PEAKS, raw_centers)
    }
    return float(scale), float(offset), residuals


def fit_auxiliary_peaks(
    primary: dict[float, PeakFit],
    scale: float,
    offset: float,
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
) -> dict[float, PeakFit]:
    corrected_wavenumbers = scale * wavenumbers + offset
    corrected_primary = {
        target: scale * primary[target].center + offset
        for target in PRIMARY_PEAKS
        if primary[target].center is not None
    }
    first_hint = AUXILIARY_PEAKS[0]
    second_hint = AUXILIARY_PEAKS[1]
    primary_hint = PRIMARY_PEAKS[-1]
    if PRIMARY_PEAKS[1] in corrected_primary:
        first_hint += corrected_primary[PRIMARY_PEAKS[1]] - PRIMARY_PEAKS[1]
    if PRIMARY_PEAKS[-1] in corrected_primary:
        primary_hint = corrected_primary[PRIMARY_PEAKS[-1]]
        second_hint += primary_hint - PRIMARY_PEAKS[-1]
    fits = {
        AUXILIARY_PEAKS[0]: _fit_pseudo_voigt_peak(
            corrected_wavenumbers,
            intensities,
            AUXILIARY_PEAKS[0],
            first_hint,
            AUXILIARY_RADIUS,
        )
    }
    fits[AUXILIARY_PEAKS[1]] = _fit_overlapping_auxiliary_peak(
        corrected_wavenumbers,
        intensities,
        second_hint,
        primary_hint,
    )
    return fits


def fit_primary_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> dict[float, PeakFit]:
    """先定位主峰，再按粗校准结果缩小拟合窗口。"""
    initial = _fit_primary(
        wavenumbers, intensities, {target: target for target in PRIMARY_PEAKS}, INITIAL_RADIUS
    )
    if any(initial[target].center is None for target in PRIMARY_PEAKS):
        return initial
    raw = np.asarray([initial[target].center for target in PRIMARY_PEAKS], dtype=np.float64)
    scale, offset = np.polyfit(raw, PRIMARY_PEAKS, 1)
    return {
        target: _fit_pseudo_voigt_peak(
            wavenumbers, intensities, target, (target - offset) / scale,
            ISOLATED_PRIMARY_RADIUS if target == PRIMARY_PEAKS[-1] else REFINED_RADIUS,
        )
        for target in PRIMARY_PEAKS
    }
