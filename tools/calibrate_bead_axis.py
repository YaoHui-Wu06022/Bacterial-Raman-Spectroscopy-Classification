"""利用聚苯乙烯小球谱拟合日期级波数轴仿射校准参数。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import font_manager


PRIMARY_PEAKS = (620.9, 1001.4, 1602.3)
PRIMARY_STD = (0.69, 0.54, 0.73)
AUXILIARY_PEAKS = (1031.8, 1583.1)
AUXILIARY_STD = (0.43, 0.86)
PRIMARY_LIMITS = tuple(3.0 * value for value in PRIMARY_STD)
AUXILIARY_LIMITS = tuple(3.0 * value for value in AUXILIARY_STD)
INITIAL_RADIUS = 45.0
REFINED_RADIUS = 18.0
AUXILIARY_RADIUS = 10.0
OVERLAP_RADIUS = 17.0
MIN_SNR = 8.0
RECOMMENDED_SNR = 100.0
DISPLAY_MIN = 400.0
DISPLAY_MAX = 1800.0
BEAD_DIR_NAME = "".join(map(chr, (23567, 29699)))


@dataclass(frozen=True)
class PeakFit:
    """保存单条谱中一个参考峰的 pseudo-Voigt 拟合结果。"""

    target: float
    center: float | None
    rmse: float | None
    snr: float | None
    saturated_enable: bool
    status: str


@dataclass(frozen=True)
class SpectrumResult:
    """保存一条小球谱的主峰拟合结果。"""

    path: Path
    fits: dict[float, PeakFit]


def resolve_project_dir() -> Path:
    """根据脚本位置定位项目根目录。"""
    project_dir = Path(__file__).resolve().parents[1]
    if (project_dir / "ramanv2").is_dir() and (project_dir / "dataset").is_dir():
        return project_dir
    raise FileNotFoundError("未找到同时包含 ramanv2 和 dataset 的项目根目录。")


PROJECT_DIR = resolve_project_dir()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ramanv2.common.arc_data import read_arc_data


def configure_chinese_font() -> None:
    """设置中文字体，保证校准图中的文字可见。"""
    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
    font_names = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei"):
        if font_name in font_names:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def resolve_date_dir(date_dir_value: str) -> Path:
    """解析待校准的日期目录。"""
    date_dir = Path(date_dir_value)
    if not date_dir.is_absolute():
        date_dir = PROJECT_DIR / date_dir
    if not date_dir.is_dir():
        raise NotADirectoryError(f"日期目录不存在：{date_dir}")
    return date_dir


def parse_natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """将文本拆为数字与文本片段，以自然数顺序排列谱文件。"""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


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
    """计算共享一次线性基线的双 pseudo-Voigt 重叠峰形。"""
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


def is_saturated_peak(intensities: np.ndarray) -> bool:
    """检查局部峰顶是否存在至少三个连续相同的最高强度值。"""
    if intensities.size < 3:
        return False
    maximum = float(np.max(intensities))
    tolerance = max(abs(maximum) * 1e-10, 1e-12)
    at_peak = np.isclose(intensities, maximum, rtol=0.0, atol=tolerance)
    longest_run = 0
    current_run = 0
    for value in at_peak:
        current_run = current_run + 1 if value else 0
        longest_run = max(longest_run, current_run)
    return longest_run >= 3


def build_display_mask(wavenumbers: np.ndarray) -> np.ndarray:
    """构建绘图波数范围掩码，使范围外强度不影响图像缩放。"""
    return (wavenumbers >= DISPLAY_MIN) & (wavenumbers <= DISPLAY_MAX)


def fit_pseudo_voigt_peak(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    target: float,
    center_hint: float,
    radius: float,
) -> PeakFit:
    """在局部窗口内以线性基线和 pseudo-Voigt 峰形拟合峰中心。"""
    mask = np.abs(wavenumbers - center_hint) <= radius
    x = wavenumbers[mask]
    y = intensities[mask]
    if x.size < 9:
        return PeakFit(target, None, None, None, False, "window_too_small")
    if is_saturated_peak(y):
        return PeakFit(target, None, None, None, True, "saturated")

    edge_count = max(2, x.size // 6)
    edge_x = np.concatenate((x[:edge_count], x[-edge_count:]))
    edge_y = np.concatenate((y[:edge_count], y[-edge_count:]))
    slope, baseline = np.polyfit(edge_x - center_hint, edge_y, 1)
    residual = y - (baseline + slope * (x - center_hint))
    center_initial = float(x[np.argmax(residual)])
    amplitude_initial = max(float(np.max(residual)), np.finfo(float).eps)
    value_range = max(float(np.ptp(y)), amplitude_initial, 1.0)
    step = float(np.median(np.diff(x)))
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
        return PeakFit(target, None, None, None, False, "fit_failed")

    fit_residual = y - evaluate_pseudo_voigt(x, *parameters)
    noise = float(np.std(fit_residual, ddof=1)) if fit_residual.size > 1 else 0.0
    snr = float(parameters[2] / max(noise, np.finfo(float).eps))
    status = "ok" if snr > MIN_SNR else "low_snr"
    return PeakFit(
        target,
        float(parameters[3]),
        float(np.sqrt(np.mean(np.square(fit_residual)))),
        snr,
        False,
        status,
    )


def fit_overlapping_auxiliary_peak(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    auxiliary_center_hint: float,
    primary_center_hint: float,
) -> PeakFit:
    """将 1583.1 与 1602.3 同时拟合，消除相邻主峰对辅助峰中心的干扰。"""
    auxiliary_target = AUXILIARY_PEAKS[1]
    lower_edge = auxiliary_center_hint - OVERLAP_RADIUS
    upper_edge = primary_center_hint + OVERLAP_RADIUS
    mask = (wavenumbers >= lower_edge) & (wavenumbers <= upper_edge)
    x = wavenumbers[mask]
    y = intensities[mask]
    if x.size < 12:
        return PeakFit(auxiliary_target, None, None, None, False, "window_too_small")
    if is_saturated_peak(y):
        return PeakFit(auxiliary_target, None, None, None, True, "saturated")

    edge_count = max(2, x.size // 6)
    edge_x = np.concatenate((x[:edge_count], x[-edge_count:]))
    edge_y = np.concatenate((y[:edge_count], y[-edge_count:]))
    slope, baseline = np.polyfit(edge_x - auxiliary_target, edge_y, 1)
    residual = y - (baseline + slope * (x - auxiliary_target))
    amplitude = max(float(np.max(residual)), np.finfo(float).eps)
    value_range = max(float(np.ptp(y)), amplitude, 1.0)
    step = float(np.median(np.diff(x)))
    slope_limit = max(value_range / (upper_edge - lower_edge) * 5.0, 1.0)
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
        return PeakFit(auxiliary_target, None, None, None, False, "fit_failed")

    fit_residual = y - evaluate_double_pseudo_voigt(x, *parameters)
    noise = float(np.std(fit_residual, ddof=1)) if fit_residual.size > 1 else 0.0
    snr = float(parameters[2] / max(noise, np.finfo(float).eps))
    status = "ok" if snr > MIN_SNR else "low_snr"
    return PeakFit(
        auxiliary_target,
        float(parameters[3]),
        float(np.sqrt(np.mean(np.square(fit_residual)))),
        snr,
        False,
        status,
    )


def build_peak_fits(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    targets: tuple[float, ...],
    center_hints: dict[float, float],
    radius: float,
) -> dict[float, PeakFit]:
    """在给定波数轴上拟合多个指定参考峰。"""
    if wavenumbers.size < 9:
        return {
            target: PeakFit(target, None, None, None, False, "unscorable")
            for target in targets
        }
    return {
        target: fit_pseudo_voigt_peak(
            wavenumbers,
            intensities,
            target,
            center_hints[target],
            radius,
        )
        for target in targets
    }


def build_affine_from_centers(
    centers: dict[float, float],
) -> tuple[float, float, dict[float, float]] | None:
    """由三个主峰中心拟合原始波数到参考波数的仿射关系。"""
    if any(target not in centers for target in PRIMARY_PEAKS):
        return None
    raw_centers = np.asarray([centers[target] for target in PRIMARY_PEAKS], dtype=float)
    # ASTM 标准差越小，参考峰位在日期级仿射拟合中的权重越高。
    weights = 1.0 / np.asarray(PRIMARY_STD, dtype=float)
    scale, offset = np.polyfit(raw_centers, PRIMARY_PEAKS, 1, w=weights)
    residuals = {
        target: float(scale * center + offset - target)
        for target, center in centers.items()
    }
    return float(scale), float(offset), residuals


def collect_fitted_primary_results(results: list[SpectrumResult]) -> list[SpectrumResult]:
    """保留三个主峰均得到峰中心的谱，用于缩小后续拟合窗口。"""
    return [
        result
        for result in results
        if all(result.fits[target].center is not None for target in PRIMARY_PEAKS)
    ]


def collect_valid_primary_results(results: list[SpectrumResult]) -> list[SpectrumResult]:
    """保留三个主峰均未饱和且 SNR 合格的小球谱。"""
    return [
        result
        for result in results
        if all(result.fits[target].status == "ok" for target in PRIMARY_PEAKS)
    ]


def build_median_centers(results: list[SpectrumResult]) -> dict[float, float]:
    """计算各主峰在有效重复小球谱中的峰中心中位数。"""
    return {
        target: float(np.median([result.fits[target].center for result in results]))
        for target in PRIMARY_PEAKS
    }


def build_refined_primary_results(results: list[SpectrumResult]) -> list[SpectrumResult]:
    """按每条谱自身的粗仿射关系缩小窗口后重新拟合三个主峰。"""
    refined_results = []
    for result in results:
        centers = {target: result.fits[target].center for target in PRIMARY_PEAKS}
        affine = build_affine_from_centers(centers)
        if affine is None:
            continue
        scale, offset, _ = affine
        center_hints = {target: (target - offset) / scale for target in PRIMARY_PEAKS}
        wavenumbers, intensities = read_arc_data(result.path)
        fits = build_peak_fits(
            wavenumbers,
            intensities,
            PRIMARY_PEAKS,
            center_hints,
            REFINED_RADIUS,
        )
        refined_results.append(SpectrumResult(result.path, fits))
    return refined_results


def build_auxiliary_fits(
    results: list[SpectrumResult],
    scale: float,
    offset: float,
) -> dict[Path, dict[float, PeakFit]]:
    """在日期级校准后的波数轴上独立拟合两个辅助验证峰。"""
    auxiliary_fits = {}
    for result in results:
        wavenumbers, intensities = read_arc_data(result.path)
        corrected_wavenumbers = scale * wavenumbers + offset
        corrected_primary_centers = {
            target: scale * result.fits[target].center + offset
            for target in PRIMARY_PEAKS
            if result.fits[target].center is not None
        }
        first_center_hint = AUXILIARY_PEAKS[0]
        second_center_hint = AUXILIARY_PEAKS[1]
        primary_center_hint = PRIMARY_PEAKS[-1]
        if PRIMARY_PEAKS[1] in corrected_primary_centers:
            first_center_hint += corrected_primary_centers[PRIMARY_PEAKS[1]] - PRIMARY_PEAKS[1]
        if PRIMARY_PEAKS[-1] in corrected_primary_centers:
            primary_center_hint = corrected_primary_centers[PRIMARY_PEAKS[-1]]
            second_center_hint += primary_center_hint - PRIMARY_PEAKS[-1]
        fits = build_peak_fits(
            corrected_wavenumbers,
            intensities,
            (AUXILIARY_PEAKS[0],),
            {AUXILIARY_PEAKS[0]: first_center_hint},
            AUXILIARY_RADIUS,
        )
        fits[AUXILIARY_PEAKS[1]] = fit_overlapping_auxiliary_peak(
            corrected_wavenumbers,
            intensities,
            second_center_hint,
            primary_center_hint,
        )
        auxiliary_fits[result.path] = fits
    return auxiliary_fits


def build_date_result(
    primary_results: list[SpectrumResult],
    auxiliary_fits: dict[Path, dict[float, PeakFit]],
) -> tuple[str, float | None, float | None, dict[float, float], dict[float, float], int]:
    """汇总主峰参数及校准后辅助峰残差，给出日期级验收状态。"""
    valid_primary_results = collect_valid_primary_results(primary_results)
    if not valid_primary_results:
        return "needs_review", None, None, {}, {}, 0
    affine = build_affine_from_centers(build_median_centers(valid_primary_results))
    if affine is None:
        return "needs_review", None, None, {}, {}, 0
    scale, offset, primary_residuals = affine

    valid_results = [
        result
        for result in valid_primary_results
        if all(
            auxiliary_fits.get(result.path, {}).get(target, PeakFit(target, None, None, None, False, "missing")).status == "ok"
            for target in AUXILIARY_PEAKS
        )
    ]
    if not valid_results:
        return "needs_review", None, None, primary_residuals, {}, 0
    auxiliary_residuals = {
        target: float(
            np.median([
                auxiliary_fits[result.path][target].center
                for result in valid_results
            ])
            - target
        )
        for target in AUXILIARY_PEAKS
    }
    primary_pass_enable = all(
        abs(primary_residuals[target]) <= limit
        for target, limit in zip(PRIMARY_PEAKS, PRIMARY_LIMITS)
    )
    auxiliary_pass_enable = all(
        abs(auxiliary_residuals[target]) <= limit
        for target, limit in zip(AUXILIARY_PEAKS, AUXILIARY_LIMITS)
    )
    if not primary_pass_enable or not auxiliary_pass_enable:
        return "needs_review", None, None, primary_residuals, auxiliary_residuals, len(valid_results)
    if len(valid_results) < 3:
        return "limited_replicates", scale, offset, primary_residuals, auxiliary_residuals, len(valid_results)
    return "ready", scale, offset, primary_residuals, auxiliary_residuals, len(valid_results)


def save_calibration_figure(
    result: SpectrumResult,
    scale: float | None,
    offset: float | None,
    status: str,
    auxiliary_fits: dict[float, PeakFit],
    output_path: Path,
) -> None:
    """保存一条小球谱校准前后全谱对比，并标注主峰与辅助峰位置。"""
    wavenumbers, intensities = read_arc_data(result.path)
    display_mask = build_display_mask(wavenumbers)
    display_wavenumbers = wavenumbers[display_mask]
    display_intensities = intensities[display_mask]
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=140, sharey=True)
    raw_axis, corrected_axis = axes
    raw_axis.plot(display_wavenumbers, display_intensities, color="#4C72B0", linewidth=0.9)
    raw_axis.set_title(f"校准前：{result.path.name}")
    raw_axis.set_ylabel("Intensity")
    for index, target in enumerate(PRIMARY_PEAKS, start=1):
        peak_fit = result.fits[target]
        if peak_fit.center is not None:
            raw_axis.axvline(peak_fit.center, color="#C44E52", linestyle="--", linewidth=0.9)
            raw_axis.text(peak_fit.center, raw_axis.get_ylim()[1], f"P{index}={peak_fit.center:.2f}", rotation=90, va="top", fontsize=8)

    if scale is not None and offset is not None:
        corrected_axis.plot(
            scale * display_wavenumbers + offset,
            display_intensities,
            color="#4C72B0",
            linewidth=0.9,
        )
        corrected_axis.set_title(f"校准后预览（{status}）：scale={scale:.8f}，offset={offset:.4f}")
        for index, target in enumerate(PRIMARY_PEAKS, start=1):
            corrected_axis.axvline(target, color="#C44E52", linestyle="--", linewidth=0.9)
            corrected_axis.text(target, corrected_axis.get_ylim()[1], f"P{index}={target:.1f}", rotation=90, va="top", fontsize=8)
        for index, target in enumerate(AUXILIARY_PEAKS, start=1):
            fit = auxiliary_fits.get(target)
            if fit is not None and fit.center is not None:
                corrected_axis.axvline(fit.center, color="#DD8452", linestyle=":", linewidth=1.2)
                corrected_axis.text(fit.center, corrected_axis.get_ylim()[0], f"A{index}={fit.center:.2f}", rotation=90, va="bottom", fontsize=8)
            corrected_axis.axvline(target, color="#8172B3", linestyle="-.", linewidth=0.8)
    else:
        corrected_axis.text(0.5, 0.5, "日期级参数未通过验收", ha="center", va="center", transform=corrected_axis.transAxes)

    for axis in axes:
        axis.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.grid(alpha=0.2)
    figure.suptitle("主峰与辅助峰均采用 pseudo-Voigt 拟合", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_summary_csv(
    output_path: Path,
    results: list[SpectrumResult],
    auxiliary_fits: dict[Path, dict[float, PeakFit]],
) -> None:
    """写出每条小球谱的主峰、辅助峰中心、SNR 与饱和检查结果。"""
    fields = ["file"]
    for prefix, targets in (("primary", PRIMARY_PEAKS), ("auxiliary", AUXILIARY_PEAKS)):
        for index in range(1, len(targets) + 1):
            fields.extend((f"{prefix}_{index}_center", f"{prefix}_{index}_rmse", f"{prefix}_{index}_snr", f"{prefix}_{index}_status"))
            if prefix == "auxiliary":
                fields.append(f"{prefix}_{index}_residual")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {"file": result.path.name}
            for prefix, targets, fits in (
                ("primary", PRIMARY_PEAKS, result.fits),
                ("auxiliary", AUXILIARY_PEAKS, auxiliary_fits.get(result.path, {})),
            ):
                for index, target in enumerate(targets, start=1):
                    peak_fit = fits.get(target)
                    if peak_fit is not None:
                        row.update(
                            {
                                f"{prefix}_{index}_center": peak_fit.center,
                                f"{prefix}_{index}_rmse": peak_fit.rmse,
                                f"{prefix}_{index}_snr": peak_fit.snr,
                                f"{prefix}_{index}_status": peak_fit.status,
                            }
                        )
                        if prefix == "auxiliary" and peak_fit.center is not None:
                            row[f"{prefix}_{index}_residual"] = peak_fit.center - target
            writer.writerow(row)


def write_affine_json(
    output_path: Path,
    date_dir: Path,
    status: str,
    scale: float | None,
    offset: float | None,
    diagnostic_scale: float,
    diagnostic_offset: float,
    manual_approval_enable: bool,
    primary_residuals: dict[float, float],
    auxiliary_residuals: dict[float, float],
    repeat_count: int,
) -> None:
    """写出后续数据生成读取的日期级参数、质量状态与人工确认开关。"""
    payload = {
        "status": status,
        "manual_approval_enable": manual_approval_enable,
        "date": date_dir.name,
        "formula": "corrected_wavenumber = scale * raw_wavenumber + offset",
        "scale": scale,
        "offset": offset,
        "diagnostic_scale": diagnostic_scale,
        "diagnostic_offset": diagnostic_offset,
        "valid_repeat_count": repeat_count,
        "minimum_recommended_repeat_count": 3,
        "peak_shape": "pseudo_voigt",
        "primary_peaks_cm-1": list(PRIMARY_PEAKS),
        "primary_limits_cm-1": list(PRIMARY_LIMITS),
        "primary_residuals_cm-1": {str(target): primary_residuals.get(target) for target in PRIMARY_PEAKS},
        "auxiliary_peaks_cm-1": list(AUXILIARY_PEAKS),
        "auxiliary_limits_cm-1": list(AUXILIARY_LIMITS),
        "auxiliary_residuals_cm-1": {str(target): auxiliary_residuals.get(target) for target in AUXILIARY_PEAKS},
        "minimum_snr": MIN_SNR,
        "recommended_snr": RECOMMENDED_SNR,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_sample_spectrum_paths(date_dir: Path) -> list[Path]:
    """从每个普通样本文件夹按自然顺序各选中间一条谱用于仿射预览。"""
    sample_paths = []
    sample_dirs = sorted(
        (
            path
            for path in date_dir.iterdir()
            if path.is_dir() and path.name not in (BEAD_DIR_NAME, "shift")
        ),
        key=lambda path: parse_natural_sort_key(path.name),
    )
    for sample_dir in sample_dirs:
        spectrum_paths = sorted(
            sample_dir.rglob("*.arc_data"),
            key=lambda path: parse_natural_sort_key(str(path.relative_to(sample_dir))),
        )
        if spectrum_paths:
            sample_paths.append(spectrum_paths[len(spectrum_paths) // 2])
    if not sample_paths:
        raise FileNotFoundError(f"未找到可用于仿射预览的样本谱：{date_dir}")
    return sample_paths


def save_sample_affine_preview(
    date_dir: Path,
    sample_path: Path,
    scale: float,
    offset: float,
    output_path: Path,
) -> None:
    """保存一条普通样本在诊断性仿射变换前后的波数轴对比图。"""
    wavenumbers, intensities = read_arc_data(sample_path)
    display_mask = build_display_mask(wavenumbers)
    display_wavenumbers = wavenumbers[display_mask]
    display_intensities = intensities[display_mask]
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharey=True)
    raw_axis, corrected_axis = axes
    raw_axis.plot(display_wavenumbers, display_intensities, color="#4C72B0", linewidth=0.8)
    raw_axis.set_title(f"原始谱：{sample_path.relative_to(date_dir)}")
    corrected_axis.plot(
        scale * display_wavenumbers + offset,
        display_intensities,
        color="#55A868",
        linewidth=0.8,
    )
    corrected_axis.set_title(f"诊断性仿射预览：scale={scale:.8f}，offset={offset:.4f}")
    for axis in axes:
        axis.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.set_ylabel("Intensity")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def build_temp_shift_dir(date_dir: Path) -> Path:
    """创建校准临时目录，在完成后再发布为 shift 目录。"""
    temp_dir = date_dir / f".shift_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def publish_shift_dir(temp_dir: Path) -> Path:
    """将完整的新校准产物覆盖发布为 shift 目录。"""
    shift_dir = temp_dir.parent / "shift"
    if shift_dir.exists():
        shutil.rmtree(shift_dir)
    temp_dir.replace(shift_dir)
    return shift_dir


def run_calibration(date_dir: Path, manual_approval_enable: bool = False) -> Path:
    """执行一个日期的小球峰形拟合、仿射校准、独立验证与结果写出。"""
    bead_dir = date_dir / BEAD_DIR_NAME
    spectrum_paths = sorted(bead_dir.glob("*.arc_data"), key=lambda path: parse_natural_sort_key(path.name))
    if not spectrum_paths:
        raise FileNotFoundError(f"未找到小球谱：{bead_dir}")

    initial_results = []
    for path in spectrum_paths:
        wavenumbers, intensities = read_arc_data(path)
        fits = build_peak_fits(
            wavenumbers,
            intensities,
            PRIMARY_PEAKS,
            {target: target for target in PRIMARY_PEAKS},
            INITIAL_RADIUS,
        )
        initial_results.append(SpectrumResult(path, fits))
    fitted_initial_results = collect_fitted_primary_results(initial_results)
    if not fitted_initial_results:
        raise RuntimeError("没有可用于粗仿射拟合的小球谱。")

    primary_results = build_refined_primary_results(fitted_initial_results)
    fitted_primary_results = collect_fitted_primary_results(primary_results)
    if not fitted_primary_results:
        raise RuntimeError("缩小窗口后没有可用于日期级仿射拟合的小球谱。")
    preview_affine = build_affine_from_centers(build_median_centers(fitted_primary_results))
    if preview_affine is None:
        raise RuntimeError("无法由精细主峰中心拟合仿射参数。")
    preview_scale, preview_offset, _ = preview_affine

    auxiliary_fits = build_auxiliary_fits(primary_results, preview_scale, preview_offset)
    status, scale, offset, primary_residuals, auxiliary_residuals, repeat_count = build_date_result(
        primary_results,
        auxiliary_fits,
    )
    if manual_approval_enable:
        # 人工确认后发布诊断性参数，同时保留自动验收的全部残差记录。
        status = "manual_approved"
        scale = preview_scale
        offset = preview_offset
    plot_scale = scale if scale is not None else preview_scale
    plot_offset = offset if offset is not None else preview_offset

    temp_dir = build_temp_shift_dir(date_dir)
    for result in primary_results:
        save_calibration_figure(
            result,
            plot_scale,
            plot_offset,
            status,
            auxiliary_fits.get(result.path, {}),
            temp_dir / f"{result.path.stem}_calibration.png",
        )
    write_summary_csv(temp_dir / "calibration_summary.csv", primary_results, auxiliary_fits)
    preview_dir = temp_dir / "sample_affine_previews"
    preview_dir.mkdir()
    for sample_path in collect_sample_spectrum_paths(date_dir):
        sample_dir_name = sample_path.relative_to(date_dir).parts[0]
        save_sample_affine_preview(
            date_dir,
            sample_path,
            plot_scale,
            plot_offset,
            preview_dir / f"{sample_dir_name}_affine_preview.png",
        )
    write_affine_json(
        temp_dir / "affine.json",
        date_dir,
        status,
        scale,
        offset,
        preview_scale,
        preview_offset,
        manual_approval_enable,
        primary_residuals,
        auxiliary_residuals,
        repeat_count,
    )
    return publish_shift_dir(temp_dir)


def run_all_calibrations(
    data_root: Path,
    manual_approval_enable: bool = False,
) -> list[Path]:
    """逐个处理数据根目录下的日期目录，并保留每个日期的独立结果。"""
    date_dirs = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: path.name,
    )
    result_dirs = []
    failures = []
    for date_dir in date_dirs:
        try:
            result_dirs.append(run_calibration(date_dir, manual_approval_enable))
            print(f"[ok] {date_dir.name}")
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            failures.append(f"{date_dir.name}: {error}")
            print(f"[failed] {date_dir.name}: {error}")
    if failures:
        raise RuntimeError("批量仿射模拟存在失败日期：\n" + "\n".join(failures))
    return result_dirs


def parse_arguments() -> argparse.Namespace:
    """读取待校准日期目录的命令行参数。"""
    parser = argparse.ArgumentParser(description="利用小球谱拟合日期级波数轴仿射参数")
    parser.add_argument(
        "--date-dir",
        default="dataset/alldata/cosmic_data/20231130",
        help="包含小球子目录的日期目录",
    )
    parser.add_argument(
        "--data-root",
        help="批量处理时包含多个日期目录的数据根目录",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="处理数据根目录下的全部日期目录",
    )
    parser.add_argument(
        "--manual-approval",
        action="store_true",
        help="将诊断性仿射参数作为人工确认后的正式参数发布",
    )
    return parser.parse_args()


def run_program() -> None:
    """配置绘图字体并执行日期校准。"""
    configure_chinese_font()
    arguments = parse_arguments()
    if arguments.all_dates:
        if arguments.data_root is None:
            raise ValueError("批量处理时需要提供 --data-root。")
        for result_dir in run_all_calibrations(
            resolve_date_dir(arguments.data_root),
            arguments.manual_approval,
        ):
            print(result_dir)
        return
    print(run_calibration(resolve_date_dir(arguments.date_dir), arguments.manual_approval))


if __name__ == "__main__":
    run_program()
