"""按日期利用小球谱计算并应用波数轴仿射校正。"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.common.naming import build_natural_key
from ramanv2.common.plotting import configure_matplotlib_fonts
from ramanv2.common.publishing import create_temporary_path, publish_file, rename_directory
from ramanv2.pipeline.lineage import (
    initialize_folder_lineage,
    load_folder_lineage,
    record_affine_approval,
    record_folder_compensation,
    record_output_path,
    save_folder_lineage,
)
from ramanv2.core.paths import PROJECT_ROOT


configure_matplotlib_fonts()


PRIMARY_PEAKS = (620.9, 1001.4, 1602.3)
AUXILIARY_PEAKS = (1031.8, 1583.1)
PRIMARY_STD = (0.69, 0.54, 0.73)
RESIDUAL_LIMIT_CM = 2.0
MIN_SNR = 8.0
INITIAL_RADIUS = 45.0
REFINED_RADIUS = 18.0
ISOLATED_PRIMARY_RADIUS = 12.0
AUXILIARY_RADIUS = 10.0
OVERLAP_RADIUS = 17.0
DISPLAY_MIN = 400.0
DISPLAY_MAX = 1800.0
BEAD_DIR_NAME = "小球"
SHIFT_DIR_NAME = "shift"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "shift_data"


@dataclass(frozen=True)
class PeakFit:
    """保存一个参考峰的拟合中心和质量状态。"""

    target: float
    center: float | None
    rmse: float | None
    snr: float | None
    status: str


@dataclass(frozen=True)
class BeadResult:
    """保存一条小球谱的主峰和辅助峰拟合结果。"""

    path: Path
    primary: dict[float, PeakFit]
    auxiliary: dict[float, PeakFit]


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
    status = "ok" if snr >= MIN_SNR else "low_snr"
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


def _has_valid_primary(fits: dict[float, PeakFit]) -> bool:
    return all(fits[target].status == "ok" for target in PRIMARY_PEAKS)


def _fit_date_affine(results: list[BeadResult]) -> tuple[float, float, dict[float, float]] | None:
    valid = [result for result in results if _has_valid_primary(result.primary)]
    if not valid:
        return None
    raw_centers = np.asarray(
        [np.median([result.primary[target].center for result in valid]) for target in PRIMARY_PEAKS],
        dtype=np.float64,
    )
    weights = 1.0 / np.asarray(PRIMARY_STD, dtype=np.float64)
    scale, offset = np.polyfit(raw_centers, PRIMARY_PEAKS, 1, w=weights)
    residuals = {
        target: float(scale * center + offset - target)
        for target, center in zip(PRIMARY_PEAKS, raw_centers)
    }
    return float(scale), float(offset), residuals


def _fit_auxiliary(
    result: BeadResult,
    scale: float,
    offset: float,
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
) -> dict[float, PeakFit]:
    corrected_wavenumbers = scale * wavenumbers + offset
    corrected_primary = {
        target: scale * result.primary[target].center + offset
        for target in PRIMARY_PEAKS
        if result.primary[target].center is not None
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


def _build_results(bead_paths: list[Path]) -> list[BeadResult]:
    initial: list[BeadResult] = []
    for path in bead_paths:
        wavenumbers, intensities = read_arc_data(path)
        initial.append(
            BeadResult(
                path,
                _fit_primary(
                    wavenumbers,
                    intensities,
                    {target: target for target in PRIMARY_PEAKS},
                    INITIAL_RADIUS,
                ),
                {},
            )
        )

    refined: list[BeadResult] = []
    for result in initial:
        if not all(result.primary[target].center is not None for target in PRIMARY_PEAKS):
            refined.append(result)
            continue
        centers = {target: result.primary[target].center for target in PRIMARY_PEAKS}
        raw = np.asarray([centers[target] for target in PRIMARY_PEAKS], dtype=np.float64)
        scale, offset = np.polyfit(raw, PRIMARY_PEAKS, 1)
        wavenumbers, intensities = read_arc_data(result.path)
        refined_primary = {
            target: _fit_pseudo_voigt_peak(
                wavenumbers,
                intensities,
                target,
                (target - offset) / scale,
                ISOLATED_PRIMARY_RADIUS if target == PRIMARY_PEAKS[-1] else REFINED_RADIUS,
            )
            for target in PRIMARY_PEAKS
        }
        refined.append(
            BeadResult(
                result.path,
                refined_primary,
                {},
            )
        )
    return refined


def _result_payload(result: BeadResult) -> dict[str, object]:
    payload: dict[str, object] = {"file": result.path.name, "primary": {}, "auxiliary": {}}
    for group, fits in (("primary", result.primary), ("auxiliary", result.auxiliary)):
        payload[group] = {
            str(target): {
                "center": fit.center,
                "rmse": fit.rmse,
                "snr": fit.snr,
                "status": fit.status,
            }
            for target, fit in fits.items()
        }
    return payload


def _save_calibration_figure(
    result: BeadResult,
    scale: float | None,
    offset: float | None,
    status: str,
    output_path: Path,
) -> None:
    wavenumbers, intensities = read_arc_data(result.path)
    mask = (wavenumbers >= DISPLAY_MIN) & (wavenumbers <= DISPLAY_MAX)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=140, sharey=True)
    raw_axis, corrected_axis = axes
    raw_axis.plot(wavenumbers[mask], intensities[mask], color="#4C72B0", linewidth=0.9)
    raw_axis.set_title(f"校准前：{result.path.name}")
    if scale is not None and offset is not None:
        corrected_axis.plot(scale * wavenumbers[mask] + offset, intensities[mask], color="#4C72B0", linewidth=0.9)
        corrected_axis.set_title(f"校准后预览（{status}）：scale={scale:.8f}，offset={offset:.4f}")
        for target in PRIMARY_PEAKS:
            corrected_axis.axvline(target, color="#C44E52", linestyle="--", linewidth=0.9)
        for target in AUXILIARY_PEAKS:
            corrected_axis.axvline(target, color="#8172B3", linestyle="-.", linewidth=0.8)
    else:
        corrected_axis.text(0.5, 0.5, "日期级参数无法通过验收", ha="center", va="center", transform=corrected_axis.transAxes)
    for axis in axes:
        axis.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.set_ylabel("Intensity")
        axis.grid(alpha=0.2)
    figure.suptitle("主峰与辅助峰采用 pseudo-Voigt 拟合", fontsize=10)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _publish_shift(temp_shift: Path, output_date: Path) -> Path:
    had_output_date = output_date.exists()
    output_date.mkdir(parents=True, exist_ok=True)
    shift_path = output_date / SHIFT_DIR_NAME
    backup = output_date / f"shift_previous_{uuid4().hex[:8]}"
    if shift_path.exists():
        rename_directory(shift_path, backup)
    try:
        rename_directory(temp_shift, shift_path)
    except Exception:
        if backup.exists() and not shift_path.exists():
            rename_directory(backup, shift_path)
        if not had_output_date and output_date.exists() and not any(output_date.iterdir()):
            output_date.rmdir()
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return shift_path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _failure_payload(date_name: str, error: str) -> dict[str, object]:
    return {
        "status": "needs_review",
        "manual_approval_enable": False,
        "date": date_name,
        "formula": "corrected_wavenumber = scale * raw_wavenumber + offset",
        "scale": None,
        "offset": None,
        "diagnostic_scale": None,
        "diagnostic_offset": None,
        "valid_repeat_count": 0,
        "primary_valid_repeat_count": 0,
        "minimum_valid_repeat_count": 1,
        "fit_method": "pseudo_voigt",
        "primary_peaks_cm-1": list(PRIMARY_PEAKS),
        "auxiliary_peaks_cm-1": list(AUXILIARY_PEAKS),
        "residual_limit_cm-1": RESIDUAL_LIMIT_CM,
        "primary_residuals_cm-1": {},
        "auxiliary_residuals_cm-1": {},
        "primary_centers_cm-1": {},
        "auxiliary_centers_cm-1": {},
        "errors": [error],
        "bead_results": [],
    }


def analyze_date(date_dir: Path | str, output_root: Path | str = OUTPUT_ROOT) -> dict[str, object]:
    """分析一个日期并原子发布该日期的 shift 诊断目录。"""
    date_path = Path(date_dir).resolve()
    output_root_path = Path(output_root).resolve()
    if not date_path.is_dir():
        raise FileNotFoundError(f"日期目录不存在：{date_path}")
    output_date = output_root_path / date_path.name
    had_output_date = output_date.exists()
    temporary = output_date / f".shift_building_{uuid4().hex}"
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        bead_dir = date_path / BEAD_DIR_NAME
        bead_paths = sorted(bead_dir.glob("*.arc_data"), key=lambda path: build_natural_key(path.name))
        if not bead_paths:
            payload = _failure_payload(date_path.name, f"未找到小球谱：{bead_dir}")
            _write_json(temporary / "affine.json", payload)
            published = _publish_shift(temporary, output_date)
            return {**payload, "shift_dir": str(published)}

        results = _build_results(bead_paths)
        primary_valid = [result for result in results if _has_valid_primary(result.primary)]
        affine = _fit_date_affine(results)
        if affine is None:
            payload = _failure_payload(date_path.name, "没有可用于日期级仿射拟合的小球谱")
            payload["bead_results"] = [_result_payload(result) for result in results]
            for result in results:
                _save_calibration_figure(result, None, None, "needs_review", temporary / f"{result.path.stem}_calibration.png")
            _write_json(temporary / "affine.json", payload)
            published = _publish_shift(temporary, output_date)
            return {**payload, "shift_dir": str(published)}

        diagnostic_scale, diagnostic_offset, primary_residuals = affine
        for result in primary_valid:
            wavenumbers, intensities = read_arc_data(result.path)
            result.auxiliary.update(_fit_auxiliary(result, diagnostic_scale, diagnostic_offset, wavenumbers, intensities))
        valid_results = [
            result
            for result in primary_valid
            if all(result.auxiliary.get(target, PeakFit(target, None, None, None, "missing")).status == "ok" for target in AUXILIARY_PEAKS)
        ]
        auxiliary_residuals = {
            # 辅助峰是在 _fit_auxiliary 中的校正后波数轴上拟合，中心已经完成一次仿射变换。
            target: float(np.median([result.auxiliary[target].center - target for result in valid_results]))
            for target in AUXILIARY_PEAKS
        } if valid_results else {}
        primary_centers = {
            str(target): float(np.median([result.primary[target].center for result in primary_valid]))
            for target in PRIMARY_PEAKS
        }
        auxiliary_centers = {
            str(target): float(np.median([result.auxiliary[target].center for result in valid_results]))
            for target in AUXILIARY_PEAKS
        } if valid_results else {}
        passes = bool(valid_results) and all(
            abs(value) <= RESIDUAL_LIMIT_CM
            for value in (*primary_residuals.values(), *auxiliary_residuals.values())
        )
        valid_count = len(valid_results)
        status = "needs_review"
        if passes:
            status = "ready" if valid_count >= 3 else "limited_replicates"
        payload = {
            "status": status,
            "manual_approval_enable": False,
            "date": date_path.name,
            "formula": "corrected_wavenumber = scale * raw_wavenumber + offset",
            "scale": diagnostic_scale if valid_count else None,
            "offset": diagnostic_offset if valid_count else None,
            "diagnostic_scale": diagnostic_scale,
            "diagnostic_offset": diagnostic_offset,
            "valid_repeat_count": valid_count,
            "primary_valid_repeat_count": len(primary_valid),
            "minimum_valid_repeat_count": 1,
            "fit_method": "pseudo_voigt",
            "primary_peaks_cm-1": list(PRIMARY_PEAKS),
            "auxiliary_peaks_cm-1": list(AUXILIARY_PEAKS),
            "residual_limit_cm-1": RESIDUAL_LIMIT_CM,
            "primary_residuals_cm-1": {str(target): value for target, value in primary_residuals.items()},
            "auxiliary_residuals_cm-1": {str(target): value for target, value in auxiliary_residuals.items()},
            "primary_centers_cm-1": primary_centers,
            "auxiliary_centers_cm-1": auxiliary_centers,
            "errors": [] if passes else ["峰拟合或残差未通过日期级验收"],
            "bead_results": [_result_payload(result) for result in results],
        }
        plot_scale = diagnostic_scale if valid_count else None
        plot_offset = diagnostic_offset if valid_count else None
        for result in results:
            _save_calibration_figure(result, plot_scale, plot_offset, status, temporary / f"{result.path.stem}_calibration.png")
        _write_json(temporary / "affine.json", payload)
        published = _publish_shift(temporary, output_date)
        return {**payload, "shift_dir": str(published)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if not had_output_date and output_date.exists() and not any(output_date.iterdir()):
            output_date.rmdir()
        raise


def analyze_all_dates(source_root: Path | str, output_root: Path | str = OUTPUT_ROOT) -> list[dict[str, object]]:
    """分析所有日期；单个日期失败时写入诊断并继续。"""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{root}")
    reports = []
    for date_dir in sorted((path for path in root.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda path: path.name):
        try:
            reports.append(analyze_date(date_dir, output_root))
        except (OSError, RuntimeError, ValueError) as error:
            output_date = Path(output_root).resolve() / date_dir.name
            existing_shift = output_date / SHIFT_DIR_NAME
            if existing_shift.is_dir():
                payload = _failure_payload(date_dir.name, str(error))
                reports.append({**payload, "shift_dir": str(existing_shift)})
                continue
            had_output_date = output_date.exists()
            temporary = output_date / f".shift_building_{uuid4().hex}"
            payload = _failure_payload(date_dir.name, str(error))
            try:
                temporary.mkdir(parents=True, exist_ok=False)
                _write_json(temporary / "affine.json", payload)
                published = _publish_shift(temporary, output_date)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                if not had_output_date and output_date.exists() and not any(output_date.iterdir()):
                    output_date.rmdir()
                raise
            reports.append({**payload, "shift_dir": str(published)})
    return reports


def load_calibration_report(output_root: Path | str, date_name: str) -> dict[str, object] | None:
    """读取一个日期的 affine.json；尚未分析时返回 None。"""
    path = Path(output_root).resolve() / date_name / SHIFT_DIR_NAME / "affine.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"校正报告必须是 JSON 对象：{path}")
    payload["shift_dir"] = str(path.parent)
    return payload


def _copy_corrected_date(
    source_date: Path,
    temporary_date: Path,
    scale: float,
    offset: float,
    shift_source: Path,
) -> int:
    count = 0
    excluded = {BEAD_DIR_NAME, SHIFT_DIR_NAME, "delete", "audit", "audit_runs", ".raman_pipeline"}
    for source in sorted(source_date.rglob("*"), key=lambda path: build_natural_key(str(path.relative_to(source_date)))):
        relative = source.relative_to(source_date)
        if any(part in excluded for part in relative.parts):
            continue
        target = temporary_date / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if source.suffix.lower() == ".arc_data":
            wavenumbers, intensities = read_arc_data(source)
            if not wavenumbers.size or wavenumbers.size != intensities.size:
                raise ValueError(f"无效光谱：{source}")
            # shift_data 只保存日期级仿射结果，文件夹补偿在 init/CSdata 发布时应用。
            write_arc_data(target, scale * wavenumbers + offset, intensities)
            count += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if count == 0:
        raise RuntimeError(f"日期没有可校正的普通光谱：{source_date}")
    shutil.copytree(shift_source, temporary_date / SHIFT_DIR_NAME)
    return count


def apply_folder_affine_compensation(
    source_root: Path | str,
    output_root: Path | str,
    date_name: str,
    folder_offsets: dict[str, float],
) -> dict[str, object]:
    """只记录文件夹补偿；shift_data 光谱始终保持日期级校正结果。"""
    source_root_path = Path(source_root).resolve()
    output_root_path = Path(output_root).resolve()
    report = load_calibration_report(output_root_path, str(date_name))
    if report is None:
        raise FileNotFoundError(f"尚未分析日期：{date_name}")
    if report.get("status") != "manual_approved":
        raise ValueError(f"日期尚未完成仿真校准：{date_name}")
    requested = {str(name): float(value) for name, value in folder_offsets.items()}
    if any(not math.isfinite(value) for value in requested.values()):
        raise ValueError("文件夹补偿必须是有限数值")
    source_date = source_root_path / str(date_name)
    lineage = load_folder_lineage(source_root_path)
    if not lineage.get("folders"):
        lineage = initialize_folder_lineage(source_root_path)
    lineage_before = load_folder_lineage(source_root_path)
    records = lineage.get("folders", {})
    known = set()
    if isinstance(records, dict):
        known.update(
            key.split("/", 1)[1]
            for key in records
            if str(key).startswith(f"{date_name}/") and "/" in str(key)
        )
    if source_date.is_dir():
        known.update(
            item.name
            for item in source_date.iterdir()
            if item.is_dir() and item.name not in {BEAD_DIR_NAME, SHIFT_DIR_NAME, "delete", "audit", "audit_runs", ".raman_pipeline"}
        )
    unknown = sorted(set(requested) - known, key=build_natural_key)
    if unknown:
        raise ValueError(f"补偿文件夹不存在：{unknown}")
    previous_raw = report.get("folder_compensations", {})
    previous = {str(name): float(value) for name, value in previous_raw.items()} if isinstance(previous_raw, dict) else {}
    changed = {name: value for name, value in requested.items() if value != previous.get(name, 0.0)}
    if not changed:
        return {**report, "changed_folders": [], "compensation_record_only": True}
    current = dict(previous)
    for name, value in changed.items():
        if value == 0.0:
            current.pop(name, None)
        else:
            current[name] = value
    history = list(report.get("compensation_history", []))
    history.append({"previous": previous, "changed": changed, "recorded_at": datetime.now(timezone.utc).isoformat()})
    updated = {
        **report,
        "status": "manual_approved",
        "folder_compensations": current,
        "operator_input": changed,
        "effective_offsets": {name: float(report.get("offset", 0.0)) + value for name, value in current.items()},
        "compensation_history": history,
        "compensation_record_only": True,
        "compensated_at": datetime.now(timezone.utc).isoformat(),
        "changed_folders": sorted(changed, key=build_natural_key),
    }
    updated.pop("shift_dir", None)
    report_path = output_root_path / str(date_name) / SHIFT_DIR_NAME / "affine.json"
    temporary = create_temporary_path(report_path)
    try:
        for name, value in changed.items():
            record_folder_compensation(source_root_path, str(date_name), name, value)
        _write_json(temporary, updated)
        publish_file(temporary, report_path)
    except Exception:
        save_folder_lineage(source_root_path, lineage_before)
        restore_report = create_temporary_path(report_path)
        try:
            _write_json(restore_report, report)
            publish_file(restore_report, report_path)
        finally:
            restore_report.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return updated

def approve_and_apply_date(
    date_dir: Path | str,
    output_root: Path | str = OUTPUT_ROOT,
) -> dict[str, object]:
    """确认一个日期并原子发布只包含日期级仿射的 shift 光谱。"""
    source_date = Path(date_dir).resolve()
    output_root_path = Path(output_root).resolve()
    report = load_calibration_report(output_root_path, source_date.name)
    if report is None:
        raise FileNotFoundError(f"尚未分析日期：{source_date.name}")
    if report.get("status") not in {"ready", "limited_replicates", "manual_approved"}:
        raise ValueError(f"日期不能确认：{report.get('status')}")
    scale, offset = report.get("scale"), report.get("offset")
    if scale is None or offset is None:
        raise ValueError("仿射参数不完整")

    output_date = output_root_path / source_date.name
    shift_source = output_date / SHIFT_DIR_NAME
    temporary_date = output_root_path / f".{source_date.name}_building_{uuid4().hex}"
    temporary_date.mkdir(parents=True, exist_ok=False)
    try:
        _copy_corrected_date(
            source_date,
            temporary_date,
            float(scale),
            float(offset),
            shift_source,
        )
        report["status"] = "manual_approved"
        report["manual_approval_enable"] = True
        report["approved_at"] = datetime.now(timezone.utc).isoformat()
        report["base_scale"] = float(scale)
        report["base_offset"] = float(offset)
        lineage = initialize_folder_lineage(source_date.parent)
        records = lineage.get("folders", {})
        resolved_offsets = {}
        if isinstance(records, dict):
            for key, record in records.items():
                if not str(key).startswith(f"{source_date.name}/") or not isinstance(record, dict):
                    continue
                compensation = record.get("compensation", {})
                value = compensation.get("value", 0.0) if isinstance(compensation, dict) else 0.0
                if float(value) != 0.0:
                    resolved_offsets[str(key).split("/", 1)[1]] = float(value)
        report["folder_compensations"] = resolved_offsets
        report["effective_offsets"] = {
            name: float(offset) + value
            for name, value in resolved_offsets.items()
        }
        report.pop("shift_dir", None)
        _write_json(temporary_date / SHIFT_DIR_NAME / "affine.json", report)
        backup = output_root_path / f"{source_date.name}_previous_{uuid4().hex[:8]}"
        if output_date.exists():
            rename_directory(output_date, backup)
        try:
            rename_directory(temporary_date, output_date)
        except Exception:
            if backup.exists() and not output_date.exists():
                rename_directory(backup, output_date)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary_date, ignore_errors=True)
        raise
    record_affine_approval(
        source_date.parent,
        source_date.name,
        report.get("approved_at"),
        float(report["scale"]),
        float(report["offset"]),
    )
    initialize_folder_lineage(source_date.parent)
    for folder in source_date.iterdir():
        if folder.is_dir() and folder.name not in {BEAD_DIR_NAME, SHIFT_DIR_NAME, "delete", "audit", "audit_runs", ".raman_pipeline"}:
            if any(folder.rglob("*.arc_data")):
                record_output_path(
                    source_date.parent,
                    source_date.name,
                    folder.name,
                    shift_path=str(output_date / folder.name),
                )
    return report


__all__ = [
    "AUXILIARY_PEAKS",
    "OUTPUT_ROOT",
    "PRIMARY_PEAKS",
    "analyze_all_dates",
    "analyze_date",
    "apply_folder_affine_compensation",
    "approve_and_apply_date",
    "load_calibration_report",
]
