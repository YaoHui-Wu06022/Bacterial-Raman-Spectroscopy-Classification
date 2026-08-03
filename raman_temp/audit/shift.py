"""1002 cm⁻¹ 锚点平移、delta 计划与文件夹波数写回。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from raman_temp.audit.config import AuditConfig, resolve_audit_config
from raman_temp.audit.io import read_raw_spectrum
from raman_temp.audit.reports import append_delta_log, read_delta, write_delta
from raman_temp.common.naming import parse_folder_prefix
from raman_temp.spectra.axis import build_wn_ref


SHIFT_FIELDS = (
    "genus",
    "folder",
    "prefix",
    "current_delta",
    "raw_from_zero_delta",
    "target_delta",
    "step_delta",
    "anchor_peak_cm",
    "anchor_model",
    "anchor_quality",
    "fit_bic_gain",
    "fit_separation_cm",
    "status",
    "residual_range",
    "total_shift_limit",
)

APPLIED_SHIFT_STATUSES = {"rebased_applied", "rebased_to_zero", "fixed_special_total"}


@dataclass(frozen=True)
class ManualShiftTarget:
    """描述人工平移的一侧目录、delta 文件与对应记录键。"""

    folder_dir: Path
    delta_path: Path
    genus: str
    folder: str
    prefix: str


def build_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """返回同长度移动平均，用于代表谱基线抑制和锚点峰搜索。"""
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def build_folder_curve(folder: Path, config: AuditConfig, wavenumber_offset: float = 0.0) -> np.ndarray | None:
    """将文件夹内可用原始谱插值并聚合为 MAD 标准化的代表谱。"""
    spectrum_config = config.input
    axis = build_wn_ref(
        spectrum_config.cut_min,
        spectrum_config.cut_max,
        spectrum_config.target_points,
    )
    curves = []
    for path in sorted(folder.glob("*.arc_data")):
        try:
            wavenumbers, intensities, malformed_lines = read_raw_spectrum(path)
        except OSError:
            continue
        if (
            malformed_lines
            or wavenumbers.size < config.raw_min_points
            or np.any(np.diff(wavenumbers) <= 0)
        ):
            continue
        shifted = wavenumbers + wavenumber_offset
        if shifted.min() > spectrum_config.cut_min or shifted.max() < spectrum_config.cut_max:
            continue
        curve = np.interp(axis, shifted, intensities)
        curve = curve - build_moving_average(curve, 101)
        scale = float(np.median(np.abs(curve - np.median(curve))))
        if scale > 1e-8:
            curves.append(curve / (1.4826 * scale))
    return np.median(np.asarray(curves), axis=0) if curves else None


def build_single_peak_model(
    axis: np.ndarray,
    baseline: float,
    slope: float,
    amplitude: float,
    center: float,
    width: float,
    config: AuditConfig,
) -> np.ndarray:
    """构造带线性局部基线的单高斯锚点峰模型。"""
    return baseline + slope * (axis - config.shift_target_cm) + amplitude * np.exp(-0.5 * ((axis - center) / width) ** 2)


def build_double_peak_model(
    axis: np.ndarray,
    baseline: float,
    slope: float,
    amplitude_1: float,
    center_1: float,
    width_1: float,
    amplitude_2: float,
    center_2: float,
    width_2: float,
    config: AuditConfig,
) -> np.ndarray:
    """构造带线性局部基线的双高斯锚点峰模型。"""
    return (
        baseline
        + slope * (axis - config.shift_target_cm)
        + amplitude_1 * np.exp(-0.5 * ((axis - center_1) / width_1) ** 2)
        + amplitude_2 * np.exp(-0.5 * ((axis - center_2) / width_2) ** 2)
    )


def calculate_fit_score(values: np.ndarray, fitted: np.ndarray, parameter_count: int) -> tuple[float, float]:
    """计算高斯拟合的 BIC 与决定系数。"""
    residual_sum = max(float(np.sum((values - fitted) ** 2)), 1e-12)
    total_sum = max(float(np.sum((values - np.mean(values)) ** 2)), 1e-12)
    bic = len(values) * math.log(residual_sum / len(values)) + parameter_count * math.log(len(values))
    return bic, 1.0 - residual_sum / total_sum


def fit_single_anchor(axis: np.ndarray, values: np.ndarray, config: AuditConfig) -> dict[str, float | str] | None:
    """拟合 1002 cm⁻¹ 邻域单峰并返回位置、质量和 BIC。"""
    baseline = float(np.median(values))
    scale = max(float(np.ptp(values)), 0.1)
    peak_index = int(np.argmax(values))
    try:
        params, _ = curve_fit(
            lambda x, baseline, slope, amplitude, center, width: build_single_peak_model(x, baseline, slope, amplitude, center, width, config),
            axis,
            values,
            p0=(baseline, 0.0, max(float(values[peak_index] - baseline), 0.05), float(axis[peak_index]), 2.0),
            bounds=((-10 * scale, -5 * scale, 0.0, config.shift_fit_min, 0.3), (10 * scale, 5 * scale, 20 * scale, config.shift_fit_max, 8.0)),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return None
    bic, r2 = calculate_fit_score(values, build_single_peak_model(axis, *params, config), 5)
    return {"model": "single", "position": float(params[3]), "bic": bic, "r2": r2}


def fit_double_anchor(axis: np.ndarray, values: np.ndarray, peak_indices: np.ndarray, config: AuditConfig) -> dict[str, float | str] | None:
    """枚举可信双峰组合，并返回 BIC 最优的锚点拟合结果。"""
    baseline = float(np.median(values))
    scale = max(float(np.ptp(values)), 0.1)
    best = None
    for rank, left_index in enumerate(peak_indices):
        for right_index in peak_indices[rank + 1:]:
            if not config.shift_double_min_separation <= axis[right_index] - axis[left_index] <= config.shift_double_max_separation:
                continue
            try:
                params, _ = curve_fit(
                    lambda x, baseline, slope, amplitude_1, center_1, width_1, amplitude_2, center_2, width_2: build_double_peak_model(x, baseline, slope, amplitude_1, center_1, width_1, amplitude_2, center_2, width_2, config),
                    axis,
                    values,
                    p0=(baseline, 0.0, max(float(values[left_index] - baseline), 0.05), float(axis[left_index]), 1.8, max(float(values[right_index] - baseline), 0.05), float(axis[right_index]), 1.8),
                    bounds=((-10 * scale, -5 * scale, 0.0, config.shift_fit_min, 0.3, 0.0, config.shift_fit_min, 0.3), (10 * scale, 5 * scale, 20 * scale, config.shift_fit_max, 8.0, 20 * scale, config.shift_fit_max, 8.0)),
                    maxfev=30_000,
                )
            except (RuntimeError, ValueError):
                continue
            components = sorted(((float(params[2]), float(params[3])), (float(params[5]), float(params[6]))), key=lambda item: item[1])
            separation = components[1][1] - components[0][1]
            amplitude_ratio = min(components[0][0], components[1][0]) / max(components[0][0], components[1][0], 1e-12)
            if not config.shift_double_min_separation <= separation <= config.shift_double_max_separation or amplitude_ratio < config.shift_double_min_amplitude_ratio:
                continue
            bic, r2 = calculate_fit_score(values, build_double_peak_model(axis, *params, config), 8)
            candidate = {"model": "double", "position": 0.5 * (components[0][1] + components[1][1]), "bic": bic, "r2": r2, "separation": separation, "amplitude_ratio": amplitude_ratio}
            if best is None or float(candidate["bic"]) < float(best["bic"]):
                best = candidate
    return best


def fit_anchor(curve: np.ndarray, axis: np.ndarray, config: AuditConfig) -> dict[str, float | str] | None:
    """在锚点范围选择可信的单峰或双峰拟合结果。"""
    mask = (axis >= config.shift_fit_min) & (axis <= config.shift_fit_max) & np.isfinite(curve)
    if mask.sum() < 10:
        return None
    fit_axis = axis[mask]
    values = curve[mask]
    single = fit_single_anchor(fit_axis, values, config)
    if single is None or float(single["r2"]) < config.shift_fit_min_r2:
        return None
    spread = float(np.percentile(values, 95) - np.percentile(values, 5))
    peak_indices, _ = find_peaks(values, prominence=max(spread * 0.10, 1e-6), distance=2)
    double = fit_double_anchor(fit_axis, values, peak_indices, config) if len(peak_indices) >= 2 else None
    if double is not None and float(double["r2"]) >= config.shift_fit_min_r2 and float(single["bic"]) - float(double["bic"]) >= config.shift_double_bic_gain:
        double["bic_gain"] = float(single["bic"]) - float(double["bic"])
        return double
    single["bic_gain"] = float(single["bic"]) - float(double["bic"]) if double is not None else 0.0
    return single


def find_peak_anchor(curve: np.ndarray, axis: np.ndarray, config: AuditConfig) -> tuple[float, float] | None:
    """从平滑代表谱返回局部最高锚点位置及其显著性比例。"""
    mask = (axis >= config.shift_anchor_min) & (axis <= config.shift_anchor_max) & np.isfinite(curve)
    if mask.sum() < config.shift_smooth_window:
        return None
    values = build_moving_average(curve[mask], config.shift_smooth_window)
    peak_index = int(np.argmax(values))
    prominence = float(values[peak_index] - np.median(values))
    spread = float(np.percentile(values, 95) - np.percentile(values, 5))
    if spread <= 1e-8 or prominence <= spread * 0.15:
        return None
    return float(axis[mask][peak_index]), prominence / spread


def build_shift_plan(input_dir: Path, current_delta: dict[tuple[str, str], float], config: AuditConfig | None = None) -> list[dict[str, object]]:
    """计算运行池所有文件夹的目标累计平移，不修改任何光谱文件。"""
    audit_config = resolve_audit_config(config)
    axis = build_wn_ref(
        audit_config.input.cut_min,
        audit_config.input.cut_max,
        audit_config.input.target_points,
    )
    plan = []
    for genus_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            genus, folder = genus_dir.name, folder_dir.name
            current = current_delta.get((genus, folder), 0.0)
            curve = build_folder_curve(folder_dir, audit_config, wavenumber_offset=-current)
            raw_total: float | str = ""
            peak_cm = ""
            position: float | None = None
            model = ""
            quality = ""
            bic_gain: float | str = ""
            separation: float | str = ""
            desired = current
            status = "unresolved"
            if curve is not None:
                fitted = fit_anchor(curve, axis, audit_config) if genus in audit_config.fungal_genera else None
                peak = find_peak_anchor(curve, axis, audit_config) if fitted is None else None
                if fitted is not None:
                    position = float(fitted["position"])
                    peak_cm = f"{position:.3f}"
                    model = f"fung_{fitted['model']}_fit"
                    quality = f"r2={float(fitted['r2']):.6f}"
                    bic_gain = f"{float(fitted['bic_gain']):.6f}"
                    separation = f"{float(fitted['separation']):.6f}" if "separation" in fitted else ""
                elif peak is not None:
                    position, peak_quality = peak
                    peak_cm = f"{position:.3f}"
                    model = "moving_average_peak"
                    quality = f"prominence_ratio={peak_quality:.6f}"
                if position is not None:
                    raw_total = round(audit_config.shift_target_cm - position, 1)
            residual_min, residual_max = (audit_config.shift_fungal_residual_min, audit_config.shift_fungal_residual_max) if genus in audit_config.fungal_genera else (audit_config.shift_default_residual_min, audit_config.shift_default_residual_max)
            fixed = audit_config.get_fixed_shift(genus, folder)
            if fixed is not None:
                desired, status = fixed, "fixed_special_total"
            elif raw_total == "" or abs(float(raw_total)) > audit_config.shift_limit:
                status = "unresolved"
            elif folder not in audit_config.shift_large_move_folders and abs(float(raw_total)) > audit_config.shift_total_limit:
                status = "unresolved"
            elif position is not None and (position < residual_min or position > residual_max):
                desired, status = float(raw_total), "rebased_applied"
            else:
                desired = 0.0
                status = "kept_within_residual" if abs(current) < 1e-9 else "rebased_to_zero"
            plan.append({"genus": genus, "folder": folder, "prefix": parse_folder_prefix(folder), "current_delta": current, "raw_from_zero_delta": raw_total, "target_delta": desired, "step_delta": desired - current, "anchor_peak_cm": peak_cm, "anchor_model": model, "anchor_quality": quality, "fit_bic_gain": bic_gain, "fit_separation_cm": separation, "status": status, "residual_range": f"{residual_min:.0f}-{residual_max:.0f}", "total_shift_limit": "" if folder in audit_config.shift_large_move_folders else f"{audit_config.shift_total_limit:.0f}"})
    return plan


def apply_folder_shift(folder: Path, delta: float) -> int:
    """对一个文件夹内全部可解析谱追加波数平移，并返回文件数量。"""
    changed_count = 0
    for path in sorted(folder.glob("*.arc_data")):
        wavenumbers, intensities, malformed_lines = read_raw_spectrum(path)
        if malformed_lines or not wavenumbers.size:
            continue
        temporary_path = path.with_suffix(path.suffix + ".audit_tmp")
        np.savetxt(temporary_path, np.column_stack((wavenumbers + delta, intensities)), fmt=["%.3f", "%.6f"], delimiter="\t")
        temporary_path.replace(path)
        changed_count += 1
    return changed_count


def apply_shift_plan(input_dir: Path, plan: list[dict[str, object]]) -> list[dict[str, object]]:
    """执行可应用的计划步骤并返回实际发生文件写回的计划行。"""
    applied = []
    for row in plan:
        step = float(row["step_delta"])
        if row["status"] not in APPLIED_SHIFT_STATUSES or abs(step) < 1e-9:
            continue
        folder = input_dir / str(row["genus"]) / str(row["folder"])
        row["files_changed"] = apply_folder_shift(folder, step)
        applied.append(row)
    return applied


def build_unresolved_folder_keys(plan: list[dict[str, object]]) -> set[tuple[str, str]]:
    """从自动平移计划提取不能进入相似性判定的文件夹键。"""
    return {
        (str(row["genus"]), str(row["folder"]))
        for row in plan
        if row["status"] == "unresolved"
    }


def apply_manual_shift(
    target: ManualShiftTarget,
    delta: float,
    counterpart: ManualShiftTarget | None = None,
) -> dict[str, object]:
    """对指定 `init/` 文件夹追加人工平移，并在需要时同步对应副本。"""
    if abs(delta) < 1e-9:
        raise ValueError("人工平移量不能为 0")
    if not target.folder_dir.is_dir():
        raise FileNotFoundError(f"人工平移目录不存在：{target.folder_dir}")
    target_files = sorted(target.folder_dir.glob("*.arc_data"))
    if not target_files:
        raise FileNotFoundError(f"人工平移目录没有 .arc_data：{target.folder_dir}")
    target_values = read_delta(target.delta_path)
    target_key = (target.genus, target.folder)
    current = target_values.get(target_key, 0.0)
    counterpart_values: dict[tuple[str, str], float] = {}
    counterpart_key: tuple[str, str] | None = None
    counterpart_files: list[Path] = []
    if counterpart is not None:
        if not counterpart.folder_dir.is_dir():
            raise FileNotFoundError(f"对应人工平移目录不存在：{counterpart.folder_dir}")
        counterpart_files = sorted(counterpart.folder_dir.glob("*.arc_data"))
        if not counterpart_files:
            raise FileNotFoundError(f"对应人工平移目录没有 .arc_data：{counterpart.folder_dir}")
        counterpart_values = read_delta(counterpart.delta_path)
        counterpart_key = (counterpart.genus, counterpart.folder)
        counterpart_current = counterpart_values.get(counterpart_key, 0.0)
        if not math.isclose(current, counterpart_current, abs_tol=1e-9):
            raise ValueError(
                f"对应 delta 不一致：{target.genus}/{target.folder}={current:+g}，"
                f"{counterpart.genus}/{counterpart.folder}={counterpart_current:+g}"
            )
    target_changed = apply_folder_shift(target.folder_dir, delta)
    counterpart_changed = apply_folder_shift(counterpart.folder_dir, delta) if counterpart is not None else 0
    cumulative = current + delta
    target_values[target_key] = cumulative
    write_delta(
        target.delta_path,
        {
            key: (parse_folder_prefix(folder), value)
            for (genus, folder), value in target_values.items()
            for key in [(genus, folder)]
        },
    )
    now = datetime.now().isoformat(timespec="seconds")
    append_delta_log(
        target.delta_path.with_name("delta_log.txt"),
        [{"time": now, "genus": target.genus, "folder": target.folder, "prefix": target.prefix, "step_delta": f"{delta:+g}", "cumulative_delta": f"{cumulative:+g}", "files_changed": target_changed, "note": "manual_shift"}],
    )
    if counterpart is not None and counterpart_key is not None:
        counterpart_values[counterpart_key] = cumulative
        write_delta(
            counterpart.delta_path,
            {
                key: (parse_folder_prefix(folder), value)
                for (genus, folder), value in counterpart_values.items()
                for key in [(genus, folder)]
            },
        )
        append_delta_log(
            counterpart.delta_path.with_name("delta_log.txt"),
            [{"time": now, "genus": counterpart.genus, "folder": counterpart.folder, "prefix": counterpart.prefix, "step_delta": f"{delta:+g}", "cumulative_delta": f"{cumulative:+g}", "files_changed": counterpart_changed, "note": f"manual_sync={target.genus}/{target.folder}"}],
        )
    return {"folder": f"{target.genus}/{target.folder}", "step_delta": delta, "cumulative_delta": cumulative, "files_changed": target_changed, "counterpart_files_changed": counterpart_changed}
