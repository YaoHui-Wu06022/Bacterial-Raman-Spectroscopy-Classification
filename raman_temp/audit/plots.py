"""审核前缀总览图与单文件夹平移对照图。"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman_temp.audit.config import AuditConfig, resolve_audit_config
from raman_temp.audit.io import read_raw_spectrum
from raman_temp.audit.reports import read_delta
from raman_temp.common.naming import parse_folder_prefix
from raman_temp.data.profiles import get_dataset_dir, get_profile
from raman_temp.spectra.bands import build_valid_mask
from raman_temp.spectra.normalize import normalize_spectrum
from raman_temp.spectra.preprocess import estimate_baseline


PLOT_STEP_CM = 0.2
PLOT_TARGET_CM = 1002.0
PLOT_COLORS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")
PLOT_STATE_FIELDS = ("genus", "folder", "prefix", "delta")


@dataclass(frozen=True)
class PlotPaths:
    """描述一个数据集的审核绘图输入与输出位置。"""

    dataset_dir: Path
    init_dir: Path
    output_dir: Path
    delta_path: Path


def resolve_plot_paths(dataset_key: str) -> PlotPaths:
    """按 profile 解析审核绘图所需的数据集路径。"""
    profile = get_profile(dataset_key)
    dataset_dir = get_dataset_dir(profile)
    return PlotPaths(
        dataset_dir=dataset_dir,
        init_dir=dataset_dir / profile.root_init,
        output_dir=dataset_dir / "fig_init",
        delta_path=dataset_dir / "delta.txt",
    )


def build_plot_axis(config: AuditConfig) -> np.ndarray:
    """按审核光谱范围构建绘图使用的等步长波数轴。"""
    spectrum = config.input
    return np.arange(
        spectrum.cut_min,
        spectrum.cut_max + PLOT_STEP_CM * 0.5,
        PLOT_STEP_CM,
        dtype=np.float32,
    )


def is_transferred_folder(folder_name: str) -> bool:
    """判断目录是否为测试菌派生的 `*t` 文件夹。"""
    return str(folder_name).lower().endswith("t")


def iter_init_folders(init_dir: Path, include_transferred_enable: bool) -> list[Path]:
    """列出 init 中可参与审核绘图的属和文件夹目录。"""
    folders = []
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            if not include_transferred_enable and is_transferred_folder(folder_dir.name):
                continue
            folders.append(folder_dir)
    return folders


def build_folder_raw_median(folder_dir: Path, axis: np.ndarray, offset_cm: float = 0.0) -> np.ndarray | None:
    """读取一个文件夹并在绘图轴上构建原始强度中位谱。"""
    curves = []
    for path in sorted(folder_dir.glob("*.arc_data")):
        wavenumbers, intensities, malformed_lines = read_raw_spectrum(path)
        if malformed_lines or wavenumbers.size < 2:
            continue
        order = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[order] + offset_cm
        intensities = intensities[order]
        if np.any(np.diff(wavenumbers) <= 0):
            continue
        curve = np.full(axis.shape, np.nan, dtype=np.float32)
        inside = (axis >= wavenumbers[0]) & (axis <= wavenumbers[-1])
        if inside.any():
            curve[inside] = np.interp(axis[inside], wavenumbers, intensities)
            curves.append(curve)
    if not curves:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.asarray(curves), axis=0).astype(np.float32, copy=False)


def preprocess_preview_curve(
    raw_curve: np.ndarray,
    axis: np.ndarray,
    config: AuditConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """将原始中位谱扣基线后生成 SNV 和 minmax 预览曲线。"""
    corrected = np.full(raw_curve.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(raw_curve)
    if finite.sum() >= 10:
        input_config = config.input
        cleaning_config = config.cleaning
        fit_mask = finite & (
            (axis >= cleaning_config.baseline_fit_min)
            & (axis <= cleaning_config.baseline_fit_max)
        )
        if fit_mask.sum() < 10:
            fit_mask = finite
        valid_mask = build_valid_mask(axis[fit_mask], input_config.bad_bands)
        baseline = estimate_baseline(
            raw_curve[fit_mask],
            method=cleaning_config.baseline_method,
            lam=cleaning_config.baseline_lam,
            p=cleaning_config.baseline_asls_p,
            niter=cleaning_config.baseline_max_iter,
            valid_mask=valid_mask,
        )
        corrected[fit_mask] = raw_curve[fit_mask] - baseline
    return (
        normalize_spectrum(corrected, "snv", preserve_nan_enable=True),
        normalize_spectrum(corrected, "minmax", preserve_nan_enable=True),
    )


def resolve_folder_dir(init_dir: Path, folder_value: str) -> Path:
    """解析属/文件夹路径或 init 内唯一文件夹名称。"""
    normalized = str(folder_value).strip().replace("\\", "/").strip("/")
    direct = (init_dir / normalized).resolve()
    root = init_dir.resolve()
    try:
        direct.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"审核文件夹超出 init 目录：{folder_value}") from exc
    if direct.is_dir():
        return direct
    matches = [path for path in init_dir.glob(f"*/{normalized}") if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"未找到审核文件夹：{folder_value}")
    raise ValueError(f"文件夹名称不唯一，请指定属/文件夹：{folder_value}")


def read_plot_state(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """读取前缀图上次使用的文件夹累计平移状态。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row["genus"], row["folder"]): (row["prefix"], row["delta"])
            for row in csv.DictReader(file, delimiter="\t")
            if row.get("genus") and row.get("folder")
        }


def write_plot_state(path: Path, rows: list[dict[str, str]]) -> None:
    """保存本次前缀图使用的文件夹累计平移状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLOT_STATE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["genus"], row["folder"])))


def build_plot_state(folders: list[Path], delta_values: dict[tuple[str, str], float]) -> list[dict[str, str]]:
    """将当前文件夹和累计平移整理为可比较的绘图状态。"""
    return [
        {
            "genus": folder.parent.name,
            "folder": folder.name,
            "prefix": parse_folder_prefix(folder.name),
            "delta": f"{delta_values.get((folder.parent.name, folder.name), 0.0):+g}",
        }
        for folder in folders
    ]


def plot_prefix_group(curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], axis: np.ndarray, output_path: Path, title: str) -> None:
    """绘制同属同前缀的 raw、SNV 与 minmax 中位谱总览。"""
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    labels = ("Raw intensity", "SNV intensity", "Minmax intensity")
    for curve_index, subplot in enumerate(axes):
        values = [items[curve_index] for items in curves.values()]
        finite = np.concatenate([value[np.isfinite(value)] for value in values if np.isfinite(value).any()])
        span = float(np.percentile(finite, 95) - np.percentile(finite, 5)) if finite.size else 1.0
        offset = 0.0 if curve_index == 0 else max(span * (0.85 if curve_index == 1 else 1.05), 2.0 if curve_index == 1 else 1.05)
        for index, (name, items) in enumerate(curves.items()):
            subplot.plot(axis, items[curve_index] + index * offset, color=PLOT_COLORS[index % len(PLOT_COLORS)], lw=1.1, label=name)
        subplot.axvline(PLOT_TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.4)
        subplot.set_ylabel(labels[curve_index])
        subplot.grid(alpha=0.15)
        subplot.legend(loc="upper left", ncol=2, fontsize=8)
    figure.suptitle(title)
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_prefix_dataset(dataset_key: str = "alldata", include_transferred_enable: bool = True, force_enable: bool = False, config: AuditConfig | None = None) -> list[Path]:
    """按属和前缀生成审核总览图；未变化分组可跳过。"""
    audit_config = resolve_audit_config(config)
    paths = resolve_plot_paths(dataset_key)
    folders = iter_init_folders(paths.init_dir, include_transferred_enable)
    state = build_plot_state(folders, read_delta(paths.delta_path))
    prior = read_plot_state(paths.output_dir / "prefix_plot_state.csv")
    changed = {
        (row["genus"], row["prefix"])
        for row in state
        if prior.get((row["genus"], row["folder"])) != (row["prefix"], row["delta"])
    }
    axis = build_plot_axis(audit_config)
    grouped = {}
    for folder in folders:
        grouped.setdefault((folder.parent.name, parse_folder_prefix(folder.name)), []).append(folder)
    outputs = []
    for (genus, prefix), group in sorted(grouped.items()):
        output_path = paths.output_dir / genus / f"{prefix}.png"
        if not force_enable and output_path.is_file() and (genus, prefix) not in changed:
            continue
        curves = {}
        for folder in group:
            raw_curve = build_folder_raw_median(folder, axis)
            if raw_curve is not None:
                curves[folder.name] = (raw_curve, *preprocess_preview_curve(raw_curve, axis, audit_config))
        if curves:
            plot_prefix_group(curves, axis, output_path, f"{genus} {prefix} median spectra")
            outputs.append(output_path)
    write_plot_state(paths.output_dir / "prefix_plot_state.csv", state)
    return outputs


def plot_shift_folder(dataset_key: str, folder_value: str, config: AuditConfig | None = None) -> Path:
    """绘制指定文件夹累计平移前后与同前缀参考的原始中位谱。"""
    audit_config = resolve_audit_config(config)
    paths = resolve_plot_paths(dataset_key)
    folder = resolve_folder_dir(paths.init_dir, folder_value)
    delta = read_delta(paths.delta_path).get((folder.parent.name, folder.name), 0.0)
    if abs(delta) < 1e-9:
        raise ValueError(f"文件夹没有累计平移记录：{folder.parent.name}/{folder.name}")
    axis = build_plot_axis(audit_config)
    before = build_folder_raw_median(folder, axis, offset_cm=-delta)
    after = build_folder_raw_median(folder, axis)
    if before is None or after is None:
        raise RuntimeError(f"无法构建文件夹中位谱：{folder}")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    prefix = parse_folder_prefix(folder.name)
    references = [path for path in folder.parent.iterdir() if path.is_dir() and path != folder and not is_transferred_folder(path.name) and parse_folder_prefix(path.name) == prefix]
    for subplot, curve, subtitle in zip(axes, (before, after), ("before shift", "after shift")):
        for index, reference in enumerate(sorted(references)):
            reference_curve = build_folder_raw_median(reference, axis)
            if reference_curve is not None:
                subplot.plot(axis, reference_curve, color=PLOT_COLORS[index % len(PLOT_COLORS)], lw=0.9, label=reference.name)
        subplot.plot(axis, curve, color="black", lw=1.2, label=folder.name)
        subplot.axvline(PLOT_TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.4)
        subplot.set_title(subtitle)
        subplot.legend(loc="upper left", ncol=2, fontsize=8)
        subplot.grid(alpha=0.15)
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    figure.suptitle(f"{folder.parent.name} {folder.name}: cumulative shift {delta:+g} cm$^{{-1}}$")
    figure.tight_layout()
    output_path = paths.output_dir / "shift" / prefix / f"{folder.name}_shift_compare.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
