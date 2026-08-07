"""审核同前缀总览图。"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ramanv2.audit.config import AuditConfig, resolve_audit_config
from ramanv2.common.naming import parse_folder_prefix
from ramanv2.data.io import read_raw_arc_data
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.spectra.bands import build_valid_mask
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import estimate_baseline


PLOT_STEP_CM = 0.2
TARGET_CM = 1003.0
PREFIX_PLOT_COLORS = (
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
PLOT_STATE_FIELDS = ("genus", "folder", "prefix")


@dataclass(frozen=True)
class PlotPaths:
    """描述审核总览图的输入与输出位置。"""

    init_dir: Path
    output_dir: Path


def resolve_plot_paths(dataset_key: str) -> PlotPaths:
    """按 profile 解析审核总览图的固定路径。"""
    profile = get_profile(dataset_key)
    dataset_dir = get_dataset_dir(profile)
    return PlotPaths(dataset_dir / profile.root_init, dataset_dir / "fig_init")


def build_plot_axis(config: AuditConfig) -> np.ndarray:
    """按审核输入范围构建等步长绘图波数轴。"""
    input_config = config.input
    return np.arange(
        input_config.cut_min,
        input_config.cut_max + PLOT_STEP_CM * 0.5,
        PLOT_STEP_CM,
        dtype=np.float32,
    )


def iter_init_folders(init_dir: Path) -> list[Path]:
    """列出 init 中全部属和文件夹目录。"""
    return [
        folder_dir
        for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir())
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir())
    ]


def build_folder_median_curves(
    folder_dir: Path,
    axis: np.ndarray,
    config: AuditConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """逐条处理文件夹光谱，再分别汇总 raw、SNV 与 minmax 中位谱。"""
    raw_curves = []
    snv_curves = []
    minmax_curves = []
    for path in sorted(folder_dir.glob("*.arc_data")):
        wavenumbers, intensities, malformed_lines = read_raw_arc_data(path)
        if malformed_lines or wavenumbers.size < 2:
            continue
        if np.any(np.diff(wavenumbers) <= 0):
            continue
        curve = np.full(axis.shape, np.nan, dtype=np.float32)
        inside = (axis >= wavenumbers[0]) & (axis <= wavenumbers[-1])
        if inside.any():
            curve[inside] = np.interp(axis[inside], wavenumbers, intensities)
            raw_curves.append(curve)
            snv_curve, minmax_curve = preprocess_preview_curve(curve, axis, config)
            snv_curves.append(snv_curve)
            minmax_curves.append(minmax_curve)
    if not raw_curves:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return tuple(
            np.nanmedian(np.vstack(curves), axis=0).astype(np.float32, copy=False)
            for curves in (raw_curves, snv_curves, minmax_curves)
        )


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
        build_config = config.cleaning
        fit_mask = finite & (
            (axis >= build_config.baseline_fit_min)
            & (axis <= build_config.baseline_fit_max)
        )
        if fit_mask.sum() < 10:
            fit_mask = finite
        valid_mask = build_valid_mask(axis[fit_mask], input_config.bad_bands)
        baseline = estimate_baseline(
            raw_curve[fit_mask],
            method=build_config.baseline_method,
            lam=build_config.baseline_lam,
            p=build_config.baseline_asls_p,
            niter=build_config.baseline_max_iter,
            valid_mask=valid_mask,
        )
        corrected[fit_mask] = raw_curve[fit_mask] - baseline
    return (
        normalize_spectrum(corrected, "snv", preserve_nan_enable=True),
        normalize_spectrum(corrected, "minmax", preserve_nan_enable=True),
    )


def read_plot_state(path: Path) -> dict[tuple[str, str], str]:
    """读取前缀图的上次文件夹前缀状态。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row["genus"], row["folder"]): row["prefix"]
            for row in csv.DictReader(file, delimiter="\t")
            if row.get("genus") and row.get("folder") and row.get("prefix")
        }


def write_plot_state(path: Path, rows: list[dict[str, str]]) -> None:
    """保存本次前缀图使用的文件夹前缀状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLOT_STATE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["genus"], row["folder"])))


def build_plot_state(folders: list[Path]) -> list[dict[str, str]]:
    """整理文件夹当前前缀，用于跳过未变化的总览图。"""
    return [
        {
            "genus": folder.parent.name,
            "folder": folder.name,
            "prefix": parse_folder_prefix(folder.name),
        }
        for folder in folders
    ]


def build_changed_plot_groups(
    current_state: list[dict[str, str]],
    prior_state: dict[tuple[str, str], str],
) -> set[tuple[str, str]]:
    """比较前后文件夹集合，返回需要重画的属与前缀分组。"""
    current_map = {
        (row["genus"], row["folder"]): row["prefix"]
        for row in current_state
    }
    changed = set()
    for folder_key in set(current_map) | set(prior_state):
        current_prefix = current_map.get(folder_key)
        prior_prefix = prior_state.get(folder_key)
        if current_prefix == prior_prefix:
            continue
        genus = folder_key[0]
        if current_prefix is not None:
            changed.add((genus, current_prefix))
        if prior_prefix is not None:
            changed.add((genus, prior_prefix))
    return changed


def add_bad_band_spans(subplot, bad_bands: tuple[tuple[float, float], ...]) -> None:
    """在总览图中标记不参与处理的坏波段。"""
    for band_min, band_max in bad_bands:
        subplot.axvspan(band_min, band_max, color="gray", alpha=0.35)


def plot_curve_outside_bad_bands(
    subplot,
    axis: np.ndarray,
    values: np.ndarray,
    bad_bands: tuple[tuple[float, float], ...],
    **kwargs,
) -> None:
    """绘制坏区两侧曲线，避免跨越无效区连线。"""
    plotted = np.asarray(values, dtype=np.float32).copy()
    valid_mask = build_valid_mask(axis, bad_bands)
    if valid_mask is not None:
        plotted[~valid_mask] = np.nan
    subplot.plot(axis, plotted, **kwargs)


def plot_prefix_group(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    axis: np.ndarray,
    output_path: Path,
    title: str,
    bad_bands: tuple[tuple[float, float], ...],
) -> None:
    """绘制同属同前缀的 raw、SNV 与 minmax 中位谱总览。"""
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    labels = ("Raw intensity", "SNV intensity", "minmax intensity")
    subtitles = (
        "raw median",
        "SNV median with vertical offsets",
        "minmax median with vertical offsets",
    )
    for curve_index, subplot in enumerate(axes):
        values = [items[curve_index] for items in curves.values()]
        finite = np.concatenate([value[np.isfinite(value)] for value in values if np.isfinite(value).any()])
        span = float(np.percentile(finite, 95) - np.percentile(finite, 5)) if finite.size else 1.0
        offset = 0.0 if curve_index == 0 else max(span * (0.85 if curve_index == 1 else 1.05), 2.0 if curve_index == 1 else 1.05)
        for index, (name, items) in enumerate(curves.items()):
            values = items[curve_index] + index * offset
            options = {
                "color": PREFIX_PLOT_COLORS[index % len(PREFIX_PLOT_COLORS)],
                "lw": 1.3,
                "label": name,
            }
            if curve_index == 0:
                subplot.plot(axis, values, **options)
            else:
                subplot.axhline(index * offset, color="0.88", lw=0.7, zorder=0)
                plot_curve_outside_bad_bands(subplot, axis, values, bad_bands, **options)
        add_bad_band_spans(subplot, bad_bands)
        subplot.axvline(TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.35)
        subplot.set_ylabel(labels[curve_index])
        subplot.set_title(subtitles[curve_index])
        subplot.grid(alpha=0.15)
        subplot.legend(loc="upper left", ncol=2, fontsize=8)
    figure.suptitle(title)
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_prefix_dataset(
    dataset_key: str = "alldata",
    force_enable: bool = False,
    config: AuditConfig | None = None,
) -> list[Path]:
    """按属和前缀生成审核总览图，未变化分组可跳过。"""
    audit_config = resolve_audit_config(config)
    paths = resolve_plot_paths(dataset_key)
    folders = iter_init_folders(paths.init_dir)
    state = build_plot_state(folders)
    prior = read_plot_state(paths.output_dir / "prefix_plot_state.csv")
    changed = build_changed_plot_groups(state, prior)
    axis = build_plot_axis(audit_config)
    grouped = {}
    for folder in folders:
        grouped.setdefault((folder.parent.name, parse_folder_prefix(folder.name)), []).append(folder)
    current_groups = set(grouped)
    for genus, prefix in changed - current_groups:
        stale_output = paths.output_dir / genus / f"{prefix}.png"
        if stale_output.is_file():
            stale_output.unlink()
    outputs = []
    for (genus, prefix), group in sorted(grouped.items()):
        output_path = paths.output_dir / genus / f"{prefix}.png"
        if not force_enable and output_path.is_file() and (genus, prefix) not in changed:
            continue
        curves = {}
        for folder in group:
            median_curves = build_folder_median_curves(folder, axis, audit_config)
            if median_curves is not None:
                curves[folder.name] = median_curves
        if curves:
            plot_prefix_group(curves, axis, output_path, f"{genus} {prefix} median spectra", audit_config.input.bad_bands)
            outputs.append(output_path)
    write_plot_state(paths.output_dir / "prefix_plot_state.csv", state)
    return outputs
