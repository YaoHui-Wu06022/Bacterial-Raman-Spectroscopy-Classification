"""数据集同前缀总览图。"""

from __future__ import annotations

import csv
import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ramanv2.common.naming import (
    SPECIAL_FOLDER_NAMES,
    build_natural_key,
    parse_folder_prefix,
)
from ramanv2.common.arc_data import is_arc_data_name, list_arc_data_paths, read_raw_arc_data
from ramanv2.common.plotting import add_bad_band_spans, configure_matplotlib_fonts
from ramanv2.core.config import InputConfig
from ramanv2.data.config import DEFAULT_BUILD_CONFIG, DataBuildConfig
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.bad_bands import build_valid_mask
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import preprocess_single_spectrum


PLOT_CACHE_DIR_NAME = ".prefix_plot_cache"
PLOT_RENDER_VERSION = "3"
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
PLOT_STATE_FIELDS = ("genus", "folder", "prefix", "file_count", "file_signature", "render_version")
PLOT_EXCLUDED_NAMES = SPECIAL_FOLDER_NAMES


@dataclass(frozen=True)
class PrefixPlotConfig:
    """prefix-plot 所需的输入轴和基线处理配置。"""

    input: InputConfig = InputConfig()
    cleaning: DataBuildConfig = DEFAULT_BUILD_CONFIG
    flat: bool = False


DEFAULT_PREFIX_PLOT_CONFIG = PrefixPlotConfig()


def resolve_plot_config(config: PrefixPlotConfig | None = None) -> PrefixPlotConfig:
    """返回调用方配置，缺失时使用数据模块默认值。"""
    return DEFAULT_PREFIX_PLOT_CONFIG if config is None else config


def build_plot_axis(config: PrefixPlotConfig) -> np.ndarray:
    """按模型输入网格构建绘图波数轴，让预览谱与训练看到的谱保持一致。"""
    input_config = config.input
    return build_wn_ref(
        input_config.cut_min,
        input_config.cut_max,
        input_config.target_points,
    ).astype(np.float32)


def _resolve_plot_cache_dir(init_dir: Path, config: PrefixPlotConfig) -> Path:
    """返回中位谱缓存目录；按绘图参数分桶，并与图片输出目录解耦。"""
    digest = hashlib.sha256(
        repr((PLOT_RENDER_VERSION, config.input, config.cleaning)).encode("utf-8")
    ).hexdigest()[:12]
    return Path(init_dir).resolve().parent / PLOT_CACHE_DIR_NAME / digest


def _load_folder_median_cache(
    path: Path,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """读取单个文件夹的 raw/SNV/minmax 中位谱缓存；缺失、损坏或轴长不符时返回 None。"""
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            curves = tuple(
                np.asarray(data[name], dtype=np.float32) for name in ("raw", "snv", "minmax")
            )
    except (OSError, ValueError, KeyError):
        return None
    return curves if all(curve.shape == axis.shape for curve in curves) else None


def _write_folder_median_cache(
    path: Path,
    curves: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """原子写入单个文件夹的中位谱缓存，供后续增量更新直接复用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez(file, raw=curves[0], snv=curves[1], minmax=curves[2])
    os.replace(temporary, path)


def _prune_folder_median_cache(cache_root: Path, state: list[dict[str, str]]) -> None:
    """删除与当前 init 文件夹状态不再匹配的中位谱缓存。"""
    if not cache_root.is_dir():
        return
    current = {
        (row["genus"], f"{row['folder']}__{row['file_signature']}.npz") for row in state
    }
    for genus_dir in cache_root.iterdir():
        if not genus_dir.is_dir():
            continue
        for path in genus_dir.iterdir():
            if path.name.endswith(".npz") and (genus_dir.name, path.name) not in current:
                path.unlink(missing_ok=True)


def iter_init_folders(init_dir: Path) -> list[Path]:
    """列出 init 中全部属和文件夹目录。"""
    if not init_dir.is_dir():
        return []
    return [
        folder_dir
        for genus_dir in sorted(
            path for path in init_dir.iterdir() if path.is_dir() and path.name not in PLOT_EXCLUDED_NAMES
        )
        for folder_dir in sorted(
            path for path in genus_dir.iterdir() if path.is_dir() and path.name not in PLOT_EXCLUDED_NAMES
        )
    ]


def _scatter_valid_segment(
    shape: tuple[int, ...],
    valid_mask: np.ndarray | None,
    values: np.ndarray,
) -> np.ndarray:
    """把紧凑有效段的数值放回完整绘图轴，坏段位置保持 NaN 供绘图断线。"""
    curve = np.full(shape, np.nan, dtype=np.float32)
    if valid_mask is None:
        curve[:] = values
    else:
        curve[valid_mask] = values
    return curve


def build_folder_median_curves(
    folder_dir: Path,
    axis: np.ndarray,
    config: PrefixPlotConfig,
    cache_dir: Path | None = None,
    signature: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """逐条按训练清洗链路处理光谱，再汇总 raw、归一化与 minmax 中位谱。

    预处理直接复用 :func:`preprocess_single_spectrum`，与 build_train 写入的
    train 谱完全一致；坏段点位在绘图轴上留 NaN。
    同时给出 cache_dir 和文件摘要时优先命中缓存，只在文件夹内容变化后重算。
    """
    cache_path = (
        cache_dir / f"{folder_dir.name}__{signature}.npz"
        if cache_dir is not None and signature
        else None
    )
    if cache_path is not None:
        cached = _load_folder_median_cache(cache_path, axis)
        if cached is not None:
            return cached
    input_config = config.input
    build_config = config.cleaning
    reference_axis = build_wn_ref(
        input_config.cut_min,
        input_config.cut_max,
        int(input_config.target_points),
    )
    valid_mask = build_valid_mask(reference_axis, input_config.bad_bands)
    raw_curves = []
    norm_curves = []
    minmax_curves = []
    for path in list_arc_data_paths(folder_dir):
        wavenumbers, intensities, malformed_lines = read_raw_arc_data(path)
        if malformed_lines or wavenumbers.size < 2:
            continue
        if np.any(np.diff(wavenumbers) <= 0):
            continue
        target_axis, corrected, _ = preprocess_single_spectrum(
            wavenumbers,
            intensities,
            cut_min=input_config.cut_min,
            cut_max=input_config.cut_max,
            reference_wavenumbers=reference_axis,
            bad_bands=input_config.bad_bands,
            baseline_lam=build_config.baseline_lam,
            baseline_asls_p=build_config.baseline_asls_p,
            baseline_max_iter=build_config.baseline_max_iter,
            baseline_fit_min=build_config.baseline_fit_min,
            baseline_fit_max=build_config.baseline_fit_max,
            baseline_method=build_config.baseline_method,
        )
        if target_axis is None:
            continue
        raw_curve = np.full(axis.shape, np.nan, dtype=np.float32)
        inside = (axis >= wavenumbers[0]) & (axis <= wavenumbers[-1])
        if inside.any():
            raw_curve[inside] = np.interp(axis[inside], wavenumbers, intensities)
        raw_curves.append(raw_curve)
        norm_curves.append(
            _scatter_valid_segment(
                axis.shape,
                valid_mask,
                normalize_spectrum(corrected, input_config.norm_method),
            )
        )
        minmax_curves.append(
            _scatter_valid_segment(
                axis.shape,
                valid_mask,
                normalize_spectrum(corrected, "minmax"),
            )
        )
    if not raw_curves:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        curves = tuple(
            np.nanmedian(np.vstack(curves), axis=0).astype(np.float32, copy=False)
            for curves in (raw_curves, norm_curves, minmax_curves)
        )
    if cache_path is not None:
        _write_folder_median_cache(cache_path, curves)
    return curves


def read_plot_state(path: Path) -> dict[tuple[str, str], object]:
    """读取前缀图的上次文件夹和内容摘要状态。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row["genus"], row["folder"]): {
                "prefix": row["prefix"],
                "file_count": row.get("file_count", ""),
                "file_signature": row.get("file_signature", ""),
            }
            for row in csv.DictReader(file, delimiter="\t")
            if row.get("genus")
            and row.get("folder")
            and row.get("prefix")
            and row.get("render_version") == PLOT_RENDER_VERSION
        }


def write_plot_state(path: Path, rows: list[dict[str, str]]) -> None:
    """保存本次前缀图使用的文件夹前缀状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLOT_STATE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["genus"], row["folder"])))


def build_plot_state(folders: list[Path]) -> list[dict[str, str]]:
    """整理文件夹前缀和文件摘要，用于增量更新总览图。"""
    rows = []
    for folder in folders:
        file_count, file_signature = _scan_folder_spectra(folder)
        rows.append(
            {
                "genus": folder.parent.name,
                "folder": folder.name,
                "prefix": parse_folder_prefix(folder.name),
                "file_count": str(file_count),
                "file_signature": file_signature,
                "render_version": PLOT_RENDER_VERSION,
            }
        )
    return rows


def build_changed_plot_groups(
    current_state: list[dict[str, str]],
    prior_state: dict[tuple[str, str], object],
) -> set[tuple[str, str]]:
    """比较前后文件夹集合，返回需要重画的属与前缀分组。"""
    current_map = {
        (row["genus"], row["folder"]): row["prefix"]
        for row in current_state
    }
    changed = set()
    for folder_key in set(current_map) | set(prior_state):
        current_row = next((row for row in current_state if (row["genus"], row["folder"]) == folder_key), None)
        current_prefix = current_row.get("prefix") if current_row else None
        prior_value = prior_state.get(folder_key)
        prior_prefix = prior_value.get("prefix") if isinstance(prior_value, dict) else prior_value
        current_signature = (current_row or {}).get("file_signature")
        prior_signature = prior_value.get("file_signature") if isinstance(prior_value, dict) else None
        current_count = (current_row or {}).get("file_count")
        prior_count = prior_value.get("file_count") if isinstance(prior_value, dict) else None
        if current_prefix == prior_prefix and (
            not current_signature
            or not prior_signature
            or (current_signature == prior_signature and current_count == prior_count)
        ):
            continue
        genus = folder_key[0]
        if current_prefix is not None:
            changed.add((genus, current_prefix))
        if prior_prefix is not None:
            changed.add((genus, prior_prefix))
    return changed


def _scan_folder_spectra(folder: Path) -> tuple[int, str]:
    """单次 scandir 统计源谱数量，并汇总名称、大小和修改时间摘要。"""
    entries = [
        (entry.name, entry.stat())
        for entry in os.scandir(folder)
        if entry.is_file() and is_arc_data_name(entry.name)
    ]
    entries.sort(key=lambda item: item[0].lower())
    digest = hashlib.sha256()
    for name, stat in entries:
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return len(entries), digest.hexdigest()


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
    norm_method: str,
) -> None:
    """绘制同属同前缀的 raw、归一化与 minmax 中位谱总览。"""
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    norm_label = str(norm_method).upper()
    labels = ("Raw intensity", f"{norm_label} intensity", "minmax intensity")
    subtitles = (
        "raw median",
        f"{norm_label} median with vertical offsets",
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
        add_bad_band_spans(subplot, bad_bands, alpha=0.35)
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


def plot_prefix_init(
    init_dir: Path | str,
    output_dir: Path | str,
    force_enable: bool = False,
    config: PrefixPlotConfig | None = None,
    *,
    progress_callback: Callable[[str, str, int, int], None] | None = None,
) -> list[Path]:
    """从任意属/种两层 init 目录生成 prefix-plot。

    ``progress_callback(属, 前缀, 已处理分组数, 待处理分组数)`` 在每组绘制后调用，
    供界面显示进度。
    """
    configure_matplotlib_fonts()
    plot_config = resolve_plot_config(config)
    input_dir = Path(init_dir).resolve()
    figure_dir = Path(output_dir).resolve()
    folders = iter_init_folders(input_dir)
    state = build_plot_state(folders)
    signatures = {(row["genus"], row["folder"]): row["file_signature"] for row in state}
    cache_root = _resolve_plot_cache_dir(input_dir, plot_config)
    prior = read_plot_state(figure_dir / "prefix_plot_state.csv")
    changed = build_changed_plot_groups(state, prior)
    axis = build_plot_axis(plot_config)
    grouped = {}
    for folder in folders:
        grouped.setdefault((folder.parent.name, parse_folder_prefix(folder.name)), []).append(folder)
    current_groups = set(grouped)
    for genus, prefix in changed - current_groups:
        stale_output = figure_dir / genus / f"{prefix}.png"
        if stale_output.is_file():
            stale_output.unlink()
    targets = [
        key
        for key in sorted(grouped)
        if force_enable or key in changed or not (figure_dir / key[0] / f"{key[1]}.png").is_file()
    ]
    total = len(targets)
    outputs = []
    for index, (genus, prefix) in enumerate(targets, start=1):
        group = grouped[(genus, prefix)]
        output_path = figure_dir / genus / f"{prefix}.png"
        curves = {}
        for folder in group:
            median_curves = build_folder_median_curves(
                folder,
                axis,
                plot_config,
                cache_root / genus,
                signatures.get((genus, folder.name)),
            )
            if median_curves is not None:
                curves[folder.name] = median_curves
        if curves:
            plot_prefix_group(
                curves,
                axis,
                output_path,
                f"{prefix} median spectra" if plot_config.flat else f"{genus} {prefix} median spectra",
                plot_config.input.bad_bands,
                plot_config.input.norm_method,
            )
            outputs.append(output_path)
        if progress_callback is not None:
            progress_callback(genus, prefix, index, total)
    _prune_folder_median_cache(cache_root, state)
    write_plot_state(figure_dir / "prefix_plot_state.csv", state)
    return outputs


def build_prefix_plots(
    init_root: Path | str,
    output_root: Path | str,
    force_enable: bool = False,
    config: PrefixPlotConfig | None = None,
    *,
    progress_callback: Callable[[str, str, int, int], None] | None = None,
) -> list[Path]:
    """为指定 init 目录生成同前缀总览图。"""
    return plot_prefix_init(init_root, output_root, force_enable, config, progress_callback=progress_callback)


def build_prefix_plots_for_groups(
    init_root: Path | str,
    output_root: Path | str,
    group_keys: set[tuple[str, str]] | list[tuple[str, str]] | tuple[tuple[str, str], ...],
    config: PrefixPlotConfig | None = None,
) -> list[Path]:
    """只重画指定属/前缀分组，不扫描其他 init 文件夹。"""
    configure_matplotlib_fonts()
    plot_config = resolve_plot_config(config)
    input_dir = Path(init_root).resolve()
    figure_dir = Path(output_root).resolve()
    axis = build_plot_axis(plot_config)
    cache_root = _resolve_plot_cache_dir(input_dir, plot_config)
    wanted = {(str(genus), str(prefix).upper()) for genus, prefix in group_keys}
    outputs: list[Path] = []
    for genus, prefix in sorted(wanted, key=lambda item: (build_natural_key(item[0]), build_natural_key(item[1]))):
        genus_dir = input_dir / genus
        if not genus_dir.is_dir():
            continue
        folders = [
            folder
            for folder in sorted(genus_dir.iterdir(), key=lambda item: build_natural_key(item.name))
            if folder.is_dir() and parse_folder_prefix(folder.name).upper() == prefix
        ]
        curves = {}
        for folder in folders:
            median_curves = build_folder_median_curves(
                folder,
                axis,
                plot_config,
                cache_root / genus,
                _scan_folder_spectra(folder)[1],
            )
            if median_curves is not None:
                curves[folder.name] = median_curves
        output_path = figure_dir / genus / f"{prefix}.png"
        if not curves:
            continue
        plot_prefix_group(
            curves,
            axis,
            output_path,
            f"{prefix} median spectra" if plot_config.flat else f"{genus} {prefix} median spectra",
            plot_config.input.bad_bands,
            plot_config.input.norm_method,
        )
        outputs.append(output_path)
    return outputs


__all__ = [
    "DEFAULT_PREFIX_PLOT_CONFIG",
    "PrefixPlotConfig",
    "build_changed_plot_groups",
    "build_prefix_plots",
    "build_prefix_plots_for_groups",
    "build_plot_state",
    "plot_prefix_init",
    "resolve_plot_config",
]
