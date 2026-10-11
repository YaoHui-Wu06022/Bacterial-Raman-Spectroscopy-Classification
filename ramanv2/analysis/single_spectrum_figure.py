"""用单条真实光谱生成预处理流程图。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator

from ramanv2.common.arc_data import read_raw_arc_data
from ramanv2.common.plotting import configure_matplotlib_fonts
from ramanv2.core.config import InputConfig
from ramanv2.data.config import DEFAULT_BUILD_CONFIG, DataBuildConfig
from ramanv2.data.runtime.augmentation import AugmentationSpec, augment_raw_spectrum
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.filters import sg_coeff
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import estimate_baseline


DEFAULT_SPECTRUM_PATH = Path(
    "dataset/cosmic_data/delete/manual/20240106/ESK01/CELL2_Area01_001.arc_data"
)
DEFAULT_OUTPUT_DIR = Path("output/single_spectrum_pipeline")
VISUAL_AXIS_WARP_ALPHA = 0.01
VISUAL_AXIS_WARP_BETA = 8.0


@dataclass(frozen=True)
class PipelineTrace:
    """保存一条光谱在六个流程阶段的数值结果。"""

    source_path: Path
    raw_wavenumbers: np.ndarray
    raw_intensities: np.ndarray
    baseline_wavenumbers: np.ndarray
    baseline_values: np.ndarray
    corrected_values: np.ndarray
    cut_wavenumbers: np.ndarray
    cut_values: np.ndarray
    interpolated_wavenumbers: np.ndarray
    interpolated_values: np.ndarray
    normalized_values: np.ndarray
    model_input: np.ndarray


@dataclass(frozen=True)
class AugmentationTrace:
    """保存同一条统一波数光谱的可重复增强结果。"""

    source_path: Path
    wavenumbers: np.ndarray
    original_values: np.ndarray
    variants: dict[str, np.ndarray]


def build_pipeline_trace(
    spectrum_path: Path | str,
    input_config: InputConfig = InputConfig(),
    build_config: DataBuildConfig = DEFAULT_BUILD_CONFIG,
) -> PipelineTrace:
    """按当前数据处理顺序构建单条光谱的六阶段数值轨迹。

    本流程有意将坏区设置为空。它用于展示指定光谱的连续处理过程，
    不会删除、遮盖或断开任何波数区间。
    """
    source_path = Path(spectrum_path).resolve()
    wavenumbers, intensities, malformed_lines = read_raw_arc_data(source_path)
    if malformed_lines:
        raise ValueError(f"光谱包含无法解析的行：{source_path} ({malformed_lines})")
    if wavenumbers.size < 10 or intensities.size != wavenumbers.size:
        raise ValueError(f"光谱点数不足或列长度不一致：{source_path}")
    if not np.isfinite(wavenumbers).all() or not np.isfinite(intensities).all():
        raise ValueError(f"光谱包含非有限值：{source_path}")
    if not np.all(np.diff(wavenumbers) > 0):
        raise ValueError(f"光谱波数轴不是严格递增：{source_path}")

    # 流程图不做坏区处理，保留当前配置的裁切、插值和标准化参数。
    visual_input_config = replace(input_config, bad_bands=())
    fit_mask = (wavenumbers >= build_config.baseline_fit_min) & (
        wavenumbers <= build_config.baseline_fit_max
    )
    fit_wavenumbers = wavenumbers[fit_mask]
    fit_intensities = intensities[fit_mask]
    if fit_wavenumbers.size < 10:
        raise ValueError(f"光谱在基线拟合范围内的点数不足：{source_path}")

    baseline = estimate_baseline(
        fit_intensities,
        method=build_config.baseline_method,
        lam=build_config.baseline_lam,
        p=build_config.baseline_asls_p,
        niter=build_config.baseline_max_iter,
        valid_mask=None,
    )
    corrected = fit_intensities - baseline
    cut_mask = (fit_wavenumbers >= visual_input_config.cut_min) & (
        fit_wavenumbers <= visual_input_config.cut_max
    )
    cut_wavenumbers = fit_wavenumbers[cut_mask]
    cut_values = corrected[cut_mask]
    if cut_wavenumbers.size < 2:
        raise ValueError(f"光谱在裁切范围内的点数不足：{source_path}")

    interpolated_wavenumbers = build_wn_ref(
        visual_input_config.cut_min,
        visual_input_config.cut_max,
        visual_input_config.target_points,
    ).astype(np.float32)
    interpolated_values = np.interp(
        interpolated_wavenumbers,
        cut_wavenumbers,
        cut_values,
    ).astype(np.float32)
    normalized_values = normalize_spectrum(
        interpolated_values,
        visual_input_config.norm_method,
    ).astype(np.float32, copy=False)

    # 流程图按要求显示原谱、平滑谱和一阶导数谱三路输入。
    step_cm = (visual_input_config.cut_max - visual_input_config.cut_min) / (
        visual_input_config.target_points - 1
    )
    smooth_values = np.correlate(
        interpolated_values,
        sg_coeff(visual_input_config.win_smooth, 3, 0),
        mode="same",
    ).astype(np.float32)
    d1_values = np.correlate(
        smooth_values,
        sg_coeff(visual_input_config.win1, 3, 1),
        mode="same",
    ).astype(np.float32) / step_cm
    model_input = np.stack(
        (
            normalized_values,
            normalize_spectrum(smooth_values, visual_input_config.norm_method),
            normalize_spectrum(d1_values, visual_input_config.norm_method),
        ),
        axis=0,
    ).astype(np.float32)

    arrays = (
        wavenumbers,
        intensities,
        fit_wavenumbers,
        baseline,
        corrected,
        cut_wavenumbers,
        cut_values,
        interpolated_wavenumbers,
        interpolated_values,
        normalized_values,
        model_input,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"流程结果包含非有限值：{source_path}")
    return PipelineTrace(
        source_path=source_path,
        raw_wavenumbers=wavenumbers.astype(np.float32),
        raw_intensities=intensities.astype(np.float32),
        baseline_wavenumbers=fit_wavenumbers.astype(np.float32),
        baseline_values=baseline.astype(np.float32),
        corrected_values=corrected.astype(np.float32),
        cut_wavenumbers=cut_wavenumbers.astype(np.float32),
        cut_values=cut_values.astype(np.float32),
        interpolated_wavenumbers=interpolated_wavenumbers,
        interpolated_values=interpolated_values,
        normalized_values=normalized_values,
        model_input=model_input.astype(np.float32),
    )


def render_pipeline_figures(trace: PipelineTrace, output_dir: Path | str) -> tuple[Path, ...]:
    """输出总览图、六张独立小图以及元数据。"""
    configure_matplotlib_fonts()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    stages = (
        ("01_raw", _plot_raw),
        ("02_baseline_removed", _plot_baseline_removed),
        ("03_cut", _plot_cut),
        ("04_interpolated", _plot_interpolated),
        ("05_normalized", _plot_normalized),
        ("06_model_input", _plot_model_input),
    )
    for style in ("ppt", "clean"):
        figure = _build_figure(trace, style)
        for suffix, options in (("png", {"dpi": 300}), ("svg", {})):
            path = output_path / f"single_spectrum_pipeline_{style}.{suffix}"
            figure.savefig(
                path,
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
                **options,
            )
            saved_paths.append(path)
        plt.close(figure)
        style_dir = output_path / style
        style_dir.mkdir(parents=True, exist_ok=True)
        for stage_name, plotter in stages:
            stage_figure = _build_stage_figure(trace, style, plotter)
            for suffix, options in (("png", {"dpi": 300}), ("svg", {})):
                path = style_dir / f"{stage_name}.{suffix}"
                stage_figure.savefig(
                    path,
                    bbox_inches=None,
                    facecolor=stage_figure.get_facecolor(),
                    **options,
                )
                saved_paths.append(path)
            plt.close(stage_figure)

    metadata = {
        "source_path": str(trace.source_path),
        "bad_bands": [],
        "raw_points": int(trace.raw_wavenumbers.size),
        "raw_range_cm1": [
            float(trace.raw_wavenumbers.min()),
            float(trace.raw_wavenumbers.max()),
        ],
        "cut_range_cm1": [
            float(trace.interpolated_wavenumbers.min()),
            float(trace.interpolated_wavenumbers.max()),
        ],
        "interpolated_points": int(trace.interpolated_wavenumbers.size),
        "model_input_shape": list(trace.model_input.shape),
    }
    metadata_path = output_path / "single_spectrum_pipeline_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saved_paths.append(metadata_path)
    return tuple(saved_paths)


def build_augmentation_trace(trace: PipelineTrace, seed: int = 42) -> AugmentationTrace:
    """对统一波数光谱分别施加单项训练增强，固定随机种子。"""
    specs = (
        ("01_piecewise_gain", AugmentationSpec(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)),
        ("02_noise", AugmentationSpec(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1)),
        ("03_axis_warp", AugmentationSpec(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1)),
        ("04_weak_baseline", AugmentationSpec(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1)),
        ("05_strong_baseline", AugmentationSpec(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1)),
        ("06_mask_attenuation", AugmentationSpec(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1)),
    )
    with _temporary_numpy_seed(seed):
        variants = {}
        for name, spec in specs:
            if name == "03_axis_warp":
                variants[name] = _augment_axis_warp_for_visual(trace.interpolated_values)
            else:
                variants[name] = augment_raw_spectrum(trace.interpolated_values, spec)
    if not all(np.isfinite(values).all() for values in variants.values()):
        raise ValueError(f"数据增强结果包含非有限值：{trace.source_path}")
    return AugmentationTrace(
        source_path=trace.source_path,
        wavenumbers=trace.interpolated_wavenumbers,
        original_values=trace.interpolated_values,
        variants=variants,
    )


def render_augmentation_figures(
    trace: PipelineTrace,
    output_dir: Path | str,
    seed: int = 42,
) -> tuple[Path, ...]:
    """输出数据增强总览图和每种增强的独立 PNG/SVG。"""
    configure_matplotlib_fonts()
    augmentation_trace = build_augmentation_trace(trace, seed)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    selected_names = (
        "01_piecewise_gain",
        "05_strong_baseline",
        "03_axis_warp",
    )
    for style in ("ppt", "clean"):
        figure = _build_augmentation_figure(augmentation_trace, style)
        for suffix, options in (("png", {"dpi": 300}), ("svg", {})):
            path = output_path / f"single_spectrum_augmentation_{style}.{suffix}"
            figure.savefig(
                path,
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
                **options,
            )
            saved_paths.append(path)
        plt.close(figure)
        style_dir = output_path / "augmentation" / style
        style_dir.mkdir(parents=True, exist_ok=True)
        for name, values in augmentation_trace.variants.items():
            stage_figure = _build_augmentation_stage_figure(
                augmentation_trace,
                values,
                style,
            )
            for suffix, options in (("png", {"dpi": 300}), ("svg", {})):
                path = style_dir / f"{name}.{suffix}"
                stage_figure.savefig(
                    path,
                    bbox_inches=None,
                    facecolor=stage_figure.get_facecolor(),
                    **options,
                )
                saved_paths.append(path)
            plt.close(stage_figure)
        selected_figure = _build_selected_augmentation_figure(
            augmentation_trace,
            selected_names,
            style,
        )
        for suffix, options in (("png", {"dpi": 300}), ("svg", {})):
            path = output_path / f"single_spectrum_augmentation_selected_{style}.{suffix}"
            selected_figure.savefig(
                path,
                bbox_inches=None,
                facecolor=selected_figure.get_facecolor(),
                **options,
            )
            saved_paths.append(path)
        plt.close(selected_figure)

    metadata = {
        "source_path": str(augmentation_trace.source_path),
        "seed": int(seed),
        "input_points": int(augmentation_trace.wavenumbers.size),
        "variants": list(augmentation_trace.variants),
        "selected_variants": list(selected_names),
        "visual_axis_warp_alpha": VISUAL_AXIS_WARP_ALPHA,
        "visual_axis_warp_beta": VISUAL_AXIS_WARP_BETA,
        "training_config_unchanged": True,
    }
    metadata_path = output_path / "single_spectrum_augmentation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saved_paths.append(metadata_path)
    return tuple(saved_paths)


@contextmanager
def _temporary_numpy_seed(seed: int):
    """固定增强示例的随机性，并恢复调用方原有 NumPy 状态。"""
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _augment_axis_warp_for_visual(values: np.ndarray) -> np.ndarray:
    """使用增强后的轴扰动幅度生成更容易辨识的展示样本。"""
    values = np.asarray(values, dtype=np.float32)
    length = values.size
    positions = np.arange(length, dtype=np.float32)
    centered = positions - (length - 1) / 2.0
    linear = np.random.uniform(-VISUAL_AXIS_WARP_ALPHA, VISUAL_AXIS_WARP_ALPHA) * centered
    nonlinear = np.random.uniform(-VISUAL_AXIS_WARP_BETA, VISUAL_AXIS_WARP_BETA) * np.sin(
        2 * np.pi * positions / length + np.random.uniform(0, 2 * np.pi)
    )
    warped = np.clip(positions + linear + nonlinear, 0, length - 1)
    return np.interp(positions, warped, values).astype(np.float32)


def _build_figure(trace: PipelineTrace, style: str):
    if style == "ppt":
        figure = plt.figure(figsize=(18, 4.8), facecolor="#f5f8fd")
        grid = figure.add_gridspec(1, 6, left=0.025, right=0.975, bottom=0.13, top=0.91, wspace=0.24)
        axes = [figure.add_subplot(grid[0, column]) for column in range(6)]
        for axis in axes:
            axis.set_facecolor("white")
            for spine in axis.spines.values():
                spine.set_color("#1d5fc2")
                spine.set_linewidth(1.6)
        figure.canvas.draw()
        for left, right in zip(axes, axes[1:]):
            left_box = left.get_position()
            right_box = right.get_position()
            figure.add_artist(
                FancyArrowPatch(
                    (left_box.x1 + 0.006, (left_box.y0 + left_box.y1) / 2),
                    (right_box.x0 - 0.006, (right_box.y0 + right_box.y1) / 2),
                    transform=figure.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.5,
                    color="#1d5fc2",
                )
            )
    elif style == "clean":
        figure, axes_grid = plt.subplots(2, 3, figsize=(15, 7.6), facecolor="white")
        axes = list(axes_grid.flat)
        figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.96, wspace=0.25, hspace=0.34)
        for axis in axes:
            axis.set_facecolor("white")
            for spine in axis.spines.values():
                spine.set_color("#5f6368")
                spine.set_linewidth(0.9)
    else:
        raise ValueError(f"未知流程图风格：{style}")

    _plot_raw(axes[0], trace)
    _plot_baseline_removed(axes[1], trace)
    _plot_cut(axes[2], trace)
    _plot_interpolated(axes[3], trace)
    _plot_normalized(axes[4], trace)
    _plot_model_input(axes[5], trace)
    return figure


def _build_stage_figure(trace: PipelineTrace, style: str, plotter):
    """为单个阶段建立可独立插入 PPT 的小图。"""
    facecolor = "#f5f8fd" if style == "ppt" else "white"
    figure, axis = plt.subplots(figsize=(4.8, 3.2), facecolor=facecolor)
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_color("#1d5fc2" if style == "ppt" else "#5f6368")
        spine.set_linewidth(1.5 if style == "ppt" else 0.9)
    plotter(axis, trace)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.97)
    return figure


def _build_augmentation_figure(trace: AugmentationTrace, style: str):
    """建立数据增强变体总览图。"""
    facecolor = "#f5f8fd" if style == "ppt" else "white"
    figure, axes_grid = plt.subplots(2, 3, figsize=(15, 7.6), facecolor=facecolor)
    axes = list(axes_grid.flat)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.96, wspace=0.25, hspace=0.34)
    for axis, values in zip(axes, trace.variants.values()):
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#1d5fc2" if style == "ppt" else "#5f6368")
            spine.set_linewidth(1.5 if style == "ppt" else 0.9)
        _plot_augmentation(axis, trace, values)
    return figure


def _build_augmentation_stage_figure(
    trace: AugmentationTrace,
    values: np.ndarray,
    style: str,
):
    """建立单个增强变体图，保留原谱作灰色参照。"""
    facecolor = "#f5f8fd" if style == "ppt" else "white"
    figure, axis = plt.subplots(figsize=(4.8, 3.2), facecolor=facecolor)
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_color("#1d5fc2" if style == "ppt" else "#5f6368")
        spine.set_linewidth(1.5 if style == "ppt" else 0.9)
    _plot_augmentation(axis, trace, values)
    figure.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.97)
    return figure


def _build_selected_augmentation_figure(
    trace: AugmentationTrace,
    selected_names: tuple[str, ...],
    style: str,
):
    """将指定的三种增强在同一坐标轴内上下错位绘制。"""
    facecolor = "#f5f8fd" if style == "ppt" else "white"
    figure, axis = plt.subplots(figsize=(4.8, 3.2), facecolor=facecolor)
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_color("#1d5fc2" if style == "ppt" else "#5f6368")
        spine.set_linewidth(1.5 if style == "ppt" else 0.9)
    _plot_selected_augmentation_stacked(axis, trace, selected_names)
    figure.subplots_adjust(left=0.14, right=0.95, bottom=0.16, top=0.97)
    return figure


def _plot_augmentation(
    axis,
    trace: AugmentationTrace,
    values: np.ndarray,
) -> None:
    """在同一坐标轴叠加原谱和单项增强谱。"""
    axis.plot(trace.wavenumbers, trace.original_values, color="#a8b0ba", linewidth=0.8)
    axis.plot(trace.wavenumbers, values, color="#175dcc", linewidth=1.0)
    _prepare_axis(axis, 600.0, 1800.0, [600, 900, 1200, 1500, 1800])


def _plot_selected_augmentation_stacked(
    axis,
    trace: AugmentationTrace,
    selected_names: tuple[str, ...],
) -> None:
    """将三种增强曲线堆叠显示，底部到顶部对应传入顺序。"""
    values = [trace.variants[name] for name in selected_names]
    combined = np.concatenate(values)
    curve_span = float(np.max(combined) - np.min(combined))
    offset = max(curve_span * 1.08, 1.0)
    colors = ("#175dcc", "#2f8f46", "#d06a19")
    for index, (variant, color) in enumerate(zip(values, colors)):
        level = index * offset
        reference = trace.original_values + level
        enhanced = variant + level
        difference = np.abs(variant - trace.original_values)
        difference_threshold = float(np.percentile(difference, 90))
        difference_mask = difference >= difference_threshold
        axis.axhline(level, color="#b4bac2", linewidth=0.65, linestyle="--", zorder=0)
        axis.plot(
            trace.wavenumbers,
            reference,
            color="#a8b0ba",
            linewidth=0.8,
            zorder=1,
        )
        axis.fill_between(
            trace.wavenumbers,
            reference,
            enhanced,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            trace.wavenumbers,
            enhanced,
            color=color,
            linewidth=1.0,
            zorder=2,
        )
        axis.scatter(
            trace.wavenumbers[difference_mask],
            enhanced[difference_mask],
            color=color,
            s=4,
            linewidths=0,
            zorder=3,
        )
    _prepare_axis(axis, 600.0, 1800.0, [600, 900, 1200, 1500, 1800])


def _prepare_axis(axis, x_min: float, x_max: float, ticks: list[float] | None) -> None:
    """统一坐标范围和数字刻度，不添加轴名称或图例。"""
    axis.set_xlim(x_min, x_max)
    if ticks is None:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        axis.set_xticks(ticks)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="both", labelsize=8, length=3, width=0.8)
    axis.ticklabel_format(axis="both", style="plain", useOffset=False)
    axis.grid(alpha=0.12, linewidth=0.6)


def _plot_raw(axis, trace: PipelineTrace) -> None:
    axis.plot(trace.raw_wavenumbers, trace.raw_intensities, color="#175dcc", linewidth=1.0)
    _prepare_axis(axis, 200.0, 2200.0, [200, 600, 1000, 1400, 1800, 2200])


def _plot_baseline_removed(axis, trace: PipelineTrace) -> None:
    axis.plot(trace.baseline_wavenumbers, trace.corrected_values, color="#175dcc", linewidth=1.0)
    axis.axhline(0.0, color="#8d939b", linewidth=0.7, linestyle="--")
    _prepare_axis(
        axis,
        float(trace.baseline_wavenumbers[0]),
        float(trace.baseline_wavenumbers[-1]),
        None,
    )


def _plot_cut(axis, trace: PipelineTrace) -> None:
    axis.plot(trace.cut_wavenumbers, trace.cut_values, color="#175dcc", linewidth=1.0)
    _prepare_axis(axis, 600.0, 1800.0, [600, 900, 1200, 1500, 1800])


def _plot_interpolated(axis, trace: PipelineTrace) -> None:
    axis.plot(trace.cut_wavenumbers, trace.cut_values, color="#a8b0ba", linewidth=0.8)
    axis.plot(trace.interpolated_wavenumbers, trace.interpolated_values, color="#175dcc", linewidth=1.0)
    _prepare_axis(axis, 600.0, 1800.0, [600, 900, 1200, 1500, 1800])


def _plot_normalized(axis, trace: PipelineTrace) -> None:
    axis.plot(trace.interpolated_wavenumbers, trace.normalized_values, color="#175dcc", linewidth=1.0)
    axis.axhline(0.0, color="#8d939b", linewidth=0.7, linestyle="--")
    _prepare_axis(axis, 600.0, 1800.0, [600, 900, 1200, 1500, 1800])


def _plot_model_input(axis, trace: PipelineTrace) -> None:
    values = trace.model_input
    finite = values[np.isfinite(values)]
    span = float(np.percentile(finite, 95) - np.percentile(finite, 5))
    span = max(span, 1e-6)
    offset = span * 1.35
    for channel_index, channel in enumerate(values):
        axis.plot(
            trace.interpolated_wavenumbers,
            channel + channel_index * offset,
            color=("#175dcc", "#2f8f46", "#d06a19")[channel_index],
            linewidth=0.9,
        )
    _prepare_axis(axis, 600.0, 1800.0, [600, 900, 1200, 1500, 1800])


def main() -> None:
    """生成计划中指定光谱的两套流程图。"""
    trace = build_pipeline_trace(DEFAULT_SPECTRUM_PATH)
    paths = render_pipeline_figures(trace, DEFAULT_OUTPUT_DIR)
    paths += render_augmentation_figures(trace, DEFAULT_OUTPUT_DIR)
    print("Generated:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SPECTRUM_PATH",
    "AugmentationTrace",
    "PipelineTrace",
    "build_augmentation_trace",
    "build_pipeline_trace",
    "render_augmentation_figures",
    "render_pipeline_figures",
]
