"""训练集均值光谱审图。"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman_temp.core.config import InputConfig
from raman_temp.core.paths import resolve_path
from raman_temp.spectra.bands import build_valid_mask, normalize_bad_bands
from raman_temp.spectra.normalize import normalize_spectrum

from .io import iter_arc_dirs, read_arc_data


@dataclass(frozen=True)
class TrainPlotConfig:
    """训练集均值图使用的归一化和坏段配置。"""

    norm_method: str
    bad_bands: tuple[tuple[float, float], ...]


def build_train_plot_config(input_config: InputConfig) -> TrainPlotConfig:
    """从统一输入配置提取绘图所需的确定性参数。"""
    return TrainPlotConfig(input_config.norm_method, input_config.bad_bands)


def _read_train_group(folder: Path, filenames: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """读取同一训练叶子目录，并验证其波数轴一致。"""
    reference_axis = None
    spectra = []
    for filename in filenames:
        wavenumbers, intensities = read_arc_data(folder / filename)
        if not wavenumbers.size or not intensities.size:
            continue
        if reference_axis is None:
            reference_axis = wavenumbers
        elif wavenumbers.shape != reference_axis.shape or not np.allclose(wavenumbers, reference_axis):
            raise ValueError(f"训练类别波数轴不一致：{folder / filename}")
        spectra.append(intensities)
    if reference_axis is None or not spectra:
        return None
    return reference_axis, np.vstack(spectra)


def _prepare_plot_data(
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    config: TrainPlotConfig,
) -> tuple[tuple[tuple[float, float], ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算均值、分位数，并用 NaN 断开 CCD 坏段。"""
    bad_bands = normalize_bad_bands(config.bad_bands)
    normalized = normalize_spectrum(spectra, config.norm_method)
    mean_values = np.mean(normalized, axis=0)
    q10_values = np.quantile(normalized, 0.10, axis=0)
    q90_values = np.quantile(normalized, 0.90, axis=0)
    valid_mask = build_valid_mask(wavenumbers, bad_bands)
    if valid_mask is not None:
        mean_values = np.where(valid_mask, mean_values, np.nan)
        q10_values = np.where(valid_mask, q10_values, np.nan)
        q90_values = np.where(valid_mask, q90_values, np.nan)
    return bad_bands, wavenumbers, mean_values, q10_values, q90_values


def _add_bad_band_spans(axis, bad_bands, label_enable: bool = False) -> None:
    """在图中标记已排除的 CCD 坏波段。"""
    for index, (lower, upper) in enumerate(bad_bands):
        axis.axvspan(
            lower,
            upper,
            color="gray",
            alpha=0.2,
            label="CCD-affected region" if label_enable and index == 0 else None,
        )


def _save_mean_plot(
    output_path: Path,
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    title: str,
    config: TrainPlotConfig,
) -> None:
    """保存一个类别的均值及 q10-q90 区间图。"""
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    bad_bands, axis, mean_values, q10_values, q90_values = _prepare_plot_data(
        wavenumbers, spectra, config
    )
    figure, plot_axis = plt.subplots(figsize=(10, 5))
    _add_bad_band_spans(plot_axis, bad_bands, label_enable=True)
    plot_axis.plot(axis, mean_values, label=f"Mean spectrum {config.norm_method}")
    plot_axis.fill_between(axis, q10_values, q90_values, alpha=0.3)
    plot_axis.set(title=title, xlabel="Wavenumber (cm$^{-1}$)", ylabel="Normalized intensity")
    plot_axis.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    handles = [Patch(facecolor="C0", alpha=0.3, label="q10-q90 range")]
    handles.extend(plot_axis.get_legend_handles_labels()[0])
    plot_axis.legend(handles=handles)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _safe_name(parts: tuple[str, ...]) -> str:
    """将层级路径转换为可作为图像文件名的稳定名称。"""
    return "__".join(part.replace("/", "_").replace("\\", "_") for part in parts)


def _append_group(groups: dict, key: tuple[int, tuple[str, ...]], wavenumbers, spectra) -> None:
    """向一个层级类别累积同轴训练光谱。"""
    item = groups.setdefault(key, {"wn": wavenumbers, "spectra": []})
    if item["wn"].shape != wavenumbers.shape or not np.allclose(item["wn"], wavenumbers):
        raise ValueError(f"层级均值图波数轴不一致：{'/'.join(key[1])}")
    item["spectra"].append(spectra)


def _save_summary_plot(groups: dict, output_path: Path, config: TrainPlotConfig) -> None:
    """将同层类别均值错位绘制为一张概览图。"""
    from matplotlib import pyplot as plt

    if not groups:
        return
    rows = []
    for _key, item in sorted(groups.items()):
        bad_bands, axis, mean_values, _, _ = _prepare_plot_data(
            item["wn"], np.vstack(item["spectra"]), config
        )
        rows.append((_key[1][-1], axis, mean_values, bad_bands))
    finite_values = np.concatenate([values[np.isfinite(values)] for _, _, values, _ in rows])
    value_span = float(np.percentile(finite_values, 95) - np.percentile(finite_values, 5))
    offset_step = max(value_span * 1.05, 1e-6)
    figure, plot_axis = plt.subplots(figsize=(8, max(4.0, 0.62 * len(rows) + 1.4)))
    _add_bad_band_spans(plot_axis, rows[0][3])
    for index, (label, axis, values, _bad_bands) in enumerate(rows):
        offset = (len(rows) - index - 1) * offset_step
        plot_axis.plot(axis, values + offset, linewidth=1.0)
        plot_axis.text(-0.01, offset, label, transform=plot_axis.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    plot_axis.set_xlim(float(rows[0][1].min()), float(rows[0][1].max()))
    plot_axis.margins(y=0.01)
    plot_axis.tick_params(axis="y", which="both", labelleft=False, left=False)
    plot_axis.set_xlabel("Wavenumber (cm$^{-1}$)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _generate_plots(train_dir: Path, figure_dir: Path, config: TrainPlotConfig) -> tuple[int, int, int]:
    """生成叶子类别图、层级类别图和每层概览图。"""
    hierarchy_groups: dict = {}
    summary_groups: dict = {}
    leaf_count = 0
    for folder, filenames in iter_arc_dirs(train_dir):
        payload = _read_train_group(folder, filenames)
        if payload is None:
            continue
        wavenumbers, spectra = payload
        relative_dir = folder.relative_to(train_dir)
        _save_mean_plot(figure_dir / relative_dir.parent / f"{relative_dir.name}.png", wavenumbers, spectra, " - ".join(relative_dir.parts), config)
        leaf_count += 1
        parts = tuple(relative_dir.parts)
        for level_index in range(1, len(parts)):
            _append_group(hierarchy_groups, (level_index, parts[:level_index]), wavenumbers, spectra)
        for level_index in range(1, len(parts) + 1):
            _append_group(summary_groups, (level_index, parts[:level_index]), wavenumbers, spectra)
    if not leaf_count:
        raise ValueError(f"训练目录没有可绘制的光谱：{train_dir}")
    hierarchy_root = figure_dir / "_hierarchy_mean"
    for (_level_index, parts), item in sorted(hierarchy_groups.items()):
        _save_mean_plot(
            hierarchy_root / f"level_{_level_index}" / f"{_safe_name(parts)}.png",
            item["wn"],
            np.vstack(item["spectra"]),
            "/".join(parts),
            config,
        )
    summary_count = 0
    for level_index in sorted({key[0] for key in summary_groups}):
        groups = {key: item for key, item in summary_groups.items() if key[0] == level_index}
        _save_summary_plot(groups, hierarchy_root / "summary" / f"level_{level_index}.png", config)
        summary_count += 1
    return leaf_count, len(hierarchy_groups), summary_count


def plot_train(profile, base_dir: Path | str, input_config: InputConfig = InputConfig()) -> Path:
    """从已构建的 train 目录生成并发布完整训练集审图。"""
    base_path = Path(base_dir).resolve()
    train_dir = resolve_path(profile.root_train_clean, base_path)
    figure_dir = resolve_path(profile.root_train_fig, base_path)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"缺少训练目录：{train_dir}")
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{figure_dir.name}_", dir=figure_dir.parent))
    try:
        leaf_count, hierarchy_count, summary_count = _generate_plots(
            train_dir, temp_dir, build_train_plot_config(input_config)
        )
        backup_dir = None
        if figure_dir.exists():
            backup_dir = figure_dir.parent / f".{figure_dir.name}_previous"
            if backup_dir.exists():
                raise FileExistsError(f"存在待处理的审图备份：{backup_dir}")
            figure_dir.replace(backup_dir)
        try:
            temp_dir.replace(figure_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not figure_dir.exists():
                backup_dir.replace(figure_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(f"训练审图已生成：{figure_dir}")
    print(f"叶子类别图：{leaf_count}，层级均值图：{hierarchy_count}，层级概览图：{summary_count}")
    return figure_dir
