"""文件夹级宇宙射线去除：参数、读写与统计。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from ramanv2.data.arc_data import list_arc_data_paths, read_arc_data, write_arc_data
from ramanv2.common.paths import build_natural_key
from ramanv2.spectra.cosmic import FolderCosmicOptions, remove_folder_cosmic_rays


@dataclass(frozen=True)
class SpectrumFolder:
    """保存一个文件夹的光谱矩阵与轴信息。"""

    folder_key: str
    source_dir: Path
    paths: tuple[Path, ...]
    names: tuple[str, ...]
    wavenumbers: np.ndarray
    intensities: np.ndarray
    axis_signature: str


def build_folder_cosmic_options(payload=None) -> FolderCosmicOptions:
    """把 dataset.json 中的 cosmic_removal 段解析为参数对象。"""
    if not isinstance(payload, Mapping):
        return FolderCosmicOptions()
    values = {}
    for field in fields(FolderCosmicOptions):
        if field.name not in payload:
            continue
        raw = payload[field.name]
        default = field.default
        try:
            if isinstance(default, bool):
                values[field.name] = bool(raw)
            elif isinstance(default, int):
                values[field.name] = int(raw)
            elif isinstance(default, float):
                values[field.name] = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"cosmic_removal.{field.name} 无效：{raw}") from error
    options = FolderCosmicOptions(**values)
    _validate_options(options)
    return options


def build_folder_cosmic_payload(options: FolderCosmicOptions) -> dict:
    """把参数对象序列化回 dataset.json 可用结构。"""
    return asdict(options)


def _validate_options(options: FolderCosmicOptions) -> None:
    """校验外部配置的边界，避免窗口或阈值退化。"""
    if options.reference_smooth_window < 3:
        raise ValueError("cosmic_removal.reference_smooth_window 必须至少为 3")
    if options.local_window < 3:
        raise ValueError("cosmic_removal.local_window 必须至少为 3")
    if options.max_iterations < 1:
        raise ValueError("cosmic_removal.max_iterations 必须至少为 1")
    if options.min_spectra < 2:
        raise ValueError("cosmic_removal.min_spectra 必须至少为 2")


def read_spectrum_folder(folder_path: Path | str, folder_key: str | None = None) -> SpectrumFolder:
    """读取一个文件夹并校验长度、有限值和波数轴一致性。"""
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"文件夹不存在：{folder}")
    paths = list_arc_data_paths(
        folder,
        key=lambda path: build_natural_key(path.name),
    )
    if not paths:
        raise ValueError(f"未找到 .arc_data：{folder}")
    nested_paths = list_arc_data_paths(folder, recursive=True)
    if len(nested_paths) != len(paths):
        raise ValueError(f"{folder.name}：源文件夹包含子目录光谱，请按单个文件夹入库")
    names = tuple(path.name for path in paths)
    wavenumbers = []
    intensities = []
    for path in paths:
        wavenumbers_row, intensities_row = read_arc_data(path)
        if not wavenumbers_row.size or wavenumbers_row.size != intensities_row.size:
            raise ValueError(f"无效光谱：{path}")
        if not np.isfinite(wavenumbers_row).all() or not np.isfinite(intensities_row).all():
            raise ValueError(f"非有限光谱：{path}")
        wavenumbers.append(wavenumbers_row)
        intensities.append(intensities_row)
    wavenumbers_array = np.vstack(wavenumbers)
    intensities_array = np.vstack(intensities)
    reference_axis = wavenumbers_array[0]
    for start in range(1, wavenumbers_array.shape[0]):
        if wavenumbers_array[start].size != reference_axis.size:
            raise ValueError(f"{folder.name}：波数点长度不一致")
        if not np.allclose(wavenumbers_array[start], reference_axis, rtol=0.0, atol=1e-9):
            raise ValueError(f"{folder.name}：波数轴不一致")
    return SpectrumFolder(
        folder_key=folder_key or folder.name,
        source_dir=folder,
        paths=tuple(paths),
        names=names,
        wavenumbers=wavenumbers_array,
        intensities=intensities_array,
        axis_signature=f"{reference_axis.size}:{reference_axis[0]:.6f}:{reference_axis[-1]:.6f}",
    )


def clean_spectrum_folder(
    folder: SpectrumFolder,
    options: FolderCosmicOptions,
    valid_mask=None,
) -> tuple[np.ndarray, dict]:
    """执行文件夹级去尖峰并返回清洗矩阵与 JSON-ready 统计。"""
    result = remove_folder_cosmic_rays(folder.intensities, options, valid_mask)
    removed_points = int(sum(stat.removed_points for stat in result.stats))
    removed_runs = int(sum(stat.removed_runs for stat in result.stats))
    plateau_runs = int(sum(stat.plateau_runs for stat in result.stats))
    removed_spectra = int(sum(1 for stat in result.stats if stat.removed_points > 0))
    max_z = float(max((stat.max_z for stat in result.stats), default=0.0))
    file_rows = []
    for name, stat in zip(folder.names, result.stats):
        if not stat.removed_points and not stat.edge_skipped_runs:
            continue
        file_rows.append({
            "name": name,
            "removed_points": int(stat.removed_points),
            "removed_runs": int(stat.removed_runs),
            "plateau_runs": int(stat.plateau_runs),
            "max_z": float(stat.max_z),
            "edge_skipped_runs": int(stat.edge_skipped_runs),
        })
    reference = result.reference
    report = {
        "folder_key": folder.folder_key,
        "source_dir": str(folder.source_dir),
        "status": result.status,
        "spectrum_count": int(result.spectrum_count),
        "points_per_spectrum": int(folder.intensities.shape[1]),
        "axis_signature": folder.axis_signature,
        "removed_points": removed_points,
        "removed_runs": removed_runs,
        "plateau_runs": plateau_runs,
        "removed_spectra": removed_spectra,
        "max_z": max_z,
        "guard_peak_count": int(reference.guard_mask.sum()) if reference is not None else 0,
        "guard_threshold": float(reference.guard_threshold) if reference is not None else 0.0,
        "small_folder": folder.intensities.shape[0] < options.small_folder_warning,
        "bead": False,
        "options": build_folder_cosmic_payload(options),
        "files": file_rows,
    }
    return result.cleaned, report


def build_bead_folder_report(folder: SpectrumFolder) -> dict:
    """构建小球批次报告：只去文件头，不做宇宙射线去除。"""
    return {
        "folder_key": folder.folder_key,
        "source_dir": str(folder.source_dir),
        "status": "bead_header_only",
        "spectrum_count": int(folder.intensities.shape[0]),
        "points_per_spectrum": int(folder.intensities.shape[1]),
        "axis_signature": folder.axis_signature,
        "removed_points": 0,
        "removed_runs": 0,
        "plateau_runs": 0,
        "removed_spectra": 0,
        "max_z": 0.0,
        "guard_peak_count": 0,
        "guard_threshold": 0.0,
        "small_folder": False,
        "bead": True,
        "options": None,
        "files": [],
    }


def write_spectrum_folder(
    folder: SpectrumFolder,
    cleaned: np.ndarray,
    target_folder: Path | str,
    fmt: str = "%.17g",
) -> int:
    """写出无文件头的两列光谱；raw_data 的文件头不进入 cosmic_data。"""
    target = Path(target_folder)
    values = np.asarray(cleaned, dtype=np.float64)
    if values.shape != folder.intensities.shape:
        raise ValueError(
            f"清洗矩阵形状不一致：{values.shape} != {folder.intensities.shape}"
        )
    for index, name in enumerate(folder.names):
        write_arc_data(target / name, folder.wavenumbers[index], values[index], fmt=fmt)
    return len(folder.paths)