"""保存每个实验实际使用的 train split 原始光谱 npz 快照。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.file_ops import create_temporary_path, publish_file
from ramanv2.common.paths import normalize_relpath
from ramanv2.core.run_context import ExperimentContext


TRAIN_ARCHIVE_NAME = "train_data.npz"
ARCHIVE_VERSION = 3


def save_or_validate_train_archive(
    experiment_context: ExperimentContext,
    dataset: Any,
    train_indices: Any,
) -> Path:
    """原子保存实验级 train split 的 npz 快照；已有快照时校验内容一致。"""
    target_path = Path(experiment_context.experiment_dir) / TRAIN_ARCHIVE_NAME
    root_dir = Path(dataset.root_dir).resolve()
    paths = _build_sample_paths(dataset, train_indices, root_dir)
    if target_path.is_file():
        _validate_archive(target_path, paths)
        return target_path

    wavenumbers, intensities = _collect_train_spectra(dataset, train_indices)
    temporary = create_temporary_path(target_path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                version=np.asarray(ARCHIVE_VERSION),
                sample_count=np.asarray(len(paths)),
                samples=np.asarray(paths, dtype=np.str_),
                wavenumbers=wavenumbers,
                intensities=intensities,
                created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
            )
        publish_file(temporary, target_path)
    finally:
        temporary.unlink(missing_ok=True)
    return target_path


def _build_sample_paths(dataset: Any, train_indices: Any, root_dir: Path) -> list[str]:
    """按 train split 顺序返回相对数据根的样本路径。"""
    return [
        normalize_relpath(
            Path(dataset.samples[int(index)]).resolve().relative_to(root_dir)
        )
        for index in train_indices
    ]


def _collect_train_spectra(
    dataset: Any,
    train_indices: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """把 train split 的强度堆叠为矩阵，并保留首条光谱的公共波数轴。"""
    indices = [int(index) for index in train_indices]
    if not indices:
        raise ValueError("训练切分为空，无法生成 train_data.npz")
    wavenumbers: np.ndarray | None = None
    rows: list[np.ndarray] = []
    for index in indices:
        values = np.asarray(dataset.get_raw_intensity(index), dtype=np.float32)
        if wavenumbers is None:
            axis, _intensities = read_arc_data(dataset.samples[index])
            wavenumbers = np.asarray(axis, dtype=np.float32)
        if values.ndim != 1 or values.size != wavenumbers.size:
            raise ValueError(
                f"训练光谱点数与首个样本不一致：{dataset.samples[index]}"
            )
        rows.append(values)
    assert wavenumbers is not None
    return wavenumbers, np.stack(rows)


def _validate_archive(path: Path, paths: list[str]) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            version = int(archive["version"])
            sample_count = int(archive["sample_count"])
            actual_paths = [str(item) for item in archive["samples"]]
            wavenumbers = archive["wavenumbers"]
            intensities = archive["intensities"]
    except (KeyError, ValueError, OSError, BadZipFile) as error:
        raise ValueError(f"训练数据快照无效：{path}") from error
    if version != ARCHIVE_VERSION:
        raise ValueError(f"训练数据快照版本不兼容：{path}")
    if sample_count != len(paths) or sorted(actual_paths) != sorted(paths):
        raise ValueError(f"训练数据快照与当前切分不一致：{path}")
    if wavenumbers.ndim != 1 or wavenumbers.size < 2:
        raise ValueError(f"训练数据快照波数轴无效：{path}")
    if intensities.shape != (len(paths), wavenumbers.size):
        raise ValueError(f"训练数据快照强度矩阵形状无效：{path}")


__all__ = ["TRAIN_ARCHIVE_NAME", "save_or_validate_train_archive"]
