"""实验切分文件的读写与样本路径映射。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ramanv2.common.file_ops import read_json, write_json
from ramanv2.common.paths import normalize_relpath
from ramanv2.experiments.run_context import TRAIN_SPLIT_FILE_NAME, VALIDATION_SPLIT_FILE_NAME


def save_split_files(
    dataset: Any,
    train_indices: np.ndarray | list[int],
    val_indices: np.ndarray | list[int],
    split_dir: Path | str,
) -> None:
    """将 train/val 索引写为相对数据根目录的文件清单。"""
    target_dir = Path(split_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(dataset.root_dir)
    train_files = [
        normalize_relpath(Path(dataset.samples[index]).relative_to(root_dir))
        for index in train_indices
    ]
    val_files = [
        normalize_relpath(Path(dataset.samples[index]).relative_to(root_dir))
        for index in val_indices
    ]
    write_json(target_dir / TRAIN_SPLIT_FILE_NAME, train_files)
    write_json(target_dir / VALIDATION_SPLIT_FILE_NAME, val_files)


def load_split_files(
    dataset: Any,
    split_dir: Path | str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取已保存的切分文件，并映射回当前数据集索引。"""
    target_dir = Path(split_dir)
    train_path = target_dir / TRAIN_SPLIT_FILE_NAME
    val_path = target_dir / VALIDATION_SPLIT_FILE_NAME
    if not train_path.exists() or not val_path.exists():
        return None

    train_files = read_json(train_path)
    val_files = read_json(val_path)
    root_dir = Path(dataset.root_dir)
    index_by_path = {
        normalize_relpath(Path(path).relative_to(root_dir)): index
        for index, path in enumerate(dataset.samples)
    }
    return (
        _map_split_paths(train_files, index_by_path, "train"),
        _map_split_paths(val_files, index_by_path, "val"),
    )


def _map_split_paths(
    paths: list[str],
    index_by_path: dict[str, int],
    split_name: str,
) -> np.ndarray:
    """将一个切分清单转换为排序后的样本索引。"""
    indices: list[int] = []
    missing_paths: list[str] = []
    for path in paths:
        index = index_by_path.get(normalize_relpath(path))
        if index is None:
            missing_paths.append(path)
        else:
            indices.append(index)
    if missing_paths:
        raise FileNotFoundError(
            f"切分文件中有 {len(missing_paths)} 个 {split_name} 样本不在当前数据集内。"
            f"首个缺失项：{missing_paths[0]}"
        )
    return np.array(sorted(indices))
