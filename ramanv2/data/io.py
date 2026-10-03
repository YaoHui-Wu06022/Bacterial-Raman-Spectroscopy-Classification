"""`.arc_data` 文本谱和目录数据集的读写工具。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ramanv2.common.arc_data import read_arc_data as _read_arc_data


def iter_arc_dirs(root_dir: Path | str):
    """递归返回包含 `.arc_data` 文件的目录及其排序后的文件名。"""
    for root, directories, filenames in os.walk(os.fspath(root_dir)):
        directories.sort()
        arc_files = sorted(name for name in filenames if name.lower().endswith(".arc_data"))
        if arc_files:
            yield Path(root), arc_files


def load_arc_intensity(path: Path | str, dtype=np.float32):
    """读取单个 `.arc_data` 文件的强度列。"""
    _wavenumbers, intensities = _read_arc_data(path)
    return np.asarray(intensities, dtype=dtype)


def iter_init_groups(input_dir: Path | str):
    """按叶子目录分组迭代目录形式的 init 数据。"""
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing init directory: {input_path}")
    for leaf_dir, filenames in iter_arc_dirs(input_path):
        samples = [
            (filename, *_read_arc_data(leaf_dir / filename))
            for filename in filenames
        ]
        yield leaf_dir.relative_to(input_path), leaf_dir.name, samples


__all__ = [
    "iter_arc_dirs",
    "iter_init_groups",
    "load_arc_intensity",
]
