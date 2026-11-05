"""`.arc_data` 两列文本的基础读取与目录扫描。"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np


ARC_DATA_SUFFIX = ".arc_data"


def is_arc_data_name(name: str) -> bool:
    """判断文件名是否为光谱后缀，忽略大小写。"""
    return str(name).lower().endswith(ARC_DATA_SUFFIX)


def read_arc_data(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """读取两列文本光谱，忽略格式错误的行。"""
    wavenumbers, intensities, _ = read_raw_arc_data(path)
    return wavenumbers, intensities


def read_raw_arc_data(path: Path | str) -> tuple[np.ndarray, np.ndarray, int]:
    """读取原始两列谱并统计无法解析的非空行。"""
    return parse_raw_arc_data(Path(path).read_text(encoding="utf-8", errors="ignore"))


def parse_raw_arc_data(text: str) -> tuple[np.ndarray, np.ndarray, int]:
    """解析两列文本光谱内容并统计无法解析的非空行。"""
    wavenumbers, intensities = [], []
    malformed_lines = 0
    for line in text.splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            malformed_lines += 1
            continue
        try:
            wavenumbers.append(float(fields[0]))
            intensities.append(float(fields[1]))
        except ValueError:
            malformed_lines += 1
    return (
        np.asarray(wavenumbers, dtype=np.float64),
        np.asarray(intensities, dtype=np.float64),
        malformed_lines,
    )


def write_arc_data(path: Path | str, wavenumbers, intensities, fmt: str = "%.8f") -> None:
    """以两列文本格式写出一条光谱，并自动创建父目录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, np.column_stack([wavenumbers, intensities]), fmt=[fmt, fmt])


def iter_arc_data_paths(root: Path | str, *, recursive: bool = False) -> Iterator[Path]:
    """按磁盘顺序产出目录下的光谱文件路径；recursive 时遍历整棵子树。"""
    base = Path(root)
    if not base.is_dir():
        return
    if recursive:
        for current, directory_names, file_names in os.walk(os.fspath(base)):
            directory_names.sort()
            for name in sorted(file_names):
                if is_arc_data_name(name):
                    yield Path(current) / name
        return
    with os.scandir(base) as entries:
        for entry in entries:
            if entry.is_file() and is_arc_data_name(entry.name):
                yield Path(entry.path)


def list_arc_data_paths(
    root: Path | str,
    *,
    recursive: bool = False,
    key: Callable[[Path], object] | None = None,
) -> list[Path]:
    """返回排序后的光谱文件路径列表。"""
    return sorted(iter_arc_data_paths(root, recursive=recursive), key=key)


def count_arc_data(root: Path | str, *, recursive: bool = False) -> int:
    """统计目录下的光谱文件数量。"""
    return sum(1 for _ in iter_arc_data_paths(root, recursive=recursive))


def has_arc_data(root: Path | str, *, recursive: bool = False) -> bool:
    """判断目录下是否至少存在一条光谱。"""
    return next(iter_arc_data_paths(root, recursive=recursive), None) is not None


__all__ = [
    "ARC_DATA_SUFFIX",
    "count_arc_data",
    "has_arc_data",
    "is_arc_data_name",
    "iter_arc_data_paths",
    "list_arc_data_paths",
    "parse_raw_arc_data",
    "read_arc_data",
    "read_raw_arc_data",
    "write_arc_data",
]
