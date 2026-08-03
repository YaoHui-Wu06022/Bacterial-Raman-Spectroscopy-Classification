"""项目根目录与稳定资源路径。

所有路径均相对于仓库根目录解析，避免调用命令时所在目录影响数据定位。
"""

from __future__ import annotations

import os
from pathlib import Path


# ``raman/core/paths.py`` 的上两级目录就是仓库根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
_STANFORD_REFERENCE_WAVENUMBERS = (
    DATASET_ROOT / "Stanforddataset" / "reference_wavenumbers.npy"
)


def resolve_path(path: Path | str | None, base_dir: Path | str | None = None) -> Path | None:
    """将相对路径解析到给定目录；默认基于项目根目录。"""
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    root = PROJECT_ROOT if base_dir is None else Path(base_dir)
    return (root / candidate).resolve()


def stanford_reference_wavenumbers_path() -> Path:
    """返回固定的 Stanford 共享参考波数文件位置，不检查文件是否存在。"""
    return _STANFORD_REFERENCE_WAVENUMBERS


def normalize_relpath(path: Path | str) -> str:
    """生成使用正斜杠的可迁移相对路径字符串。"""
    return os.path.normpath(os.fspath(path)).replace("\\", "/")


def safe_relative_to(path: Path | str, parent: Path | str) -> Path | None:
    """路径位于父目录时返回相对路径，否则返回 ``None``。"""
    try:
        return Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return None


def is_relative_to(path: Path | str, parent: Path | str) -> bool:
    """以布尔形式判断路径是否位于父目录内。"""
    return safe_relative_to(path, parent) is not None


def relpath(path: Path | str, start: Path | str) -> str:
    """返回相对于 ``start`` 的可迁移路径。"""
    return normalize_relpath(os.path.relpath(path, start))
