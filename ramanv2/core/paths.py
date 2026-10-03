"""项目根目录与稳定资源路径。

所有路径均相对于仓库根目录解析，避免调用命令时所在目录影响数据定位。
"""

from __future__ import annotations

import os
from pathlib import Path


# 本模块位于 ``<项目根目录>/ramanv2/core/``，上两级目录即项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"


def resolve_path(path: Path | str | None, base_dir: Path | str | None = None) -> Path | None:
    """将相对路径解析到给定目录；默认基于项目根目录。"""
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    root = PROJECT_ROOT if base_dir is None else Path(base_dir)
    return (root / candidate).resolve()


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


def find_ancestor_dir(start_path: Path | str, marker_name: str) -> Path:
    """从路径自身向上查找包含指定标记文件或目录的祖先目录。"""
    start = Path(start_path).resolve()
    candidates = (
        (start, *start.parents)
        if start.is_dir()
        else (start.parent, *start.parent.parents)
    )
    for candidate in candidates:
        if (candidate / marker_name).exists():
            return candidate
    raise FileNotFoundError(f"无法定位包含 {marker_name} 的目录：{start}")


def relpath(path: Path | str, start: Path | str) -> str:
    """返回相对于 ``start`` 的可迁移路径。"""
    return normalize_relpath(os.path.relpath(path, start))
