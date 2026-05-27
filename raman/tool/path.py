"""跨包通用路径工具"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path, project_root: Path | str = PROJECT_ROOT):
    """把项目相对路径解析为绝对路径，绝对路径保持不变"""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root) / path).resolve()


def resolve_under_base(base_dir, path_value):
    """把相对路径解析到指定基准目录下"""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def normalize_relpath(path):
    """统一相对路径分隔符，便于跨平台保存和比较"""
    return os.path.normpath(os.fspath(path)).replace("\\", "/")


def safe_relative_to(path, parent):
    """如果 path 位于 parent 下则返回相对路径，否则返回 None"""
    try:
        return Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return None


def is_relative_to(path, parent):
    """兼容不同 Python 版本的路径包含判断"""
    return safe_relative_to(path, parent) is not None


def relpath(path, start):
    """返回使用正斜杠的相对路径字符串"""
    return normalize_relpath(os.path.relpath(path, start))


def exp_relpath(exp_dir, path):
    """把路径转成相对实验根目录的稳定字符串"""
    if path is None:
        return None
    path = Path(path)
    exp_dir = Path(exp_dir)
    if not path.is_absolute():
        path = exp_dir / path
    return relpath(path, exp_dir)


def exp_abspath(exp_dir, path):
    """把相对实验根目录的路径还原为绝对 Path"""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path(exp_dir) / path).resolve()
