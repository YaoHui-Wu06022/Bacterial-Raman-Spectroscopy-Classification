"""跨模块路径与 JSON 工具。

所有相对路径统一经 ``resolve_path`` 解析；实验元数据仍使用可迁移的相对路径。
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path, base_dir: Path | str = PROJECT_ROOT):
    """将相对路径解析到 ``base_dir``；绝对路径保持不变。"""
    if path is None:
        return None
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base_dir) / path).resolve()


def ensure_dir(path):
    """创建并返回目录；可选输出传入 ``None`` 时直接返回。"""
    if path is None:
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_relpath(path):
    """返回使用正斜杠的可迁移相对路径字符串。"""
    return os.path.normpath(os.fspath(path)).replace("\\", "/")


def safe_relative_to(path, parent):
    """若 ``path`` 位于 ``parent`` 下则返回相对路径，否则返回 ``None``。"""
    try:
        return Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return None


def is_relative_to(path, parent):
    """兼容不同 Python 版本的路径包含关系布尔判断。"""
    return safe_relative_to(path, parent) is not None


def relpath(path, start):
    """返回相对于 ``start`` 的可迁移路径。"""
    return normalize_relpath(os.path.relpath(path, start))


def exp_relpath(exp_dir, path):
    """将路径按实验根目录存为可迁移的相对路径。"""
    if path is None:
        return None
    path = Path(path)
    exp_dir = Path(exp_dir)
    return relpath(path if path.is_absolute() else exp_dir / path, exp_dir)


def exp_abspath(exp_dir, path):
    """将实验元数据中的相对路径还原为绝对 ``Path``。"""
    if path is None:
        return None
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(exp_dir) / path).resolve()


def write_json(path, payload):
    """使用 UTF-8 和稳定缩进写出 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
