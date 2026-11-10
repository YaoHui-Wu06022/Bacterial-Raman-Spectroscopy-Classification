"""评估与推理共用的模型槽位路径解析。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from ramanv2.common.naming import build_natural_key
from ramanv2.core.run_context import MODEL_FILE_NAME


LEVEL_DIR_PATTERN = re.compile(r"^level_\d+$")
_PARENT_SLOT_PATTERN = re.compile(r"^level_\d+_\d+$")


def find_model_slot_dirs(level_dir: Path | str) -> tuple[Path, ...]:
    """列出层级目录下的全局模型槽位和父类子模型槽位。"""
    root = Path(level_dir)
    if not root.is_dir():
        return ()
    slots: list[Path] = []
    if (root / MODEL_FILE_NAME).is_file():
        slots.append(root)
    slots.extend(
        path
        for path in root.iterdir()
        if path.is_dir()
        and _PARENT_SLOT_PATTERN.match(path.name)
        and (path / MODEL_FILE_NAME).is_file()
    )
    return tuple(sorted(slots, key=lambda path: build_natural_key(path.as_posix())))


def resolve_entry_path(
    experiment_dir: Path,
    entry: Mapping[str, Any],
    key: str,
) -> Path:
    """将模型条目中的相对或绝对路径解析为绝对路径。"""
    value = entry.get(key)
    if not value:
        raise FileNotFoundError(f"模型条目缺少 {key}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (experiment_dir / path).resolve()


def has_model_path(entry: Any) -> bool:
    """判断模型槽位是否已经训练并记录了模型文件。"""
    return isinstance(entry, Mapping) and bool(entry.get("model_path"))


__all__ = [
    "LEVEL_DIR_PATTERN",
    "find_model_slot_dirs",
    "has_model_path",
    "resolve_entry_path",
]
