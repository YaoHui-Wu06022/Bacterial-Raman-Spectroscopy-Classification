"""评估与推理共用的模型槽位路径解析。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from ramanv2.common.file_ops import read_json, write_json
from ramanv2.core.label_schema import validate_label_schema
from ramanv2.common.paths import build_natural_key
from ramanv2.experiments.snapshot import SHARED_CONFIG_NAME, resolve_experiment_dir
from ramanv2.experiments.run_context import (
    HIERARCHY_META_FILE_NAME,
    MODEL_FILE_NAME,
    TRAIN_SPLIT_FILE_NAME,
    VALIDATION_SPLIT_FILE_NAME,
)


LEVEL_DIR_PATTERN = re.compile(r"^level_\d+$")
_PARENT_SLOT_PATTERN = re.compile(r"^level_\d+_\d+$")


def find_level_dirs(experiment_dir: Path | str) -> tuple[Path, ...]:
    """按自然顺序列出实验目录中的业务层级。"""
    root = Path(experiment_dir)
    if not root.is_dir():
        return ()
    return tuple(sorted(
        (path for path in root.iterdir() if path.is_dir() and LEVEL_DIR_PATTERN.fullmatch(path.name)),
        key=lambda path: build_natural_key(path.name),
    ))


def find_experiment_dirs(experiment_parent: Path | str, *, require_complete: bool) -> tuple[Path, ...]:
    """发现已训练的实验；评估入口额外要求完整快照和切分。"""
    root = Path(experiment_parent)
    if not root.is_dir():
        return ()
    markers = (HIERARCHY_META_FILE_NAME,)
    if require_complete:
        markers += (SHARED_CONFIG_NAME, TRAIN_SPLIT_FILE_NAME, VALIDATION_SPLIT_FILE_NAME)
    return tuple(sorted(
        (
            path.resolve() for path in root.iterdir()
            if path.is_dir()
            and all((path / marker).is_file() for marker in markers)
            and any(find_model_slot_dirs(level_dir) for level_dir in find_level_dirs(path))
        ),
        key=lambda path: build_natural_key(path.name),
        reverse=True,
    ))


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


@dataclass(frozen=True)
class RunEntry:
    """层级元数据中的一个明确模型槽位。"""

    level_name: str
    parent_id: int | None
    values: Mapping[str, Any]
    run_dir: Path


def _iter_run_entries(experiment_dir: Path, meta: Mapping[str, Any]) -> Iterator[RunEntry]:
    """按层级元数据顺序产出全局与父类模型槽位条目。"""
    for level_name, values in (meta.get("level_models") or {}).items():
        if not isinstance(values, Mapping) or not values.get("run_dir"):
            continue
        run_dir = resolve_entry_path(experiment_dir, values, "run_dir")
        yield RunEntry(str(level_name), None, values, run_dir)
    for level_name, entries in (meta.get("parent_models") or {}).items():
        for parent_text, values in (entries or {}).items():
            if not isinstance(values, Mapping) or not values.get("run_dir"):
                continue
            run_dir = resolve_entry_path(experiment_dir, values, "run_dir")
            yield RunEntry(str(level_name), int(parent_text), values, run_dir)


def find_run_entry(experiment_dir: Path, meta: Mapping[str, Any], run_dir: Path | str) -> RunEntry:
    """按槽位目录定位唯一元数据条目。"""
    target = Path(run_dir).resolve()
    for entry in _iter_run_entries(experiment_dir, meta):
        if entry.run_dir == target:
            return entry
    raise FileNotFoundError(f"hierarchy_meta.json 未记录模型槽位：{target}")


def resolve_slot_target(source_dir: Path | str) -> tuple[str, int | None]:
    """读取实验清单并定位指定槽位的层级与父类索引。"""
    experiment_dir = resolve_experiment_dir(source_dir)
    meta_path = experiment_dir / HIERARCHY_META_FILE_NAME
    meta = load_hierarchy_meta(meta_path)
    if meta is None:
        raise FileNotFoundError(f"缺少实验层级清单：{meta_path}")
    entry = find_run_entry(experiment_dir, meta, source_dir)
    return entry.level_name, entry.parent_id


def entry_model_exists(experiment_dir: Path, entry: Any) -> bool:
    """校验已登记槽位的权重文件是否存在。"""
    return has_model_path(entry) and resolve_entry_path(experiment_dir, entry, "model_path").is_file()


def resolve_slot_class_ids(
    meta: Mapping[str, Any], level_name: str, parent_id: int | None, entry: Mapping[str, Any],
) -> list[int]:
    """返回模型输出位置对应的全局类别索引。"""
    if parent_id is None:
        return list(range(len((meta.get("class_names_by_level") or {}).get(level_name) or [])))
    return [int(item) for item in entry.get("child_ids") or []]


def assert_hierarchy_compatible(existing_meta: Mapping[str, Any], current_meta: Mapping[str, Any]) -> None:
    """保证补训练、评估和元数据合并不会改变分类头对应的标签顺序。"""
    validate_label_schema(existing_meta)
    for key in ("head_names", "class_names_by_level", "parent_to_children"):
        if existing_meta.get(key) != current_meta.get(key):
            raise ValueError(f"实验标签空间不一致：{key}，请使用新实验目录。")


def load_hierarchy_meta(path: Path | str) -> dict[str, Any] | None:
    """读取已有层级元数据；路径不存在时返回 ``None``。"""
    target = Path(path)
    if not target.is_file():
        return None
    meta = read_json(target)
    validate_label_schema(meta)
    return meta


def build_hierarchy_entry(
    run_dir: Path | str | None,
    model_path: Path | str | None,
    *,
    log_path: Path | str | None = None,
    trained_at: str | None = None,
    child_ids: list[int] | None = None,
) -> dict[str, Any]:
    """构建一个全局层或父类子模型的层级条目。"""
    entry: dict[str, Any] = {
        "run_dir": None if run_dir is None else str(run_dir),
        "model_path": None if model_path is None else str(model_path),
    }
    if log_path is not None:
        entry["log_path"] = str(log_path)
    if trained_at is not None:
        entry["trained_at"] = trained_at
    if child_ids is not None:
        entry["child_ids"] = list(child_ids)
    return entry


def build_hierarchy_meta(
    head_names: list[str],
    class_names_by_level: dict[str, list[str]],
    parent_to_children: dict[str, dict[str, list[int]]],
    level_models: dict[str, dict[str, Any]],
    parent_models: dict[str, dict[str, dict[str, Any]]],
    *,
    label_schema: str,
    hierarchy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 ``hierarchy_meta.json`` 的完整可读取内容。

    显式层级配置随实验保存，推理与评估无需再依赖运行时数据集目录。
    """
    return {
        "label_schema": label_schema,
        "hierarchy": dict(hierarchy) if hierarchy is not None else None,
        "head_names": list(head_names),
        "class_names_by_level": class_names_by_level,
        "parent_to_children": parent_to_children,
        "level_models": level_models,
        "parent_models": parent_models,
    }


def merge_hierarchy_meta(
    existing_meta: dict[str, Any] | None,
    current_meta: dict[str, Any],
) -> dict[str, Any]:
    """以当前数据集结构为准，合并已有模型条目。"""
    if existing_meta is None:
        return current_meta
    validate_label_schema(current_meta)
    assert_hierarchy_compatible(existing_meta, current_meta)
    merged_meta = dict(current_meta)
    merged_meta["level_models"] = _merge_entries(
        existing_meta.get("level_models"),
        current_meta.get("level_models"),
    )
    merged_meta["parent_models"] = _merge_parent_entries(
        existing_meta.get("parent_models"),
        current_meta.get("parent_models"),
    )
    return merged_meta


def _merge_entries(
    existing_entries: Any,
    current_entries: Any,
) -> dict[str, Any]:
    """将当前全局层模型条目覆盖到已有条目上。"""
    entries = dict(existing_entries or {})
    entries.update(current_entries or {})
    return entries


def _merge_parent_entries(
    existing_entries: Any,
    current_entries: Any,
) -> dict[str, dict[str, Any]]:
    """按层级和父类标识合并父类子模型条目。"""
    entries = {
        str(level_name): dict(parent_entries)
        for level_name, parent_entries in (existing_entries or {}).items()
        if isinstance(parent_entries, dict)
    }
    for level_name, parent_entries in (current_entries or {}).items():
        entries.setdefault(str(level_name), {})
        entries[str(level_name)].update(parent_entries)
    return entries


def write_hierarchy_meta(path: Path | str, meta: dict[str, Any]) -> None:
    """以缩进 JSON 写入完整层级元数据文件。"""
    write_json(path, meta)
