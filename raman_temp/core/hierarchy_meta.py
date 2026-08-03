"""实验层级模型元数据的构建与读写。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_hierarchy_meta(path: Path | str) -> dict[str, Any] | None:
    """读取已有层级元数据；路径不存在时返回 ``None``。"""
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def build_hierarchy_entry(
    run_dir: Path | str | None,
    model_path: Path | str | None,
    *,
    config_path: Path | str | None = None,
    resolved_config_path: Path | str | None = None,
    train_split_path: Path | str | None = None,
    val_split_path: Path | str | None = None,
    split_hash: str | None = None,
    log_path: Path | str | None = None,
    trained_at: str | None = None,
    child_ids: list[int] | None = None,
    child_names: list[str] | None = None,
    status: str = "trained",
) -> dict[str, Any]:
    """构建一个全局层或父类子模型的层级条目。"""
    entry: dict[str, Any] = {
        "run_dir": None if run_dir is None else str(run_dir),
        "model_path": None if model_path is None else str(model_path),
        "status": status,
    }
    optional_paths = {
        "config_path": config_path,
        "resolved_config_path": resolved_config_path,
        "train_split_path": train_split_path,
        "val_split_path": val_split_path,
        "log_path": log_path,
    }
    entry.update(
        {
            key: None if value is None else str(value)
            for key, value in optional_paths.items()
        }
    )
    if split_hash is not None:
        entry["split_hash"] = split_hash
    if trained_at is not None:
        entry["trained_at"] = trained_at
    if child_ids is not None:
        entry["child_ids"] = list(child_ids)
    if child_names is not None:
        entry["child_names"] = list(child_names)
    return entry


def compute_split_hash(
    train_split_path: Path | str,
    validation_split_path: Path | str,
) -> str | None:
    """计算 train/validation 切分文件共同对应的稳定哈希。"""
    paths = (Path(train_split_path), Path(validation_split_path))
    if not all(path.is_file() for path in paths):
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def build_hierarchy_meta(
    head_names: list[str],
    class_names_by_level: dict[str, list[str]],
    parent_to_children: dict[str, dict[str, list[int]]],
    parent_level_name: dict[str, str | None],
    current_train_level: str,
    level_models: dict[str, dict[str, Any]],
    parent_models: dict[str, dict[str, dict[str, Any]]],
    runs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """构建 ``hierarchy_meta.json`` 的完整可读取内容。"""
    return {
        "head_names": list(head_names),
        "level_names": list(head_names),
        "class_names_by_level": class_names_by_level,
        "parent_to_children": parent_to_children,
        "parent_level_name": parent_level_name,
        "current_train_level": current_train_level,
        "level_models": level_models,
        "parent_models": parent_models,
        "runs": runs,
    }


def merge_hierarchy_meta(
    existing_meta: dict[str, Any] | None,
    current_meta: dict[str, Any],
) -> dict[str, Any]:
    """以当前数据集结构为准，合并已有模型条目与 run 历史。"""
    if existing_meta is None:
        return current_meta
    merged_meta = dict(current_meta)
    merged_meta["level_models"] = _merge_entries(
        existing_meta.get("level_models"),
        current_meta.get("level_models"),
    )
    merged_meta["parent_models"] = _merge_parent_entries(
        existing_meta.get("parent_models"),
        current_meta.get("parent_models"),
    )
    merged_meta["runs"] = _merge_run_history(
        existing_meta.get("runs"),
        current_meta.get("runs"),
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


def _merge_run_history(
    existing_runs: Any,
    current_runs: Any,
) -> dict[str, list[dict[str, Any]]]:
    """合并每个模型槽位的 run 历史，并避免重复条目。"""
    history = {
        str(slot_name): list(entries)
        for slot_name, entries in (existing_runs or {}).items()
        if isinstance(entries, list)
    }
    for slot_name, entries in (current_runs or {}).items():
        target = history.setdefault(str(slot_name), [])
        for entry in entries:
            if entry not in target:
                target.append(entry)
    return history


def save_hierarchy_meta(path: Path | str, meta: dict[str, Any]) -> None:
    """以缩进 JSON 写入完整层级元数据文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
