"""验证集评估共用的实验、数据与输出位置解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ramanv2.core.config import Config, build_config
from ramanv2.core.config_snapshot import SHARED_CONFIG_NAME, read_yaml_dict
from ramanv2.core.run_context import HIERARCHY_META_FILE_NAME
from ramanv2.common.hierarchy import assert_hierarchy_compatible, load_hierarchy_meta, validate_label_schema
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.common.paths import find_ancestor_dir
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.hierarchy_spec import resolve_hierarchy_from_meta
from ramanv2.data.profiles import resolve_selection
from ramanv2.modeling.run_manifest import resolve_entry_path
from ramanv2.training.split import load_split_files


@dataclass(frozen=True)
class EvaluationContext:
    """一次评估固定使用的实验快照、数据索引与 train/val 切分。"""

    experiment_dir: Path
    source_dir: Path
    config: Config
    input_spec: InputSpec
    dataset_index: DatasetIndex
    meta: dict[str, Any]
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class RunEntry:
    """层级元数据中的一个明确模型槽位。"""

    level_name: str
    parent_id: int | None
    values: Mapping[str, Any]
    run_dir: Path


def load_evaluation_context(source_dir: Path | str) -> EvaluationContext:
    """从实验或模型槽位目录读取评估所需的不可变输入。"""
    source_path = Path(source_dir).resolve()
    experiment_dir = resolve_experiment_dir(source_path)
    shared_values = read_yaml_dict(experiment_dir / SHARED_CONFIG_NAME)
    validate_label_schema(shared_values)
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    if meta is None:
        raise FileNotFoundError(f"缺少 {HIERARCHY_META_FILE_NAME}：{experiment_dir}")
    config = build_config(shared_values)
    selection = resolve_selection(
        config.dataset.dataset_id,
        config.dataset.profile_id,
    )
    dataset_index = DatasetIndex(
        selection.train_dir,
        hierarchy=resolve_hierarchy_from_meta(meta),
    )
    assert_hierarchy_compatible(meta, dataset_index.hierarchy_metadata)
    split = load_split_files(dataset_index, experiment_dir)
    if split is None:
        raise FileNotFoundError(
            f"实验根缺少 train_split.json/val_split.json：{experiment_dir}"
        )
    train_indices, validation_indices = split
    return EvaluationContext(
        experiment_dir=experiment_dir,
        source_dir=source_path,
        config=config,
        input_spec=build_input_spec(config.input),
        dataset_index=dataset_index,
        meta=meta,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
    )


def resolve_experiment_dir(source_dir: Path | str) -> Path:
    """从实验根或其子目录向上定位共享配置快照。"""
    source_path = Path(source_dir).resolve()
    return find_ancestor_dir(source_path, SHARED_CONFIG_NAME)


def resolve_level_name(context: EvaluationContext, level_value: str) -> str:
    """校验并规范化待评估业务层级名称。"""
    return context.dataset_index.resolve_level_name(level_value)


def _iter_run_entries(experiment_dir: Path, meta: Mapping[str, Any]):
    """按层级元数据顺序产出全局与父类模型槽位条目。"""
    for level_name, values in (meta.get("level_models") or {}).items():
        if not isinstance(values, Mapping):
            continue
        run_dir = resolve_entry_path(experiment_dir, values, "run_dir")
        yield RunEntry(str(level_name), None, values, run_dir)
    for level_name, entries in (meta.get("parent_models") or {}).items():
        for parent_text, values in (entries or {}).items():
            if not isinstance(values, Mapping) or not values.get("run_dir"):
                continue
            run_dir = resolve_entry_path(experiment_dir, values, "run_dir")
            yield RunEntry(str(level_name), int(parent_text), values, run_dir)


def resolve_run_entry(context: EvaluationContext) -> RunEntry:
    """从输入目录在层级元数据中定位唯一模型 run。"""
    for entry in _iter_run_entries(context.experiment_dir, context.meta):
        if entry.run_dir == context.source_dir:
            return entry
    raise FileNotFoundError(f"hierarchy_meta.json 未记录 run：{context.source_dir}")


def resolve_slot_target(source_dir: Path | str) -> tuple[str, int | None]:
    """按实验元数据把模型槽位目录解析为层级名称与父类索引。"""
    source_path = Path(source_dir).resolve()
    experiment_dir = resolve_experiment_dir(source_path)
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    if meta is None:
        raise FileNotFoundError(f"缺少 {HIERARCHY_META_FILE_NAME}：{experiment_dir}")
    for entry in _iter_run_entries(experiment_dir, meta):
        if entry.run_dir == source_path:
            return entry.level_name, entry.parent_id
    raise FileNotFoundError(f"hierarchy_meta.json 未记录模型槽位：{source_path}")


def resolve_run_class_ids(dataset_index: Any, run_entry: Any) -> list[int]:
    """解析一个全局或父类 run 对应的全局类别标识。"""
    if run_entry.parent_id is None:
        return list(range(dataset_index.num_classes_by_level[run_entry.level_name]))
    return [int(item) for item in run_entry.values.get("child_ids") or []]
