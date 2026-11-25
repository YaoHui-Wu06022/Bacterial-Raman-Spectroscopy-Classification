"""实验快照、训练数据索引和切分的共用读取入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ramanv2.core.config import Config
from ramanv2.experiments.snapshot import load_run_snapshot
from ramanv2.experiments.run_context import HIERARCHY_META_FILE_NAME
from ramanv2.experiments.run_manifest import RunEntry, assert_hierarchy_compatible, find_run_entry, load_hierarchy_meta, resolve_slot_class_ids
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.hierarchy_spec import resolve_hierarchy_from_meta
from ramanv2.data.profiles import resolve_selection
from ramanv2.experiments.split import load_split_files


@dataclass(frozen=True)
class ExperimentDataContext:
    """一次实验读取所使用的实验快照、数据索引与 train/val 切分。"""

    experiment_dir: Path
    source_dir: Path
    config: Config
    input_spec: InputSpec
    dataset_index: DatasetIndex
    meta: dict[str, Any]
    train_indices: np.ndarray
    validation_indices: np.ndarray


def load_experiment_data_context(
    source_dir: Path | str,
    train_dir: Path | str | None = None,
) -> ExperimentDataContext:
    """从实验或模型槽位目录读取评估和分析所需的不可变输入。

    ``train_dir`` 用于非注册表数据集（如 BID）显式指定训练根；
    传入时不再通过注册表解析数据集。
    """
    source_path = Path(source_dir).resolve()
    snapshot = load_run_snapshot(source_path)
    experiment_dir = snapshot.experiment_dir
    config = snapshot.config
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    if meta is None:
        raise FileNotFoundError(f"缺少 {HIERARCHY_META_FILE_NAME}：{experiment_dir}")
    if train_dir is None:
        selection = resolve_selection(
            config.dataset.dataset_id,
            config.dataset.profile_id,
        )
        resolved_train_dir = selection.train_dir
    else:
        resolved_train_dir = Path(train_dir).resolve()
    dataset_index = DatasetIndex(
        resolved_train_dir,
        hierarchy=resolve_hierarchy_from_meta(meta),
    )
    assert_hierarchy_compatible(meta, dataset_index.hierarchy_metadata)
    split = load_split_files(dataset_index, experiment_dir)
    if split is None:
        raise FileNotFoundError(
            f"实验根缺少 train_split.json/val_split.json：{experiment_dir}"
        )
    train_indices, validation_indices = split
    return ExperimentDataContext(
        experiment_dir=experiment_dir,
        source_dir=source_path,
        config=config,
        input_spec=build_input_spec(config.input),
        dataset_index=dataset_index,
        meta=meta,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
    )


def resolve_level_name(context: ExperimentDataContext, level_value: str) -> str:
    """校验并规范化待评估业务层级名称。"""
    return context.dataset_index.resolve_level_name(level_value)


def resolve_run_entry(context: ExperimentDataContext) -> RunEntry:
    """从当前实验数据上下文定位模型槽位。"""
    return find_run_entry(context.experiment_dir, context.meta, context.source_dir)


def resolve_run_class_ids(context: ExperimentDataContext, run_entry: RunEntry) -> list[int]:
    """返回所选模型槽位的全局类别索引。"""
    return resolve_slot_class_ids(
        context.meta, run_entry.level_name, run_entry.parent_id, run_entry.values
    )
