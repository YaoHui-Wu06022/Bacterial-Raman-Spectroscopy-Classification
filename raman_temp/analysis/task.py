"""分析模型任务与局部类别空间解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from raman_temp.evaluation.context import EvaluationContext, RunEntry, resolve_entry_path, resolve_run_entry


@dataclass(frozen=True)
class AnalysisTask:
    """一个 global 或 parent 子模型的分析范围。"""

    level_name: str
    parent_id: int | None
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    run_dir: str
    entry: Mapping[str, Any]
    train_indices: np.ndarray
    validation_indices: np.ndarray


def build_run_task(context: EvaluationContext, level_name: str) -> AnalysisTask:
    """从明确 run 构建唯一的分析任务。"""
    run_entry = resolve_run_entry(context)
    if run_entry.level_name != level_name:
        raise ValueError(f"run 属于 {run_entry.level_name}，不能按 {level_name} 分析")
    return _build_task(context, level_name, run_entry.parent_id, run_entry.values, run_entry.run_dir)


def build_parent_tasks(
    context: EvaluationContext,
    level_name: str,
    parent_value: str | None,
) -> list[AnalysisTask]:
    """按全部或指定父类构建 target-level 分析任务。"""
    global_entry = (context.meta.get("level_models") or {}).get(level_name)
    if isinstance(global_entry, dict) and global_entry.get("model_path"):
        run_dir = resolve_entry_path(context.experiment_dir, global_entry, "run_dir")
        return [_build_task(context, level_name, None, global_entry, run_dir)]
    parent_level = context.dataset_index.get_parent_level(level_name)
    if parent_level is None:
        raise FileNotFoundError(f"{level_name} 缺少全局模型")
    entries = (context.meta.get("parent_models") or {}).get(level_name) or {}
    selected = _resolve_parent_ids(context, level_name, parent_level, parent_value)
    tasks = []
    for parent_id in selected:
        entry = entries.get(str(parent_id))
        if not isinstance(entry, dict) or not entry.get("model_path"):
            continue
        run_dir = resolve_entry_path(context.experiment_dir, entry, "run_dir")
        tasks.append(_build_task(context, level_name, parent_id, entry, run_dir))
    if not tasks:
        raise ValueError(f"没有可分析模型：level={level_name}")
    return tasks


def _build_task(context, level_name, parent_id, entry, run_dir) -> AnalysisTask:
    """按任务类别空间筛选固定 train/val 索引。"""
    class_ids = tuple(
        range(context.dataset_index.num_classes_by_level[level_name])
        if parent_id is None
        else (int(item) for item in entry.get("child_ids") or [])
    )
    if not class_ids:
        raise ValueError(f"{level_name} 缺少可分析类别")
    level_index = context.dataset_index.head_name_to_idx[level_name]
    parent_index = None if parent_id is None else context.dataset_index.head_name_to_idx[context.dataset_index.get_parent_level(level_name)]
    def select(indices):
        labels = context.dataset_index.level_labels[indices]
        mask = np.isin(labels[:, level_index], class_ids)
        if parent_index is not None:
            mask &= labels[:, parent_index] == parent_id
        return indices[mask]
    all_names = context.dataset_index.get_class_names(level_name)
    return AnalysisTask(level_name, parent_id, class_ids, tuple(all_names[index] for index in class_ids), str(run_dir), entry, select(context.train_indices), select(context.validation_indices))


def _resolve_parent_ids(context, level_name, parent_level, parent_value):
    """解析可选父类名称或索引，默认返回全部父类。"""
    mapping = context.meta.get("parent_to_children", {}).get(level_name, {})
    available = sorted(int(item) for item in mapping)
    if parent_value is None:
        return available
    if str(parent_value).isdigit():
        parent_id = int(parent_value)
    else:
        parent_id = context.dataset_index.get_class_names(parent_level).index(str(parent_value))
    if parent_id not in available:
        raise ValueError(f"未找到可分析父类：{parent_value}")
    return [parent_id]
