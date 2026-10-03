"""评估阶段共用的类别空间和父类样本筛选。"""

from __future__ import annotations

from typing import Any

import numpy as np


def select_parent_indices(
    labels: np.ndarray,
    indices: np.ndarray,
    parent_index: int,
    level_index: int,
    parent_id: int,
) -> np.ndarray:
    """筛选属于一个父类且目标层标签有效的样本索引。"""
    values = labels[indices]
    mask = (values[:, parent_index] == int(parent_id)) & (values[:, level_index] >= 0)
    return indices[mask]


def resolve_run_class_ids(dataset_index: Any, run_entry: Any) -> list[int]:
    """解析一个全局或父类 run 对应的全局类别标识。"""
    if run_entry.parent_id is None:
        return list(range(dataset_index.num_classes_by_level[run_entry.level_name]))
    return [int(item) for item in run_entry.values.get("child_ids") or []]


__all__ = ["resolve_run_class_ids", "select_parent_indices"]
