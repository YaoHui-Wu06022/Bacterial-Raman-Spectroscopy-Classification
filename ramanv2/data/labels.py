"""层级类别键与标签索引。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


ROOT_TAG = "__root__"


def parts_to_key(parts: Sequence[str]) -> str:
    """将目录层级片段转换为稳定的层级类别键。"""
    if not parts:
        return ROOT_TAG
    return "/".join(str(part) for part in parts)


def normalize_level_name(level: int | str) -> str:
    """将 ``1``、``level1``、``level_1`` 规范为 ``level_1``。"""
    text = str(level).strip()
    if text.startswith("level_"):
        return text
    if text.startswith("level"):
        return f"level_{text[5:].lstrip('_')}"
    if text.isdigit():
        return f"level_{text}"
    raise ValueError(f"Invalid level: {level}")


def select_level_targets(labels: torch.Tensor, level_index: int | None = None) -> torch.Tensor:
    """从多层标签矩阵选择指定层级；一维标签直接返回。"""
    if labels.ndim != 2:
        return labels
    target_index = labels.size(1) - 1 if level_index is None else level_index
    return labels[:, target_index]


def resolve_allowed_indices(
    class_names: Sequence[str],
    allowed_values: Sequence[int | str] | int | str | None,
) -> list[int]:
    """将类别名称或全局索引解析为排序后的允许索引。"""
    if not allowed_values:
        return []
    values = (
        list(allowed_values)
        if isinstance(allowed_values, (list, tuple, set))
        else [allowed_values]
    )
    index_by_name = {name: index for index, name in enumerate(class_names)}
    return sorted(
        {
            int(value)
            if isinstance(value, int)
            else index_by_name[str(value)]
            for value in values
            if isinstance(value, int) or str(value) in index_by_name
        }
    )


def select_parent_indices(
    labels: np.ndarray,
    indices: np.ndarray | list[int],
    *,
    parent_index: int,
    level_index: int,
    parent_id: int,
) -> np.ndarray:
    """筛选属于指定父类且目标层标签有效的样本索引。"""
    candidate_indices = np.asarray(indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    mask = (candidate_labels[:, parent_index] == int(parent_id)) & (
        candidate_labels[:, level_index] >= 0
    )
    return candidate_indices[mask]
