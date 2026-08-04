"""层级标签选择与分类 logits 约束。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


ROOT_TAG = "__root__"
MISSING_TAG = "__missing__"


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


def mask_logits_by_parent(
    logits: torch.Tensor,
    parent_labels: torch.Tensor | None,
    parent_to_children: Mapping[int | str, Sequence[int]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按样本父类保留可用子类 logits，并返回父类标签有效掩码。"""
    if parent_labels is None or parent_to_children is None:
        valid_mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, valid_mask

    allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
    valid_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
    for row_index, parent_id in enumerate(parent_labels.tolist()):
        if parent_id < 0:
            continue
        child_ids = parent_to_children.get(parent_id)
        if child_ids is None:
            child_ids = parent_to_children.get(str(parent_id))
        if not child_ids:
            continue
        _validate_child_indices(child_ids, parent_id, logits.size(1))
        allowed_mask[row_index, list(child_ids)] = True
        valid_mask[row_index] = True

    masked_logits = logits.masked_fill(~allowed_mask, float("-inf"))
    masked_logits[~valid_mask] = 0.0
    return masked_logits, valid_mask


def mask_logits_by_allowed(
    logits: torch.Tensor,
    allowed_indices: Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """按显式类别索引保留 logits；未指定限制时不生成掩码。"""
    if not allowed_indices:
        return logits, None
    allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
    allowed_mask[:, list(allowed_indices)] = True
    masked_logits = logits.masked_fill(~allowed_mask, float("-inf"))
    valid_mask = allowed_mask.any(dim=1)
    masked_logits[~valid_mask] = 0.0
    return masked_logits, valid_mask


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


def _validate_child_indices(child_ids: Sequence[int], parent_id: int, num_classes: int) -> None:
    """检查父类映射中的子类索引位于当前分类头输出范围内。"""
    invalid_indices = [
        child_id
        for child_id in child_ids
        if child_id < 0 or child_id >= num_classes
    ]
    if invalid_indices:
        raise ValueError(
            "parent_to_children index out of range: "
            f"parent={parent_id}, invalid={invalid_indices}, num_classes={num_classes}"
        )
