"""训练层级与父类 scope 的解析和预览。"""

from __future__ import annotations

from dataclasses import dataclass

from ramanv2.data.runtime.index import DatasetIndex


@dataclass(frozen=True)
class ParentPreview:
    """一个层级父类在当前数据中的可训练信息。"""

    level_name: str
    parent_level: str
    parent_id: int
    parent_name: str
    child_count: int
    trainable: bool
    child_labels: tuple[str, ...]


def collect_parent_preview(dataset_index: DatasetIndex) -> list[ParentPreview]:
    """返回所有非首层中实际存在子类的父类预览。"""
    previews: list[ParentPreview] = []
    level_names = dataset_index.level_names
    for level_index in range(1, len(level_names)):
        level_name = level_names[level_index]
        parent_level = level_names[level_index - 1]
        child_names = dataset_index.get_class_names(level_name)
        for parent_id, parent_name, child_ids in _iter_parent_children(dataset_index, level_name):
            previews.append(
                ParentPreview(
                    level_name=level_name,
                    parent_level=parent_level,
                    parent_id=parent_id,
                    parent_name=parent_name,
                    child_count=len(child_ids),
                    trainable=len(child_ids) >= 2,
                    child_labels=tuple(
                        child_names[child_id].split("/")[-1] for child_id in child_ids
                    ),
                )
            )
    return previews


def resolve_training_parent_ids(
    dataset_index: DatasetIndex,
    level_name: str,
    parent_index: int | None = None,
    global_head_enable: bool = False,
) -> list[int | None]:
    """解析本次训练的父类索引集合；None 表示全局模型。"""
    if global_head_enable:
        if parent_index is not None:
            raise ValueError("训练全局模型时 PARENT_INDEX 必须为 None")
        return [None]
    if dataset_index.get_parent_level(level_name) is None:
        if parent_index is not None:
            raise ValueError(f"{level_name} 没有父层，PARENT_INDEX 必须为 None")
        return [None]
    if parent_index is not None:
        return [int(parent_index)]
    return [
        parent_id
        for parent_id, _parent_name, child_ids in _iter_parent_children(dataset_index, level_name)
        if len(child_ids) >= 2
    ]


def _iter_parent_children(
    dataset_index: DatasetIndex,
    level_name: str,
) -> list[tuple[int, str, list[int]]]:
    """返回目标层每个父类的索引、名称和子类列表。"""
    resolved_level = dataset_index.resolve_level_name(level_name)
    parent_level = dataset_index.get_parent_level(resolved_level)
    if parent_level is None:
        return []
    parent_names = dataset_index.get_class_names(parent_level)
    children_by_parent = dataset_index.parent_to_children.get(resolved_level, {})
    return [
        (parent_id, parent_name, list(children_by_parent.get(parent_id, [])))
        for parent_id, parent_name in enumerate(parent_names)
        if children_by_parent.get(parent_id)
    ]


__all__ = ["ParentPreview", "collect_parent_preview", "resolve_training_parent_ids"]
