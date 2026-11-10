"""训练集切分、范围解析与局部标签映射。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ramanv2.common.file_ops import read_json, write_json
from ramanv2.common.hierarchy import select_parent_indices
from ramanv2.common.paths import normalize_relpath


TRAIN_SPLIT_FILE_NAME = "train_split.json"
VAL_SPLIT_FILE_NAME = "val_split.json"


@dataclass(frozen=True)
class TrainTask:
    """单个模型训练所需的冻结层级语义与样本索引。"""

    level_name: str
    level_index: int
    head_names: tuple[str, ...]
    visible_class_ids: tuple[int, ...]
    label_map: np.ndarray | None
    train_indices: np.ndarray
    val_indices: np.ndarray
    weight_labels: np.ndarray
    parent_id: int | None
    model_tag: str

    @property
    def num_classes(self) -> int:
        """返回当前分类头可见的类别数。"""
        return len(self.visible_class_ids)


def build_global_train_task(
    dataset: Any,
    level_name: str,
    train_indices: np.ndarray | list[int],
    val_indices: np.ndarray | list[int],
) -> TrainTask:
    """构建使用指定层全部类别的全局训练任务。"""
    resolved_level = dataset.resolve_level_name(level_name)
    level_index = dataset.head_name_to_idx[resolved_level]
    task_train_indices = _freeze_indices(train_indices)
    task_val_indices = _freeze_indices(val_indices)
    visible_class_ids = tuple(range(dataset.num_classes_by_level[resolved_level]))
    return TrainTask(
        level_name=resolved_level,
        level_index=level_index,
        head_names=tuple(dataset.head_names),
        visible_class_ids=visible_class_ids,
        label_map=None,
        train_indices=task_train_indices,
        val_indices=task_val_indices,
        weight_labels=_freeze_labels(
            dataset.level_labels[task_train_indices, level_index]
        ),
        parent_id=None,
        model_tag=resolved_level,
    )


def build_parent_train_task(
    dataset: Any,
    level_name: str,
    parent_id: int,
    train_indices: np.ndarray | list[int],
    val_indices: np.ndarray | list[int],
) -> TrainTask | None:
    """构建指定父类内的子模型任务；样本从实验级切分中筛选，保证与上层同源。"""
    resolved_level = dataset.resolve_level_name(level_name)
    parent_level = dataset.get_parent_level(resolved_level)
    if parent_level is None:
        raise ValueError(f"{resolved_level} 没有父层，不能构建父类子模型任务")

    child_ids = tuple(
        dataset.parent_to_children.get(resolved_level, {}).get(int(parent_id), [])
    )
    if len(child_ids) <= 1:
        return None

    level_index = dataset.head_name_to_idx[resolved_level]
    parent_level_index = dataset.head_name_to_idx[parent_level]
    task_train_indices = _freeze_indices(
        select_parent_indices(
            dataset.level_labels,
            train_indices,
            parent_index=parent_level_index,
            level_index=level_index,
            parent_id=parent_id,
        )
    )
    task_val_indices = _freeze_indices(
        select_parent_indices(
            dataset.level_labels,
            val_indices,
            parent_index=parent_level_index,
            level_index=level_index,
            parent_id=parent_id,
        )
    )
    label_map = build_label_map_np(
        list(child_ids),
        dataset.num_classes_by_level[resolved_level],
    )
    return TrainTask(
        level_name=resolved_level,
        level_index=level_index,
        head_names=tuple(dataset.head_names),
        visible_class_ids=child_ids,
        label_map=_freeze_labels(label_map),
        train_indices=task_train_indices,
        val_indices=task_val_indices,
        weight_labels=_freeze_labels(
            label_map[dataset.level_labels[task_train_indices, level_index]]
        ),
        parent_id=int(parent_id),
        model_tag=f"{resolved_level}_{int(parent_id)}",
    )


def _freeze_indices(indices: np.ndarray | list[int]) -> np.ndarray:
    """复制并冻结任务持有的样本索引数组。"""
    frozen_indices = np.asarray(indices, dtype=np.int64).copy()
    frozen_indices.setflags(write=False)
    return frozen_indices


def _freeze_labels(labels: np.ndarray) -> np.ndarray:
    """复制并冻结任务持有的标签或局部映射数组。"""
    frozen_labels = np.asarray(labels, dtype=np.int64).copy()
    frozen_labels.setflags(write=False)
    return frozen_labels


def save_split_files(
    dataset: Any,
    train_indices: np.ndarray | list[int],
    val_indices: np.ndarray | list[int],
    split_dir: Path | str,
) -> None:
    """将 train/val 索引写为相对数据根目录的文件清单。"""
    target_dir = Path(split_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(dataset.root_dir)
    train_files = [
        normalize_relpath(Path(dataset.samples[index]).relative_to(root_dir))
        for index in train_indices
    ]
    val_files = [
        normalize_relpath(Path(dataset.samples[index]).relative_to(root_dir))
        for index in val_indices
    ]
    write_json(target_dir / TRAIN_SPLIT_FILE_NAME, train_files)
    write_json(target_dir / VAL_SPLIT_FILE_NAME, val_files)


def load_split_files(
    dataset: Any,
    split_dir: Path | str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取已保存的切分文件，并映射回当前数据集索引。"""
    target_dir = Path(split_dir)
    train_path = target_dir / TRAIN_SPLIT_FILE_NAME
    val_path = target_dir / VAL_SPLIT_FILE_NAME
    if not train_path.exists() or not val_path.exists():
        return None

    train_files = read_json(train_path)
    val_files = read_json(val_path)
    root_dir = Path(dataset.root_dir)
    index_by_path = {
        normalize_relpath(Path(path).relative_to(root_dir)): index
        for index, path in enumerate(dataset.samples)
    }
    return (
        _map_split_paths(train_files, index_by_path, "train"),
        _map_split_paths(val_files, index_by_path, "val"),
    )


def _map_split_paths(
    paths: list[str],
    index_by_path: dict[str, int],
    split_name: str,
) -> np.ndarray:
    """将一个切分清单转换为排序后的样本索引。"""
    indices: list[int] = []
    missing_paths: list[str] = []
    for path in paths:
        index = index_by_path.get(normalize_relpath(path))
        if index is None:
            missing_paths.append(path)
        else:
            indices.append(index)
    if missing_paths:
        raise FileNotFoundError(
            f"切分文件中有 {len(missing_paths)} 个 {split_name} 样本不在当前数据集内。"
            f"首个缺失项：{missing_paths[0]}"
        )
    return np.array(sorted(indices))


def split_by_lowest_level_ratio(
    dataset: Any,
    train_ratio: float = 0.8,
    seed: int = 42,
    min_train_samples: int = 1,
) -> tuple[list[int], list[int]]:
    """按最后一个真实层级分组，执行样本级 train/val 切分。"""
    lowest_level = dataset.level_names[-1]
    return _split_indices_by_sample(dataset, lowest_level, train_ratio, seed, min_train_samples)


def _split_indices_by_sample(
    dataset: Any,
    lowest_level: str,
    train_ratio: float,
    seed: int,
    min_train_samples: int,
) -> tuple[list[int], list[int]]:
    """在每个层级分组内随机切分样本。"""
    random_state = np.random.RandomState(seed)
    indices_by_group: dict[Any, list[int]] = {}
    for index in range(len(dataset)):
        group_key = _resolve_split_group_key(dataset, index, lowest_level)
        indices_by_group.setdefault(group_key, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in indices_by_group.values():
        group_indices = np.array(indices)
        random_state.shuffle(group_indices)
        if len(group_indices) == 1:
            train_indices.append(group_indices[0])
            continue
        train_count = max(min_train_samples, int(len(group_indices) * train_ratio))
        train_count = min(train_count, len(group_indices) - 1)
        train_indices.extend(group_indices[:train_count])
        val_indices.extend(group_indices[train_count:])
    return train_indices, val_indices


def _resolve_split_group_key(dataset: Any, index: int, lowest_level: str) -> Any:
    """读取样本在最后一个真实层级中的分组键。"""
    group_key = dataset.get_level_key(index, lowest_level)
    if group_key is None:
        raise ValueError(
            f"样本缺少 {lowest_level} 类别：{dataset.samples[index]}"
        )
    return group_key


def resolve_train_split(
    dataset: Any,
    train_ratio: float,
    seed: int,
    split_dir: Path | str,
    reuse_existing_enable: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """生成或复用实验级训练切分；同实验内所有模型共用同一份索引。"""
    existing_split = load_split_files(dataset, split_dir) if reuse_existing_enable else None
    if existing_split is not None:
        train_indices, val_indices = existing_split
        print(
            f"[Split] 复用已有 train_split.json/val_split.json："
            f"train={len(train_indices)}, val={len(val_indices)}"
        )
        return existing_split

    train_indices, val_indices = split_by_lowest_level_ratio(
        dataset,
        train_ratio=train_ratio,
        seed=seed,
    )
    save_split_files(dataset, train_indices, val_indices, split_dir)
    print(
        f"[Split] 新建 train_split.json/val_split.json："
        f"train={len(train_indices)}, val={len(val_indices)}"
    )
    return np.array(sorted(train_indices)), np.array(sorted(val_indices))


def resolve_parent_target(
    dataset: Any,
    level_name: str,
    parent_id: int | None,
    parent_name: str | None,
) -> int | None:
    """解析本次训练的目标父类；首层返回 None，其余层级必须指定 parent。"""
    resolved_level = dataset.resolve_level_name(level_name)
    parent_level = dataset.get_parent_level(resolved_level)
    if parent_level is None:
        if parent_id is not None or parent_name is not None:
            raise ValueError(f"{resolved_level} 没有父层，不能指定 parent")
        return None
    if parent_id is not None and parent_name is not None:
        raise ValueError("parent_id 与 parent_name 只能指定一个")
    if parent_name is not None:
        parent_level_index = dataset.head_name_to_idx[parent_level]
        parent_id = _resolve_parent_index_by_name(
            dataset,
            parent_level_index,
            parent_name,
        )
        if parent_id is None:
            raise ValueError(f"{parent_level} 中找不到父类名称：{parent_name}")
    if parent_id is None:
        raise ValueError(f"训练 {resolved_level} 必须指定 parent（父类索引或名称）")
    parent_id = int(parent_id)
    if parent_id not in dataset.parent_to_children.get(resolved_level, {}):
        raise ValueError(f"{parent_level} 中不存在父类索引：{parent_id}")
    return parent_id


def _resolve_parent_index_by_name(
    dataset: Any,
    parent_level_index: int,
    parent_name: str,
) -> int | None:
    """读取指定父层名称对应的全局类别索引。"""
    return dataset.label_maps_by_level[parent_level_index].get(parent_name)


def build_label_map_np(
    child_ids: list[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """将全局类别索引映射为父类内子模型使用的局部类别索引。"""
    mapping = np.full(num_classes, -1, dtype=np.int64)
    for local_index, global_index in enumerate(child_ids):
        mapping[int(global_index)] = local_index
    return mapping


