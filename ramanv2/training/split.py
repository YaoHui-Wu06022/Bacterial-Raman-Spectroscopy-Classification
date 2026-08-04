"""训练集切分、范围解析与局部标签映射。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ramanv2.common.naming import parse_source_prefix
from ramanv2.core.paths import normalize_relpath


DEFAULT_SPLIT_LEVEL = "leaf"
TRAIN_SPLIT_FILE_NAME = "train_split.json"
VAL_SPLIT_FILE_NAME = "val_split.json"
TRANSFERRED_SOURCE_SUFFIX = "t"


@dataclass(frozen=True)
class TrainScope:
    """一次训练使用的父类与样本筛选约束。"""

    only_parent: int | None
    filter_level: str | None
    filter_values: tuple[Any, ...] | None


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
    """构建指定父类内的子模型任务；单子类时返回 ``None``。"""
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
    task_train_indices = _select_parent_indices(
        dataset.level_labels,
        train_indices,
        level_index,
        parent_level_index,
        parent_id,
    )
    task_val_indices = _select_parent_indices(
        dataset.level_labels,
        val_indices,
        level_index,
        parent_level_index,
        parent_id,
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


def _select_parent_indices(
    labels: np.ndarray,
    indices: np.ndarray | list[int],
    level_index: int,
    parent_level_index: int,
    parent_id: int,
) -> np.ndarray:
    """筛选属于指定父类且当前层标签有效的样本索引。"""
    candidate_indices = np.asarray(indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]
    mask = (candidate_labels[:, parent_level_index] == int(parent_id)) & (
        candidate_labels[:, level_index] >= 0
    )
    return _freeze_indices(candidate_indices[mask])


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
    (target_dir / TRAIN_SPLIT_FILE_NAME).write_text(
        json.dumps(train_files, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (target_dir / VAL_SPLIT_FILE_NAME).write_text(
        json.dumps(val_files, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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

    train_files = json.loads(train_path.read_text(encoding="utf-8"))
    val_files = json.loads(val_path.read_text(encoding="utf-8"))
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
    lowest_level: str = DEFAULT_SPLIT_LEVEL,
    train_ratio: float = 0.8,
    seed: int = 42,
    min_train_samples: int = 1,
    split_by_source_prefix_enable: bool = False,
) -> tuple[list[int], list[int]]:
    """按指定层级分组，执行样本级或来源前缀的 train/val 切分。"""
    if split_by_source_prefix_enable:
        return _split_indices_by_source_prefix(dataset, lowest_level, train_ratio, seed)
    return _split_indices_by_sample(dataset, lowest_level, train_ratio, seed, min_train_samples)


def _split_indices_by_sample(
    dataset: Any,
    lowest_level: str,
    train_ratio: float,
    seed: int,
    min_train_samples: int,
) -> tuple[list[int], list[int]]:
    """在每个层级与来源类型分组内随机切分样本。"""
    random_state = np.random.RandomState(seed)
    indices_by_group: dict[tuple[Any, bool], list[int]] = {}
    for index in range(len(dataset)):
        group_key = (
            _resolve_split_group_key(dataset, index, lowest_level),
            _is_transferred_sample(dataset.samples[index]),
        )
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


def _split_indices_by_source_prefix(
    dataset: Any,
    lowest_level: str,
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """以来源前缀为不可拆分分组，避免同源谱同时进入 train 和 val。"""
    random_state = np.random.RandomState(seed)
    prefix_groups_by_bucket: dict[tuple[Any, bool], dict[str, list[int]]] = {}
    for index in range(len(dataset)):
        source_prefix = parse_source_prefix(dataset.samples[index])
        bucket_key = (
            _resolve_split_group_key(dataset, index, lowest_level),
            str(source_prefix).lower().endswith(TRANSFERRED_SOURCE_SUFFIX),
        )
        prefix_groups = prefix_groups_by_bucket.setdefault(bucket_key, {})
        prefix_groups.setdefault(source_prefix, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for (level_key, is_transferred), prefix_groups in prefix_groups_by_bucket.items():
        groups = [
            (prefix, np.array(indices, dtype=np.int64))
            for prefix, indices in prefix_groups.items()
        ]
        random_state.shuffle(groups)
        if len(groups) == 1:
            prefix, indices = groups[0]
            train_indices.extend(indices.tolist())
            source_kind = "*t" if is_transferred else "non-*t"
            print(
                "[Warn] 来源前缀切分："
                f"{level_key!r}/{source_kind} 只有一个来源前缀 {prefix!r}。"
                "全部归入 train。"
            )
            continue

        group_train: list[tuple[str, np.ndarray]] = []
        group_val: list[tuple[str, np.ndarray]] = []
        target_train = sum(len(indices) for _, indices in groups) * float(train_ratio)
        current_train = 0
        for group_index, group in enumerate(groups):
            if not group_train:
                group_train.append(group)
                current_train += len(group[1])
                continue
            if group_index == len(groups) - 1 and not group_val:
                group_val.append(group)
                continue
            add_error = abs(target_train - (current_train + len(group[1])))
            current_error = abs(target_train - current_train)
            if add_error <= current_error:
                group_train.append(group)
                current_train += len(group[1])
            else:
                group_val.append(group)
        if not group_val:
            group_val.append(group_train.pop())
        for _, indices in group_train:
            train_indices.extend(indices.tolist())
        for _, indices in group_val:
            val_indices.extend(indices.tolist())
    return train_indices, val_indices


def _resolve_split_group_key(dataset: Any, index: int, lowest_level: str) -> Any:
    """读取样本在指定 split 层级中的分组键，并在缺失时回退到 leaf。"""
    if "/" in str(lowest_level):
        group_key = dataset.get_split_key(index, lowest_level)
    elif lowest_level == DEFAULT_SPLIT_LEVEL:
        group_key = dataset.get_leaf_key(index)
    else:
        group_key = dataset.get_level_key(index, lowest_level)
    return dataset.get_leaf_key(index) if group_key is None else group_key


def _is_transferred_sample(sample_path: Path | str) -> bool:
    """判断样本是否来自名称以 ``t`` 结尾的迁移来源前缀。"""
    return parse_source_prefix(sample_path).lower().endswith(TRANSFERRED_SOURCE_SUFFIX)


def resolve_train_split(
    dataset: Any,
    train_ratio: float,
    seed: int,
    split_dir: Path | str,
    reuse_existing_enable: bool = True,
    split_by_source_prefix_enable: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """生成或复用训练切分，返回排序后的索引数组。"""
    existing_split = load_split_files(dataset, split_dir) if reuse_existing_enable else None
    if existing_split is not None:
        return existing_split

    train_indices, val_indices = split_by_lowest_level_ratio(
        dataset,
        train_ratio=train_ratio,
        seed=seed,
        split_by_source_prefix_enable=split_by_source_prefix_enable,
    )
    save_split_files(dataset, train_indices, val_indices, split_dir)
    return np.array(sorted(train_indices)), np.array(sorted(val_indices))


def build_train_scope(
    dataset: Any,
    current_train_level: str,
    head_name_to_index: dict[str, int],
    *,
    only_parent: int | None = None,
    only_parent_name: str | None = None,
    filter_level: str | None = None,
    filter_value: object | None = None,
) -> TrainScope:
    """从训练请求解析父类与筛选约束。"""
    if only_parent_name is not None and only_parent is None:
        parent_level = dataset.get_parent_level(current_train_level)
        if parent_level is None:
            raise ValueError(f"{current_train_level} 没有父层，不能使用 train_only_parent_name")
        parent_level_index = head_name_to_index[parent_level]
        only_parent = _resolve_parent_index_by_name(dataset, parent_level_index, only_parent_name)
        if only_parent is None:
            raise ValueError(f"{parent_level} 中找不到父类名称：{only_parent_name}")
        if filter_level is None and filter_value is None:
            filter_level = parent_level
            filter_value = only_parent_name

    filter_values = _normalize_filter_values(filter_value)
    return TrainScope(
        only_parent=None if only_parent is None else int(only_parent),
        filter_level=filter_level,
        filter_values=filter_values,
    )


def apply_train_filter(
    dataset: Any,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_scope: TrainScope,
    head_name_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """根据冻结的训练范围过滤 train/val 样本。"""
    if not train_scope.filter_level or train_scope.filter_values is None:
        return train_indices, val_indices

    filter_level = dataset.resolve_level_name(train_scope.filter_level)
    if filter_level not in head_name_to_index:
        raise ValueError(f"未知过滤层级：{filter_level}；可选值：{dataset.head_names}")
    level_index = head_name_to_index[filter_level]
    desired_ids = _resolve_filter_ids(dataset, level_index, train_scope.filter_values)
    labels = dataset.level_labels[:, level_index]
    mask = np.isin(labels, list(desired_ids))
    filtered_train = train_indices[mask[train_indices]]
    filtered_val = val_indices[mask[val_indices]]
    print(
        f"[Filter] level={filter_level}, values={list(train_scope.filter_values)} -> "
        f"Train {len(filtered_train)}, Val {len(filtered_val)}"
    )
    return filtered_train, filtered_val


def _normalize_filter_values(value: Any) -> tuple[Any, ...] | None:
    """将单个或多个过滤值统一为不可变元组。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _resolve_parent_index_by_name(
    dataset: Any,
    parent_level_index: int,
    parent_name: str,
) -> int | None:
    """读取指定父层名称对应的全局类别索引。"""
    return dataset.label_maps_by_level[parent_level_index].get(parent_name)


def _resolve_filter_ids(
    dataset: Any,
    level_index: int,
    filter_values: tuple[Any, ...],
) -> set[int]:
    """将过滤名称或索引转换为当前层级的全局类别索引集合。"""
    desired_ids: set[int] = set()
    for value in filter_values:
        if isinstance(value, int):
            desired_ids.add(value)
            continue
        label_index = dataset.label_maps_by_level[level_index].get(str(value))
        if label_index is None:
            print(f"[Warn] 当前层级中找不到过滤值：{value}")
            continue
        desired_ids.add(int(label_index))
    if not desired_ids:
        raise ValueError("没有解析出有效的 train_filter_value，请检查配。")
    return desired_ids


def build_label_map_np(
    child_ids: list[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """将全局类别索引映射为父类内子模型使用的局部类别索引。"""
    mapping = np.full(num_classes, -1, dtype=np.int64)
    for local_index, global_index in enumerate(child_ids):
        mapping[int(global_index)] = local_index
    return mapping


def log_split_summary(
    dataset: Any,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    stats_level: str,
    head_name_to_index: dict[str, int],
) -> None:
    """输出指定层级在 train/val 中的样本数量摘要。"""
    level_index = head_name_to_index[stats_level]
    labels = dataset.level_labels[:, level_index]
    print(
        f"[Sample-level Split] Train samples: {len(train_indices)}, "
        f"Val samples: {len(val_indices)}"
    )
    print(
        f"Train {stats_level} counts:",
        np.bincount(
            labels[train_indices][labels[train_indices] >= 0],
            minlength=dataset.num_classes_by_level[stats_level],
        ),
    )
    print(
        f"Val   {stats_level} counts:",
        np.bincount(
            labels[val_indices][labels[val_indices] >= 0],
            minlength=dataset.num_classes_by_level[stats_level],
        ),
    )
