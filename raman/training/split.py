import json
import os

import numpy as np


def _norm_relpath(path):
    """
    统一路径分隔符，便于跨平台保存和重载切分文件。
    """
    return os.path.normpath(path).replace("\\", "/")


def save_split_files(dataset, train_idx, test_idx, out_dir):
    """
    将 train/test 切分保存为相对 `dataset.root_dir` 的文件路径列表。
    """
    os.makedirs(out_dir, exist_ok=True)
    root = dataset.root_dir
    samples = dataset.samples
    train_files = [
        _norm_relpath(os.path.relpath(samples[i], root)) for i in train_idx
    ]
    test_files = [
        _norm_relpath(os.path.relpath(samples[i], root)) for i in test_idx
    ]

    with open(os.path.join(out_dir, "train_files.json"), "w", encoding="utf-8") as f:
        json.dump(train_files, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "test_files.json"), "w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2, ensure_ascii=False)


def load_split_files(dataset, split_dir):
    """
    从实验目录加载已有切分，并映射回当前数据集中的样本索引。

    若切分文件不存在，返回 `None`。
    """
    train_path = os.path.join(split_dir, "train_files.json")
    test_path = os.path.join(split_dir, "test_files.json")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    with open(train_path, "r", encoding="utf-8") as f:
        train_files = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_files = json.load(f)

    root = dataset.root_dir
    rel_to_idx = {}
    for idx, sample_path in enumerate(dataset.samples):
        rel = _norm_relpath(os.path.relpath(sample_path, root))
        rel_to_idx[rel] = idx

    def map_list(items, kind):
        mapped_idx = []
        missing = []
        for rel in items:
            rel_norm = _norm_relpath(rel)
            if rel_norm not in rel_to_idx:
                missing.append(rel)
            else:
                mapped_idx.append(rel_to_idx[rel_norm])
        if missing:
            raise FileNotFoundError(
                f"切分文件中有 {len(missing)} 个 {kind} 样本在当前数据集中找不到，"
                f"首个缺失项：{missing[0]}"
            )
        return np.array(sorted(mapped_idx))

    train_idx = map_list(train_files, "train")
    test_idx = map_list(test_files, "test")
    return train_idx, test_idx


def split_by_lowest_level_ratio(
    dataset,
    lowest_level="leaf",
    train_ratio=0.8,
    seed=42,
    min_train_samples=1,
):
    """
    按指定层级分组后做样本切分，尽量避免同组样本泄漏到不同集合。

    返回：
    - `train_indices`
    - `test_indices`
    """
    rng = np.random.RandomState(seed)
    group_to_indices = {}

    for i in range(len(dataset)):
        if "/" in str(lowest_level):
            key = dataset.get_split_key(i, lowest_level)
        elif lowest_level == "leaf":
            key = dataset.get_leaf_key(i)
        else:
            key = dataset.get_level_key(i, lowest_level)
        if key is None:
            key = dataset.get_leaf_key(i)
        group_to_indices.setdefault(key, []).append(i)

    train_idx = []
    test_idx = []

    for indices in group_to_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            train_idx.append(indices[0])
            continue

        n_train = int(len(indices) * train_ratio)
        n_train = max(min_train_samples, n_train)
        n_train = min(n_train, len(indices) - 1)

        train_idx.extend(indices[:n_train])
        test_idx.extend(indices[n_train:])

    return train_idx, test_idx


def resolve_level_order(dataset, target_level):
    """
    解析目标训练层级，并返回从顶层到该层的顺序列表。
    """
    target_level = dataset._resolve_level_name(
        target_level,
        field_name="current_train_level",
    )
    stop_idx = dataset.level_names.index(target_level) + 1
    return target_level, list(dataset.level_names[:stop_idx])


def resolve_train_split(full_dataset, config):
    """
    生成或复用训练/验证切分。
    """
    split_level = config.split_level or "leaf"
    train_idx, test_idx = split_by_lowest_level_ratio(
        full_dataset,
        lowest_level=split_level,
        train_ratio=config.train_split,
        seed=config.seed,
    )

    existing_split = load_split_files(full_dataset, config.output_dir)
    if existing_split is None:
        save_split_files(full_dataset, train_idx, test_idx, config.output_dir)
    else:
        train_idx, test_idx = existing_split

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def _normalize_filter_values(value):
    """
    把过滤值统一整理成列表形式，便于后续统一处理。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _resolve_parent_idx_by_name(dataset, parent_level_idx, parent_name):
    """
    按父类名称解析其在指定父层中的类别索引。
    """
    if parent_name is None:
        return None
    name_to_idx = dataset.label_maps_by_level[parent_level_idx]
    return name_to_idx.get(parent_name)


def resolve_train_scope(full_dataset, config, current_train_level, head_name_to_idx):
    """
    解析本次训练的父类范围。
    """
    only_parent = getattr(config, "train_only_parent", None)
    only_parent_name = getattr(config, "train_only_parent_name", None)
    if only_parent_name is not None and only_parent is None:
        parent_level = full_dataset.get_parent_level(current_train_level)
        if parent_level is None:
            raise ValueError(
                f"{current_train_level} 没有父层，不能使用 train_only_parent_name。"
            )

        parent_level_idx = head_name_to_idx[parent_level]
        resolved = _resolve_parent_idx_by_name(
            full_dataset,
            parent_level_idx,
            only_parent_name,
        )
        if resolved is None:
            raise ValueError(
                f"{parent_level} 中找不到父类名称：{only_parent_name}"
            )

        only_parent = int(resolved)
        config.train_only_parent = only_parent
        if (
            getattr(config, "train_filter_level", None) is None
            and getattr(config, "train_filter_value", None) is None
        ):
            config.train_filter_level = parent_level
            config.train_filter_value = only_parent_name

    return only_parent


def apply_train_filter(full_dataset, train_idx, test_idx, config, head_name_to_idx):
    """
    在切分完成后，按指定层级和取值过滤样本。
    """
    filter_level = getattr(config, "train_filter_level", None)
    filter_value = getattr(config, "train_filter_value", None)
    if not filter_level or filter_value is None:
        return train_idx, test_idx

    filter_level = full_dataset._resolve_level_name(filter_level)
    if filter_level not in head_name_to_idx:
        raise ValueError(
            f"未知过滤层级：{filter_level}，可选值为：{full_dataset.head_names}"
        )

    filter_level_idx = head_name_to_idx[filter_level]
    values = _normalize_filter_values(filter_value)
    desired_ids = set()

    for value in values:
        if isinstance(value, int):
            desired_ids.add(int(value))
            continue

        label_idx = full_dataset.label_maps_by_level[filter_level_idx].get(str(value))
        if label_idx is None:
            print(f"[Warn] {filter_level} 中找不到过滤值：{value}")
            continue
        desired_ids.add(int(label_idx))

    if not desired_ids:
        raise ValueError("没有解析出有效的 train_filter_value，请检查配置。")

    labels_filter = full_dataset.level_labels[:, filter_level_idx]
    mask = np.isin(labels_filter, list(desired_ids))
    train_idx = train_idx[mask[train_idx]]
    test_idx = test_idx[mask[test_idx]]

    print(
        f"[Filter] level={filter_level}, values={values} -> "
        f"Train {len(train_idx)}, Test {len(test_idx)}"
    )
    return train_idx, test_idx


def log_split_summary(full_dataset, train_idx, test_idx, stats_level, head_name_to_idx):
    """
    输出当前训练层级对应的 train/test 样本分布摘要。
    """
    stats_level_idx = head_name_to_idx[stats_level]
    labels = full_dataset.level_labels[:, stats_level_idx]

    print(
        f"[Sample-level Split] Train samples: {len(train_idx)}, "
        f"Test samples: {len(test_idx)}"
    )
    print(
        f"Train {stats_level} counts:",
        np.bincount(
            labels[train_idx][labels[train_idx] >= 0],
            minlength=full_dataset.num_classes_by_level[stats_level],
        ),
    )
    print(
        f"Test  {stats_level} counts:",
        np.bincount(
            labels[test_idx][labels[test_idx] >= 0],
            minlength=full_dataset.num_classes_by_level[stats_level],
        ),
    )


def resolve_levels_to_train(current_train_level):
    """
    当前训练入口一次只训练一个层级。
    """
    return [current_train_level]


def build_label_map_np(child_ids, num_classes):
    """
    把全局类别索引映射成父类内子模型使用的局部类别索引。
    """
    mapping = np.full(num_classes, -1, dtype=np.int64)
    for local_idx, global_idx in enumerate(child_ids):
        mapping[int(global_idx)] = int(local_idx)
    return mapping
