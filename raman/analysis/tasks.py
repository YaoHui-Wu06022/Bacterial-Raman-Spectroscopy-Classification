import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from raman.training import build_label_map_np


class LabelMapDataset(Dataset):
    # 将某一层级的标签映射为局部索引（用于父类内子模型分析）
    def __init__(self, base_dataset, level_idx, label_map_np):
        self.base_dataset = base_dataset
        self.level_idx = level_idx
        self.label_map_np = label_map_np

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, labels, hier = self.base_dataset[idx]
        labels = np.array(labels, copy=True)
        if labels[self.level_idx] >= 0:
            labels[self.level_idx] = self.label_map_np[labels[self.level_idx]]
        else:
            labels[self.level_idx] = -1
        return x, labels, hier

def normalize_parent_idx(parent_idx):
    """统一 parent_idx 输入格式（None/int/'all'）"""
    if parent_idx is None:
        return None
    if isinstance(parent_idx, str):
        text = parent_idx.strip().lower()
        if text == "all":
            return "all"
        if text.isdigit():
            return int(text)
        raise ValueError(
            f"Unknown PARENT_IDX value: {parent_idx}. Use int or 'all'."
        )
    return int(parent_idx)

def build_analysis_tasks(
    exp_dir,
    analysis_level,
    head_index,
    full_dataset,
    level_models,
    parent_models,
    parent_idx_setting,
):
    """解析需要分析的模型任务（单模型或按 parent 拆分）"""
    parent_idx_setting = normalize_parent_idx(parent_idx_setting)
    parent_entries = parent_models.get(analysis_level, {})
    tasks = []
    auto_all = False

    if parent_idx_setting is None:
        model_path = level_models.get(analysis_level)
        if model_path and os.path.exists(model_path):
            num_classes = full_dataset.num_classes_by_level[analysis_level]
            class_names = full_dataset.class_names_by_level[head_index]
            tasks.append(
                {
                    "parent_idx": None,
                    "model_path": model_path,
                    "child_ids": None,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "tag": analysis_level,
                }
            )
            return tasks, auto_all

        if parent_entries:
            parent_idx_setting = "all"
            auto_all = True
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}. If you trained per-parent, set PARENT_IDX."
            )

    if parent_idx_setting == "all":
        if not parent_entries:
            raise ValueError(
                f"No parent model entries for level={analysis_level}."
            )
        for parent_idx in sorted(parent_entries.keys()):
            entry = parent_entries[parent_idx]
            if entry.get("model_path") is None:
                continue
            model_path = entry["model_path"]
            if not os.path.isabs(model_path):
                model_path = os.path.join(exp_dir, model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Parent model not found: {model_path}")
            child_ids = entry.get("child_ids", [])
            class_names = [
                full_dataset.class_names_by_level[head_index][cid]
                for cid in child_ids
            ]
            tasks.append(
                {
                    "parent_idx": parent_idx,
                    "model_path": model_path,
                    "child_ids": child_ids,
                    "num_classes": len(child_ids),
                    "class_names": class_names,
                    "tag": f"{analysis_level}_parent_{parent_idx}",
                }
            )

        if not tasks:
            raise ValueError(
                f"No parent models to analyze for level={analysis_level}."
            )
        return tasks, auto_all

    parent_idx = int(parent_idx_setting)
    entry = parent_entries.get(parent_idx)
    if entry is None:
        raise ValueError(
            f"No parent model entry for level={analysis_level}, parent={parent_idx}"
        )
    if entry.get("model_path") is None:
        raise ValueError(
            f"Parent {parent_idx} has only one child; no model to analyze."
        )

    model_path = entry["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(exp_dir, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Parent model not found: {model_path}")

    child_ids = entry.get("child_ids", [])
    class_names = [
        full_dataset.class_names_by_level[head_index][cid]
        for cid in child_ids
    ]
    tasks.append(
        {
            "parent_idx": parent_idx,
            "model_path": model_path,
            "child_ids": child_ids,
            "num_classes": len(child_ids),
            "class_names": class_names,
            "tag": f"{analysis_level}_parent_{parent_idx}",
        }
    )
    return tasks, auto_all

def build_task_loaders(
    task,
    config,
    full_dataset,
    analysis_level,
    head_index,
    train_idx_all,
    val_idx_all,
    base_train_dataset,
    base_val_dataset,
):
    """按 parent 过滤样本并构建 DataLoader"""
    parent_idx = task["parent_idx"]
    child_ids = task["child_ids"]
    inherit_missing = getattr(config, "inherit_missing_levels", False)

    train_idx = train_idx_all
    val_idx = val_idx_all

    if parent_idx is not None:
        parent_level = full_dataset.get_parent_level(analysis_level)
        if parent_level is None:
            raise ValueError("Top level has no parent; cannot use PARENT_IDX.")
        parent_level_idx = full_dataset.head_name_to_idx[parent_level]

        labels_train = full_dataset.level_labels[train_idx]
        labels_val = full_dataset.level_labels[val_idx]

        train_mask = (labels_train[:, parent_level_idx] == parent_idx)
        val_mask = (labels_val[:, parent_level_idx] == parent_idx)
        if not inherit_missing:
            train_mask = train_mask & (labels_train[:, head_index] >= 0)
            val_mask = val_mask & (labels_val[:, head_index] >= 0)

        train_idx = train_idx[train_mask]
        val_idx = val_idx[val_mask]

        label_map_np = build_label_map_np(
            child_ids,
            full_dataset.num_classes_by_level[analysis_level]
        )

        train_dataset = LabelMapDataset(base_train_dataset, head_index, label_map_np)
        val_dataset = LabelMapDataset(base_val_dataset, head_index, label_map_np)
    else:
        train_dataset = base_train_dataset
        val_dataset = base_val_dataset

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False
    )

    if len(val_subset) == 0:
        val_loader = train_loader

    return train_loader, val_loader, train_subset, val_subset

