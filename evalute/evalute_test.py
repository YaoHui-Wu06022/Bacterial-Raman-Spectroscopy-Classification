import os
import json
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, recall_score
)

from raman.config_io import load_experiment
from raman.dataset import RamanDataset
from raman.model import ResNeXt1D_Transformer
from raman.preprocess import InputPreprocessor
from raman.train_utils import mask_logits_by_parent, load_split_files

# BASE_DIR = ""
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path):
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def _load_hierarchy_meta(exp_dir):
    meta_path = os.path.join(exp_dir, "hierarchy_meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    parent_to_children_raw = meta.get("parent_to_children", {})
    parent_to_children = {}
    for level, mapping in parent_to_children_raw.items():
        parent_to_children[level] = {int(k): list(v) for k, v in mapping.items()}

    parent_models_raw = meta.get("parent_models", {})
    parent_models = {}
    for level, mapping in parent_models_raw.items():
        parent_models[level] = {}
        for k, v in mapping.items():
            entry = dict(v)
            entry["child_ids"] = [int(c) for c in entry.get("child_ids", [])]
            parent_models[level][int(k)] = entry

    meta["parent_to_children"] = parent_to_children
    meta["parent_models"] = parent_models
    meta["level_models"] = meta.get("level_models", {})
    return meta


# =====================
# 实验目录（需要包含多个层级的模型）
# =====================
EXP_DIR = "output_耐药菌/20260129_052515"
EXP_DIR = resolve_path(EXP_DIR)
config = load_experiment(EXP_DIR)

DATASET_ROOT = resolve_path(config.dataset_root)
if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(
        f"Dataset root not found: {DATASET_ROOT}. Please check config.dataset_root."
    )
config.dataset_root = DATASET_ROOT

# 手动设置评估层级
EVAL_LEVEL = "level_2"
INHERIT_MISSING_LEVELS = True  # 缺失层级时向最低级继承（便于展示）

# =====================
# 单模型评估
# - EVAL_ONLY_LEVEL: 只评估某一层级
# - EVAL_ONLY_PARENT: 只评估某个父类下的子模型（父类 id）
# =====================
# EVAL_ONLY_LEVEL = "level_3"
# EVAL_ONLY_PARENT = 2
EVAL_ONLY_LEVEL = None
EVAL_ONLY_PARENT = None

# =====================
# 输出目录
# =====================
RESULT_DIR = os.path.join(EXP_DIR, f"{EVAL_LEVEL}_test_result")
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_CSV = os.path.join(RESULT_DIR, "test_eval_results.csv")
OUT_CM_PNG = os.path.join(RESULT_DIR, "confusion_matrix.png")
OUT_CM_RAW = os.path.join(RESULT_DIR, "confusion_matrix_raw.csv")
OUT_REPORT = os.path.join(RESULT_DIR, "classification_report.txt")


# =====================
# 加载模型（逐层）
# =====================

def resolve_level_order(dataset, target_level):
    # 校验目标层级是否存在
    if target_level is None:
        target_level = (
            getattr(config, "eval_level", None)
            or getattr(config, "train_level", None)
            or "leaf"
        )
    if target_level not in dataset.head_names:
        raise ValueError(
            f"Unknown eval_level: {target_level}. Available: {dataset.head_names}"
        )
    level_order = []
    for name in dataset.head_names:
        level_order.append(name)
        if name == target_level:
            break
    return target_level, level_order


def _effective_label_name(dataset, idx, target_level):
    # 取该样本在指定层级的名称（缺失时退到 leaf）
    if hasattr(dataset, "_resolve_level_name"):
        target_level = dataset._resolve_level_name(target_level)
    hier = dataset.hier_names[idx]
    name = hier.get(target_level)
    if name is None:
        name = hier.get("leaf")
    return name


def _pred_id_to_name(dataset, level_name, pred_id):
    level_idx = dataset.head_name_to_idx[level_name]
    return dataset.inv_label_maps_by_level[level_idx].get(int(pred_id))


def _build_display_classes(dataset, eval_level, parent_models):
    # 构建“展示用”的类别：有下一层就展开，没有就保留父类
    eval_idx = dataset.head_name_to_idx[eval_level]
    parent_level = dataset.get_parent_level(eval_level)
    if parent_level is None:
        return list(dataset.class_names_by_level[eval_idx])

    parent_idx = dataset.head_name_to_idx[parent_level]
    parent_names = dataset.class_names_by_level[parent_idx]
    parent_label_map = dataset.label_maps_by_level[parent_idx]
    child_inv = dataset.inv_label_maps_by_level[eval_idx]
    children_by_parent = dataset.parent_to_children.get(eval_level, {})
    parent_entries = parent_models.get(eval_level, {})

    classes = []
    for parent_name in parent_names:
        parent_id = parent_label_map.get(parent_name)
        child_ids = children_by_parent.get(parent_id)
        entry = parent_entries.get(parent_id)
        has_model = entry is not None and entry.get("model_path") is not None

        if child_ids and (has_model or len(child_ids) == 1):
            for cid in child_ids:
                child_name = child_inv.get(int(cid))
                if child_name is not None:
                    classes.append(child_name)
        else:
            classes.append(parent_name)

    return classes


def _sample_effective_level(dataset, idx, target_level):
    # 找到该样本最深的有效层级（用于继承）
    level = target_level
    while True:
        level_idx = dataset.head_name_to_idx[level]
        if dataset.level_labels[idx, level_idx] >= 0:
            return level
        parent = dataset.get_parent_level(level)
        if parent is None:
            return level
        level = parent


# =====================
# 测试集评估
# =====================

def evaluate_test_set():
    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Using: {device} (config.use_gpu={config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})"
    )

    # 数据集
    dataset = RamanDataset(DATASET_ROOT, augment=False, config=config)

    eval_level = EVAL_ONLY_LEVEL or EVAL_LEVEL
    eval_level, level_order = resolve_level_order(dataset, eval_level)
    eval_idx = dataset.head_name_to_idx[eval_level]

    labels = dataset.level_labels[:, eval_idx]
    inv_label_map = dataset.inv_label_maps_by_level[eval_idx]

    # evaluate 阶段，优先使用保存的切分清单
    split = load_split_files(dataset, EXP_DIR)
    if split is not None:
        _, test_idx = split
    else:
        # fallback: 按 split_level 切分
        split_level = getattr(config, "split_level", None) or "leaf"
        group_keys = [
            dataset.get_split_key(i, split_level)
            for i in range(len(dataset))
        ]

        group_to_indices = defaultdict(list)
        for idx, k in enumerate(group_keys):
            group_to_indices[k].append(idx)

        rng = np.random.RandomState(seed=config.seed)

        test_idx = []
        for _, idxs in group_to_indices.items():
            idxs = list(idxs)
            rng.shuffle(idxs)

            n = len(idxs)
            n_train = int(round(n * config.train_split))
            if n == 1:
                continue

            n_train = max(1, min(n - 1, n_train))
            test_idx.extend(idxs[n_train:])

        test_idx = np.array(sorted(test_idx))

    print(f"\n[Test Split] Test samples: {len(test_idx)}")

    # 模型
    meta = _load_hierarchy_meta(EXP_DIR) or {}
    parent_models = meta.get("parent_models", {})
    parent_to_children = meta.get("parent_to_children", dataset.parent_to_children)
    level_models_meta = meta.get("level_models", {})

    level_model_paths = {}

    def _build_parent_models_from_files(level):
        mapping = {}
        for parent_idx, child_ids in parent_to_children.get(level, {}).items():
            mapping[int(parent_idx)] = {"model_path": None, "child_ids": list(child_ids)}

        pattern = re.compile(rf"^{re.escape(level)}_parent_(\d+)_model\.pt$")
        for name in os.listdir(EXP_DIR):
            m = pattern.match(name)
            if not m:
                continue
            parent_idx = int(m.group(1))
            entry = mapping.get(parent_idx, {"child_ids": []})
            entry["model_path"] = name
            mapping[parent_idx] = entry

        return mapping

    for level in level_order:
        model_name = level_models_meta.get(level, f"{level}_model.pt")
        level_model_paths[level] = os.path.join(EXP_DIR, model_name)

    for level in level_order:
        if os.path.exists(level_model_paths[level]):
            continue
        if parent_models.get(level):
            continue
        built = _build_parent_models_from_files(level)
        if built:
            parent_models[level] = built

    # 额外补全 eval_level 的 parent 模型
    built = _build_parent_models_from_files(eval_level)
    if built:
        if eval_level not in parent_models or not parent_models.get(eval_level):
            parent_models[eval_level] = built
        else:
            parent_models[eval_level].update(built)

    if INHERIT_MISSING_LEVELS:
        classes = _build_display_classes(dataset, eval_level, parent_models)
        num_classes = len(classes)
        train_name_to_idx = {name: i for i, name in enumerate(classes)}
    else:
        # 使用训练时的类顺序作为评估坐标系
        train_class_names_by_level = meta.get("class_names_by_level", {})
        classes = None
        if isinstance(train_class_names_by_level, dict):
            classes = train_class_names_by_level.get(eval_level)
        if not classes:
            classes = [inv_label_map[i] for i in range(dataset.num_classes_by_level[eval_level])]
        num_classes = len(classes)
        train_name_to_idx = {name: i for i, name in enumerate(classes)}

    # 单模型选择
    target_parent_idx = None
    if EVAL_ONLY_PARENT is not None:
        target_parent_idx = int(EVAL_ONLY_PARENT)

    if target_parent_idx is not None:
        built = _build_parent_models_from_files(eval_level)
        if eval_level not in parent_models or not parent_models.get(eval_level):
            parent_models[eval_level] = built or {}
        elif built:
            parent_models[eval_level].update(built)

        pm = parent_models.get(eval_level, {})
        entry = pm.get(target_parent_idx)
        if not entry or entry.get("model_path") is None:
            raise FileNotFoundError(
                f"No parent model for {eval_level}, parent={target_parent_idx}"
            )
        parent_models[eval_level] = {target_parent_idx: entry}

        child_ids = entry.get("child_ids", [])
        if child_ids and (entry.get("model_path") is not None or len(child_ids) == 1):
            allowed_names = [inv_label_map[i] for i in child_ids]
        else:
            parent_level_name = dataset.get_parent_level(eval_level)
            parent_inv = None
            if parent_level_name is not None:
                parent_idx = dataset.head_name_to_idx[parent_level_name]
                parent_inv = dataset.inv_label_maps_by_level[parent_idx]
            allowed_names = []
            if parent_inv is not None:
                parent_name = parent_inv.get(int(target_parent_idx))
                if parent_name is not None:
                    allowed_names = [parent_name]
        classes = [c for c in classes if c in allowed_names]
        num_classes = len(classes)
        train_name_to_idx = {name: i for i, name in enumerate(classes)}

    print(f"\n[Evaluate] level = {eval_level}, num_classes = {num_classes}")
    print("[Evaluate] Label Mapping (train order):")
    for i in range(num_classes):
        print(f"{i:2d} -> {classes[i]}")

    level_model_cache = {}
    parent_model_cache = {}

    def get_level_model(level):
        if level in level_model_cache:
            return level_model_cache[level]

        num = dataset.num_classes_by_level[level]
        model = ResNeXt1D_Transformer(num_classes=num, config=config).to(device)

        model_path = level_model_paths.get(level)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found for level '{level}': {model_path}"
            )

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        level_model_cache[level] = model
        return model

    def get_parent_model(level, parent_idx, child_ids, model_path):
        key = (level, parent_idx)
        if key in parent_model_cache:
            return parent_model_cache[key]

        full_path = model_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(EXP_DIR, model_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Parent model not found: {full_path}"
            )

        model = ResNeXt1D_Transformer(num_classes=len(child_ids), config=config).to(device)

        state = torch.load(full_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        parent_model_cache[key] = model
        return model

    def _map_pred_to_display(pred_level, pred_id, x):
        # 预测落在父层时，如果该父层有子模型，则继续下探
        if pred_id < 0:
            return None
        if pred_level == eval_level:
            return (pred_level, pred_id)

        parent_level = dataset.get_parent_level(eval_level)
        if parent_level is None or pred_level != parent_level:
            return (pred_level, pred_id)

        entry = parent_models.get(eval_level, {}).get(int(pred_id))
        if entry is None:
            return (pred_level, pred_id)

        child_ids = entry.get("child_ids", [])
        model_path = entry.get("model_path")
        if not child_ids:
            return (pred_level, pred_id)

        if model_path is None:
            if len(child_ids) == 1:
                return (eval_level, child_ids[0])
            return (pred_level, pred_id)

        logits = get_parent_model(eval_level, int(pred_id), child_ids, model_path)(x)
        pred_local = logits.argmax(1).item()
        pred_global = child_ids[pred_local]
        return (eval_level, pred_global)

    def predict_up_to_level(x, target_level):
        parent_pred = None

        for level in level_order:
            if parent_pred is None:
                logits = get_level_model(level)(x)
                pred_global = logits.argmax(1).item()
            else:
                parent_idx = int(parent_pred)
                if level in parent_models and parent_models[level]:
                    entry = parent_models[level].get(parent_idx)
                    if entry is None:
                        return -1
                    child_ids = entry.get("child_ids", [])
                    model_path = entry.get("model_path")
                    if model_path is None:
                        if len(child_ids) == 1:
                            pred_global = child_ids[0]
                        else:
                            return -1
                    else:
                        logits = get_parent_model(level, parent_idx, child_ids, model_path)(x)
                        pred_local = logits.argmax(1).item()
                        pred_global = child_ids[pred_local]
                else:
                    logits = get_level_model(level)(x)
                    if level in parent_to_children:
                        logits, valid_parent = mask_logits_by_parent(
                            logits,
                            torch.tensor([parent_pred], device=device),
                            parent_to_children[level]
                        )
                        if not valid_parent.any():
                            return -1
                    pred_global = logits.argmax(1).item()

            if level == target_level:
                return pred_global

            parent_pred = pred_global

        return -1

    # 输入预处理
    preprocessor = InputPreprocessor(config, device)

    all_preds, all_labels, all_paths = [], [], []
    skipped = 0
    invalid_pred = 0

    print("\n>>> Running TEST SET evaluation\n")

    parent_level_for_eval = dataset.get_parent_level(eval_level)

    for i in test_idx:
        sample_level = eval_level
        if INHERIT_MISSING_LEVELS:
            sample_level = _sample_effective_level(dataset, i, eval_level)
            true_name = _effective_label_name(dataset, i, sample_level)
            if true_name is None:
                skipped += 1
                continue
            if true_name not in train_name_to_idx:
                skipped += 1
                continue
            true_label_mapped = train_name_to_idx[true_name]
        else:
            true_label = labels[i]
            if true_label < 0:
                skipped += 1
                continue

            true_name = inv_label_map.get(int(true_label))
            if true_name not in train_name_to_idx:
                skipped += 1
                continue
            true_label_mapped = train_name_to_idx[true_name]

        if EVAL_ONLY_PARENT is not None and parent_level_for_eval is not None:
            parent_label = dataset.level_labels[i, dataset.head_name_to_idx[parent_level_for_eval]]
            if parent_label < 0 or int(parent_label) != int(EVAL_ONLY_PARENT):
                invalid_pred += 1
                continue

        path = dataset.samples[i]
        X = preprocessor(path)

        with torch.no_grad():
            pred = predict_up_to_level(X, sample_level)

        # pred 是“全局 class id”
        if INHERIT_MISSING_LEVELS:
            mapped = _map_pred_to_display(sample_level, pred, X)
            if mapped is None:
                invalid_pred += 1
                continue
            pred_level_name, pred_id = mapped
            pred_name = _pred_id_to_name(dataset, pred_level_name, pred_id)
        else:
            pred_name = inv_label_map.get(int(pred))
        if pred_name not in train_name_to_idx:
            invalid_pred += 1
            continue
        pred_mapped = train_name_to_idx[pred_name]

        all_preds.append(pred_mapped)
        all_labels.append(true_label_mapped)
        all_paths.append(path)

    if not all_labels:
        print("No valid samples for this level.")
        return

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels, all_preds,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0
    )
    macro_recall = recall_score(
        all_labels, all_preds,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0
    )

    print("\n=====================================")
    print(f" Test Set Accuracy: {acc * 100:.4f}%")
    print(f" Macro F1-score:    {macro_f1 * 100:.4f}%")
    print(f" Macro Recall:      {macro_recall * 100:.4f}%")
    print("=====================================\n")

    # 报告
    report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )

    md = [
        "| Class | Precision | Recall | F1-score | Support |",
        "|-------|-----------|--------|----------|---------|"
    ]

    for cls in classes:
        p = report[cls]["precision"] * 100
        r = report[cls]["recall"] * 100
        f1 = report[cls]["f1-score"] * 100
        sup = report[cls]["support"]
        md.append(f"| {cls} | {p:.2f} | {r:.2f} | {f1:.2f} | {sup} |")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    denom = cm.sum(axis=1, keepdims=True).astype(np.float32)
    denom[denom == 0] = 1.0
    cm_norm = cm.astype(np.float32) / denom

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT_CM_RAW)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                annot[i, j] = "0\n(0)"
            else:
                annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        annot=annot,
        fmt="",
        annot_kws={"size": 10}
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_CM_PNG, dpi=300)
    plt.show()
    plt.close()

    # 单样本明细
    df = pd.DataFrame({
        "path": all_paths,
        "label_true": all_labels,
        "label_pred": all_preds
    })
    df.to_csv(OUT_CSV, index=False)

    print("All TEST SET results saved to:", RESULT_DIR)


if __name__ == "__main__":
    evaluate_test_set()
