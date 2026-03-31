import os
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, recall_score
)

from .experiment import (
    load_experiment_with_train_dataset,
    resolve_head_level_name,
)
from .report import (
    format_classification_report_text,
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    write_text,
)
from raman.data import InputPreprocessor, RamanDataset
from raman.model import ResNeXt1D_Transformer
from raman.training import load_split_files, mask_logits_by_parent


@dataclass
class EvaluationContext:
    """收拢一次测试集评估所需的运行上下文。"""

    exp_dir: str
    config: object
    dataset_root: str
    eval_level: str | None
    inherit_missing_levels: bool
    eval_only_level: str | None
    eval_only_parent: int | None


def _load_hierarchy_meta(exp_dir):
    """读取层级训练元数据，并把 JSON 中的数字键恢复成整数。"""
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


@dataclass
class EvaluateOverrides:
    """统一收拢 Colab 或脚本侧传入的评估覆盖项。"""

    exp_dir: str | None = None
    eval_level: str | None = None
    inherit_missing_levels: bool | None = True
    eval_only_level: str | None = None
    eval_only_parent: int | None = None


def configure_evaluation(overrides=None):
    """按覆盖项构建测试集评估上下文。"""
    overrides = overrides or EvaluateOverrides()
    if not overrides.exp_dir:
        raise ValueError("evaluate_test_set 需要显式传入 exp_dir。")

    exp_dir, config = load_experiment_with_train_dataset(overrides.exp_dir)
    return EvaluationContext(
        exp_dir=exp_dir,
        config=config,
        dataset_root=config.dataset_root,
        eval_level=overrides.eval_level,
        inherit_missing_levels=bool(overrides.inherit_missing_levels),
        eval_only_level=overrides.eval_only_level,
        eval_only_parent=overrides.eval_only_parent,
    )


def resolve_level_order(dataset, target_level, config):
    """解析目标层级，并返回从顶层到该层的顺序列表。"""
    target_level = resolve_head_level_name(
        dataset,
        target_level,
        getattr(config, "eval_level", None)
        or getattr(config, "current_train_level", None)
        or "leaf",
    )
    if target_level not in dataset.head_names:
        raise ValueError(
            f"Unknown eval_level: {target_level}. Available: {dataset.head_names}"
        )
    stop_idx = dataset.head_names.index(target_level) + 1
    level_order = list(dataset.head_names[:stop_idx])
    return target_level, level_order


def _effective_label_name(dataset, idx, target_level):
    """返回样本在指定层级上的名称；缺失时回退到 leaf。"""
    if hasattr(dataset, "_resolve_level_name"):
        target_level = dataset._resolve_level_name(target_level)
    hier = dataset.hier_names[idx]
    return hier.get(target_level) or hier.get("leaf")


def _pred_id_to_name(dataset, level_name, pred_id):
    """将某一层的预测类别 id 映射回类别名。"""
    level_idx = dataset.head_name_to_idx[level_name]
    return dataset.inv_label_maps_by_level[level_idx].get(int(pred_id))


def _build_display_classes(dataset, eval_level, parent_models):
    """构建展示用类别顺序：可下钻时展开子类，否则保留父类。"""
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
    """找到样本在目标层级链路上最深的有效层级。"""
    level = target_level
    while True:
        level_idx = dataset.head_name_to_idx[level]
        if dataset.level_labels[idx, level_idx] >= 0:
            return level
        parent = dataset.get_parent_level(level)
        if parent is None:
            return level
        level = parent


def _build_fallback_test_split(dataset, cfg):
    """缺少切分清单时，按 split_level 重新构造测试集索引。"""
    split_level = getattr(cfg, "split_level", None) or "leaf"
    group_keys = [dataset.get_split_key(i, split_level) for i in range(len(dataset))]

    group_to_indices = defaultdict(list)
    for idx, group_key in enumerate(group_keys):
        group_to_indices[group_key].append(idx)

    rng = np.random.RandomState(seed=cfg.seed)
    test_idx = []

    for idxs in group_to_indices.values():
        idxs = list(idxs)
        rng.shuffle(idxs)

        n = len(idxs)
        if n == 1:
            continue

        n_train = int(round(n * cfg.train_split))
        n_train = max(1, min(n - 1, n_train))
        test_idx.extend(idxs[n_train:])

    return np.array(sorted(test_idx))


# 测试集评估

def evaluate_test_set(context):
    """按实验目录中的层级模型对测试集进行评估并输出结果文件。"""
    config = context.config
    exp_dir = context.exp_dir
    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Using: {device} (config.use_gpu={config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})"
    )

    dataset = RamanDataset(context.dataset_root, augment=False, config=config)

    eval_level = context.eval_only_level or context.eval_level
    eval_level, level_order = resolve_level_order(dataset, eval_level, config)
    eval_idx = dataset.head_name_to_idx[eval_level]

    result_dir = os.path.join(exp_dir, f"{eval_level}_test_result")
    os.makedirs(result_dir, exist_ok=True)
    out_csv = os.path.join(result_dir, "test_eval_results.csv")
    out_cm_png = os.path.join(result_dir, "confusion_matrix.png")
    out_cm_raw = os.path.join(result_dir, "confusion_matrix_raw.csv")
    out_report = os.path.join(result_dir, "classification_report.txt")

    labels = dataset.level_labels[:, eval_idx]
    inv_label_map = dataset.inv_label_maps_by_level[eval_idx]

    split = load_split_files(dataset, exp_dir)
    if split is not None:
        _, test_idx = split
    else:
        test_idx = _build_fallback_test_split(dataset, config)

    print(f"\n[Test Split] Test samples: {len(test_idx)}")

    meta = _load_hierarchy_meta(exp_dir) or {}
    parent_models = meta.get("parent_models", {})
    parent_to_children = meta.get("parent_to_children", dataset.parent_to_children)
    level_models_meta = meta.get("level_models", {})

    level_model_paths = {}

    def _build_parent_models_from_files(level):
        """从实验目录扫描某一层已有的 parent 子模型文件。"""
        mapping = {}
        for parent_idx, child_ids in parent_to_children.get(level, {}).items():
            mapping[int(parent_idx)] = {"model_path": None, "child_ids": list(child_ids)}

        pattern = re.compile(rf"^{re.escape(level)}_parent_(\d+)_model\.pt$")
        for name in os.listdir(exp_dir):
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
        level_model_paths[level] = os.path.join(exp_dir, model_name)

    for level in level_order:
        if os.path.exists(level_model_paths[level]):
            continue
        if parent_models.get(level):
            continue
        built = _build_parent_models_from_files(level)
        if built:
            parent_models[level] = built

    built = _build_parent_models_from_files(eval_level)
    if built:
        if eval_level not in parent_models or not parent_models.get(eval_level):
            parent_models[eval_level] = built
        else:
            parent_models[eval_level].update(built)

    if context.inherit_missing_levels:
        classes = _build_display_classes(dataset, eval_level, parent_models)
        num_classes = len(classes)
        train_name_to_idx = {name: i for i, name in enumerate(classes)}
    else:
        train_class_names_by_level = meta.get("class_names_by_level", {})
        classes = None
        if isinstance(train_class_names_by_level, dict):
            classes = train_class_names_by_level.get(eval_level)
        if not classes:
            classes = [inv_label_map[i] for i in range(dataset.num_classes_by_level[eval_level])]
        num_classes = len(classes)
        train_name_to_idx = {name: i for i, name in enumerate(classes)}

    target_parent_idx = None
    if context.eval_only_parent is not None:
        target_parent_idx = int(context.eval_only_parent)

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
        """延迟加载某一层的全局模型。"""
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
        """延迟加载某个父类对应的子模型。"""
        key = (level, parent_idx)
        if key in parent_model_cache:
            return parent_model_cache[key]

        full_path = model_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(exp_dir, model_path)

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
        """把预测结果映射到展示坐标；必要时继续下钻到 eval_level。"""
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
        """按层级级联预测，直到目标层级为止。"""
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

    preprocessor = InputPreprocessor(config, device)

    all_preds, all_labels, all_paths = [], [], []
    skipped = 0
    invalid_pred = 0

    print("\n>>> Running TEST SET evaluation\n")

    parent_level_for_eval = dataset.get_parent_level(eval_level)

    for i in test_idx:
        sample_level = eval_level
        if context.inherit_missing_levels:
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

        if context.eval_only_parent is not None and parent_level_for_eval is not None:
            parent_label = dataset.level_labels[i, dataset.head_name_to_idx[parent_level_for_eval]]
            if parent_label < 0 or int(parent_label) != int(context.eval_only_parent):
                invalid_pred += 1
                continue

        path = dataset.samples[i]
        X = preprocessor(path)

        with torch.no_grad():
            pred = predict_up_to_level(X, sample_level)

        if context.inherit_missing_levels:
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

    report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )
    report_text = format_classification_report_text(report, classes, acc)
    write_text(out_report, report_text)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    save_confusion_matrix_csv(cm, classes, out_cm_raw)
    save_confusion_matrix_figure(cm, classes, out_cm_png, show=True)

    df = pd.DataFrame({
        "path": all_paths,
        "label_true": all_labels,
        "label_pred": all_preds
    })
    df.to_csv(out_csv, index=False)

    print("All TEST SET results saved to:", result_dir)
    return result_dir


def run_evaluate_test_set(overrides=None):
    """先应用覆盖项，再执行测试集评估。"""
    context = configure_evaluation(overrides)
    return evaluate_test_set(context)
