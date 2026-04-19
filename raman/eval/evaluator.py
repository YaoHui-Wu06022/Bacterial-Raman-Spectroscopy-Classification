import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .common import compute_classification_metrics, run_cascade_inference, select_logits
from .experiment import (
    load_experiment_with_dataset,
    load_hierarchy_meta,
    resolve_head_level_name,
)
from .report import (
    format_classification_report_text,
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    write_text,
)
from .runtime import build_experiment_runtime
from raman.data import InputPreprocessor, RamanDataset
from raman.training import load_split_files


@dataclass
class EvaluationContext:
    """收拢一次测试集评估所需的运行上下文"""

    exp_dir: str
    config: object
    dataset_root: str
    eval_level: str | None
    inherit_missing_levels: bool
    eval_only_level: str | None
    eval_only_parent: int | None


@dataclass
class EvaluateOverrides:
    """统一收拢 Colab 或脚本侧传入的评估覆盖项"""

    exp_dir: str | None = None
    eval_level: str | None = None
    inherit_missing_levels: bool | None = True
    eval_only_level: str | None = None
    eval_only_parent: int | None = None


def configure_evaluation(overrides=None):
    """按覆盖项构建测试集评估上下文"""
    overrides = overrides or EvaluateOverrides()
    if not overrides.exp_dir:
        raise ValueError("evaluate_test_set 需要显式传入 exp_dir")

    exp_dir, config = load_experiment_with_dataset(overrides.exp_dir)
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
    """解析目标层级，并返回从顶层到该层的顺序列表"""
    target_level = resolve_head_level_name(dataset, target_level)
    if target_level not in dataset.level_names:
        raise ValueError(
            f"Unknown eval_level: {target_level}. Available: {dataset.level_names}"
        )
    stop_idx = dataset.level_names.index(target_level) + 1
    level_order = list(dataset.level_names[:stop_idx])
    return target_level, level_order


def _effective_label_name(dataset, idx, target_level):
    if hasattr(dataset, "_resolve_level_name"):
        target_level = dataset._resolve_level_name(target_level)
    hier = dataset.hier_names[idx]
    return hier.get(target_level)


def _pred_id_to_name(dataset, level_name, pred_id):
    level_idx = dataset.head_name_to_idx[level_name]
    return dataset.inv_label_maps_by_level[level_idx].get(int(pred_id))


def _build_display_classes(dataset, eval_level, parent_models):
    """构建展示用类别顺序"""
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
            for child_id in child_ids:
                child_name = child_inv.get(int(child_id))
                if child_name is not None:
                    classes.append(child_name)
        else:
            classes.append(parent_name)

    return classes


def _sample_effective_level(dataset, idx, target_level):
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
    """缺少切分清单时，按 split_level 重新构造测试集索引"""
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


def evaluate_test_set(context):
    """按实验目录中的层级模型对测试集进行评估并输出结果文件"""
    config = context.config
    exp_dir = context.exp_dir
    use_cuda = config.use_gpu and torch.cuda.is_available()
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

    meta = load_hierarchy_meta(exp_dir) or {}
    runtime = build_experiment_runtime(exp_dir, device, config=config, meta=meta)
    if not runtime.parent_to_children:
        runtime.parent_to_children = dataset.parent_to_children
    parent_to_children = runtime.parent_to_children
    runtime.build_level_model_paths(level_order)

    for level_name in level_order:
        model_path = runtime.level_model_paths.get(level_name)
        if model_path and os.path.exists(model_path):
            continue
        runtime.ensure_parent_models(level_name, parent_to_children)
    runtime.ensure_parent_models(eval_level, parent_to_children)
    parent_models = runtime.parent_models

    if context.inherit_missing_levels:
        classes = _build_display_classes(dataset, eval_level, parent_models)
        num_classes = len(classes)
        train_name_to_idx = {name: idx for idx, name in enumerate(classes)}
    else:
        train_class_names_by_level = meta.get("class_names_by_level", {})
        classes = None
        if isinstance(train_class_names_by_level, dict):
            classes = train_class_names_by_level.get(eval_level)
        if not classes:
            classes = [
                inv_label_map[idx]
                for idx in range(dataset.num_classes_by_level[eval_level])
            ]
        num_classes = len(classes)
        train_name_to_idx = {name: idx for idx, name in enumerate(classes)}

    target_parent_idx = None
    if context.eval_only_parent is not None:
        target_parent_idx = int(context.eval_only_parent)

    if target_parent_idx is not None:
        runtime.ensure_parent_models(eval_level, parent_to_children)
        level_parent_models = runtime.parent_models.get(eval_level, {})
        entry = level_parent_models.get(target_parent_idx)
        if not entry or entry.get("model_path") is None:
            raise FileNotFoundError(
                f"No parent model for {eval_level}, parent={target_parent_idx}"
            )
        runtime.parent_models[eval_level] = {target_parent_idx: entry}
        parent_models = runtime.parent_models

        child_ids = entry.get("child_ids", [])
        if child_ids and (entry.get("model_path") is not None or len(child_ids) == 1):
            allowed_names = [inv_label_map[child_id] for child_id in child_ids]
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
        classes = [name for name in classes if name in allowed_names]
        num_classes = len(classes)
        train_name_to_idx = {name: idx for idx, name in enumerate(classes)}

    print(f"\n[Evaluate] level = {eval_level}, num_classes = {num_classes}")
    print("[Evaluate] Label Mapping (train order):")
    for idx in range(num_classes):
        print(f"{idx:2d} -> {classes[idx]}")

    def _map_pred_to_display(pred_level, pred_id, x):
        if pred_id < 0:
            return None
        if pred_level == eval_level:
            return pred_level, pred_id

        parent_level = dataset.get_parent_level(eval_level)
        if parent_level is None or pred_level != parent_level:
            return pred_level, pred_id

        entry = parent_models.get(eval_level, {}).get(int(pred_id))
        if entry is None:
            return pred_level, pred_id

        child_ids = entry.get("child_ids", [])
        model_path = entry.get("model_path")
        if not child_ids:
            return pred_level, pred_id

        if model_path is None:
            if len(child_ids) == 1:
                return eval_level, child_ids[0]
            return pred_level, pred_id

        logits = select_logits(
            runtime.get_parent_model(
                eval_level,
                int(pred_id),
                child_ids=child_ids,
                model_path=model_path,
            )(x)
        )
        probs = torch.softmax(logits, dim=1)
        pred_local = probs.argmax(1).item()
        pred_global = child_ids[pred_local]
        return eval_level, pred_global

    preprocessor = InputPreprocessor(config, device)
    num_classes_by_level = {
        level_name: dataset.num_classes_by_level[level_name]
        for level_name in level_order
    }
    class_names_by_level = {
        level_name: dataset.class_names_by_level[dataset.head_name_to_idx[level_name]]
        for level_name in level_order
    }

    all_preds, all_labels, all_paths = [], [], []
    skipped = 0
    invalid_pred = 0

    print("\n>>> Running TEST SET evaluation\n")

    parent_level_for_eval = dataset.get_parent_level(eval_level)

    for idx in test_idx:
        sample_level = eval_level
        if context.inherit_missing_levels:
            sample_level = _sample_effective_level(dataset, idx, eval_level)
            true_name = _effective_label_name(dataset, idx, sample_level)
            if true_name is None:
                skipped += 1
                continue
            if true_name not in train_name_to_idx:
                skipped += 1
                continue
            true_label_mapped = train_name_to_idx[true_name]
        else:
            true_label = labels[idx]
            if true_label < 0:
                skipped += 1
                continue

            true_name = inv_label_map.get(int(true_label))
            if true_name not in train_name_to_idx:
                skipped += 1
                continue
            true_label_mapped = train_name_to_idx[true_name]

        if context.eval_only_parent is not None and parent_level_for_eval is not None:
            parent_label = dataset.level_labels[
                idx, dataset.head_name_to_idx[parent_level_for_eval]
            ]
            if parent_label < 0 or int(parent_label) != int(context.eval_only_parent):
                invalid_pred += 1
                continue

        path = dataset.samples[idx]
        x = preprocessor(path)

        with torch.no_grad():
            result = run_cascade_inference(
                runtime,
                x,
                level_order=level_order,
                target_level=sample_level,
                num_classes_by_level=num_classes_by_level,
                class_names_by_level=class_names_by_level,
                parent_to_children=parent_to_children,
                fallback_to_previous=False,
            )
            pred = -1 if result is None else int(result["pred_global"])

        if context.inherit_missing_levels:
            mapped = _map_pred_to_display(sample_level, pred, x)
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

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        labels=range(num_classes),
    )
    acc = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]
    macro_recall = metrics["macro_recall"]

    print("\n=====================================")
    print(f" Test Set Accuracy: {acc * 100:.4f}%")
    print(f" Macro F1-score:    {macro_f1 * 100:.4f}%")
    print(f" Macro Recall:      {macro_recall * 100:.4f}%")
    print("=====================================\n")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    report_text = format_classification_report_text(
        report,
        classes,
        acc,
        macro_f1,
        macro_recall,
    )
    write_text(out_report, report_text)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    save_confusion_matrix_csv(cm, classes, out_cm_raw)
    save_confusion_matrix_figure(cm, classes, out_cm_png, show=True)

    df = pd.DataFrame({"path": all_paths, "label_true": all_labels, "label_pred": all_preds})
    df.to_csv(out_csv, index=False)

    print("All TEST SET results saved to:", result_dir)
    return result_dir


def run_evaluate_test_set(overrides=None):
    """先应用覆盖项，再执行测试集评估"""
    context = configure_evaluation(overrides)
    return evaluate_test_set(context)
