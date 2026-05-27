import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .common import compute_classification_metrics, run_cascade_inference, select_logits
from .experiment import (
    collect_used_runs,
    load_experiment_context_with_dataset,
    resolve_mode_result_dir,
    resolve_mode_result_root,
    resolve_split_dir,
    validate_parent_split_hashes,
    write_used_runs,
)
from raman.tool.hierarchy import load_hierarchy_meta, resolve_level_order
from .report import (
    format_classification_report_text,
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    write_text,
)
from .runtime import build_experiment_runtime
from raman.data import InputPreprocessor, RamanDataset
from raman.training import load_split_files


def _load_eval_context(exp_dir, target_level=None):
    """加载验证评估所需的通用上下文"""
    input_context, config = load_experiment_context_with_dataset(exp_dir)
    dataset = RamanDataset(config.dataset_root, augment=False, config=config)
    target_level = target_level or input_context.input_level
    target_level, level_order = resolve_level_order(dataset, target_level)

    split_dir = resolve_split_dir(input_context.exp_dir)
    split = load_split_files(dataset, split_dir) if split_dir else None
    if split is None:
        raise FileNotFoundError(
            f"实验根缺少 train_split.json/val_split.json，无法进行验证集评估：{input_context.exp_dir}"
        )
    _, val_idx = split

    use_cuda = config.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Using: {device} (config.use_gpu={config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})"
    )

    meta = load_hierarchy_meta(input_context.exp_dir) or {}
    runtime = build_experiment_runtime(
        input_context.exp_dir,
        device,
        config=config,
        meta=meta,
        run_selection=input_context.run_selection,
    )
    if not runtime.parent_to_children:
        runtime.parent_to_children = dataset.parent_to_children

    return {
        "input_context": input_context,
        "config": config,
        "dataset": dataset,
        "target_level": target_level,
        "level_order": level_order,
        "val_idx": np.array(sorted(val_idx)),
        "device": device,
        "runtime": runtime,
    }


def _write_eval_outputs(result_dir, classes, all_paths, all_labels, all_preds):
    """写出验证集明细、混淆矩阵和分类报告"""
    result_dir = os.fspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    if not all_labels:
        print("No valid samples for this level.")
        return result_dir

    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)
    labels = list(range(len(classes)))
    metrics = compute_classification_metrics(all_labels, all_preds, labels=labels)
    acc = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]
    macro_recall = metrics["macro_recall"]

    print("\n=====================================")
    print(f" Val Set Accuracy:  {acc * 100:.4f}%")
    print(f" Macro F1-score:    {macro_f1 * 100:.4f}%")
    print(f" Macro Recall:      {macro_recall * 100:.4f}%")
    print("=====================================\n")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        labels=labels,
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
    write_text(os.path.join(result_dir, "classification_report.txt"), report_text)

    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    save_confusion_matrix_csv(cm, classes, os.path.join(result_dir, "confusion_matrix_raw.csv"))
    save_confusion_matrix_figure(
        cm,
        classes,
        os.path.join(result_dir, "confusion_matrix.png"),
    )

    df = pd.DataFrame(
        {"path": all_paths, "label_true": all_labels, "label_pred": all_preds}
    )
    df.to_csv(os.path.join(result_dir, "val_eval_results.csv"), index=False)
    print("All VAL SET results saved to:", result_dir)
    return result_dir


def _eval_global_model(ctx, result_dir):
    """评估某一层的全局单模型"""
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    level_idx = dataset.head_name_to_idx[level_name]
    classes = dataset.get_class_names(level_name)
    runtime.build_level_model_paths([level_name])
    model = runtime.get_level_model(level_name, num_classes=len(classes))
    preprocessor = InputPreprocessor(ctx["config"], ctx["device"])

    all_paths, all_labels, all_preds = [], [], []
    labels = dataset.level_labels[:, level_idx]
    for idx in ctx["val_idx"]:
        true_label = int(labels[idx])
        if true_label < 0:
            continue
        x = preprocessor(dataset.samples[idx])
        with torch.no_grad():
            logits = select_logits(model(x), head_name=level_name)
            pred = int(torch.softmax(logits, dim=1).argmax(1).item())
        all_paths.append(dataset.samples[idx])
        all_labels.append(true_label)
        all_preds.append(pred)

    return _write_eval_outputs(result_dir, classes, all_paths, all_labels, all_preds)


def _eval_parent_model(ctx, result_dir, parent_idx):
    """评估某个 parent 对应的单个子模型"""
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    level_idx = dataset.head_name_to_idx[level_name]
    parent_level = dataset.get_parent_level(level_name)
    if parent_level is None:
        raise ValueError(f"{level_name} 没有父层，不能按 parent 单测")
    parent_level_idx = dataset.head_name_to_idx[parent_level]

    runtime.ensure_parent_models(level_name, runtime.parent_to_children)
    entry = runtime.parent_models.get(level_name, {}).get(int(parent_idx))
    if entry is None or entry.get("model_path") is None:
        raise FileNotFoundError(f"No parent model for {level_name}, parent={parent_idx}")

    child_ids = [int(item) for item in entry.get("child_ids", [])]
    if not child_ids:
        raise ValueError(f"缺少 child_ids：level={level_name}, parent={parent_idx}")
    classes_all = dataset.get_class_names(level_name)
    classes = [classes_all[child_id] for child_id in child_ids]
    child_to_local = {child_id: local_idx for local_idx, child_id in enumerate(child_ids)}
    model = runtime.get_parent_model(
        level_name,
        int(parent_idx),
        child_ids=child_ids,
        model_path=entry.get("model_path"),
    )
    preprocessor = InputPreprocessor(ctx["config"], ctx["device"])

    all_paths, all_labels, all_preds = [], [], []
    labels = dataset.level_labels
    for idx in ctx["val_idx"]:
        if int(labels[idx, parent_level_idx]) != int(parent_idx):
            continue
        true_global = int(labels[idx, level_idx])
        if true_global not in child_to_local:
            continue
        x = preprocessor(dataset.samples[idx])
        with torch.no_grad():
            logits = select_logits(model(x), head_name=level_name)
            pred_local = int(torch.softmax(logits, dim=1).argmax(1).item())
        all_paths.append(dataset.samples[idx])
        all_labels.append(child_to_local[true_global])
        all_preds.append(pred_local)

    return _write_eval_outputs(result_dir, classes, all_paths, all_labels, all_preds)


def run_eval_single_model(run_dir, level=None):
    """只使用传入 run_* 目录中的单个模型做验证集评估"""
    ctx = _load_eval_context(run_dir, target_level=level)
    input_context = ctx["input_context"]
    if not input_context.is_single_run:
        raise ValueError("run_eval_single_model 必须传入具体 run_* 或 best/run_* 目录")

    result_dir = os.path.join(input_context.input_run_dir, "val_result")
    print(f"\n[Val Split] Val samples: {len(ctx['val_idx'])}")
    if input_context.input_parent_idx is not None:
        return _eval_parent_model(ctx, result_dir, input_context.input_parent_idx)
    return _eval_global_model(ctx, result_dir)


def run_eval_level_only(exp_dir, target_level):
    """只使用目标层模型，按真实父类分发 parent 子模型"""
    ctx = _load_eval_context(exp_dir, target_level=target_level)
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    result_root = resolve_mode_result_root(ctx["input_context"].exp_dir, level_name, "level_only")
    result_dir = resolve_mode_result_dir(ctx["input_context"].exp_dir, "val", level_name, "level_only")

    print(f"\n[Val Split] Val samples: {len(ctx['val_idx'])}")
    parent_level = dataset.get_parent_level(level_name)
    if parent_level is None:
        result = _eval_global_model(ctx, result_dir)
    else:
        runtime.ensure_parent_models(level_name, runtime.parent_to_children)
        parent_entries = runtime.parent_models.get(level_name, {})
        has_parent_model = any(entry.get("model_path") is not None for entry in parent_entries.values())
        all_single_child = bool(parent_entries) and all(
            len(entry.get("child_ids", [])) <= 1 for entry in parent_entries.values()
        )
        if has_parent_model or all_single_child:
            validate_parent_split_hashes(ctx["input_context"].exp_dir, level_name, parent_entries)
            result = _eval_level_parent_routed(ctx, result_dir)
        else:
            result = _eval_global_model(ctx, result_dir)

    used_runs = collect_used_runs(
        ctx["input_context"].exp_dir,
        runtime,
        level_order=[level_name],
        target_level=level_name,
    )
    write_used_runs(
        result_root,
        mode="level_only",
        target_level=level_name,
        runs=used_runs,
    )
    return result


def _eval_level_parent_routed(ctx, result_dir):
    """单层多模型评估：按真实 parent 把样本分发给对应子模型"""
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    level_idx = dataset.head_name_to_idx[level_name]
    parent_level = dataset.get_parent_level(level_name)
    parent_level_idx = dataset.head_name_to_idx[parent_level]
    classes = dataset.get_class_names(level_name)
    preprocessor = InputPreprocessor(ctx["config"], ctx["device"])

    all_paths, all_labels, all_preds = [], [], []
    labels = dataset.level_labels
    parent_entries = runtime.parent_models.get(level_name, {})
    for idx in ctx["val_idx"]:
        true_label = int(labels[idx, level_idx])
        parent_idx = int(labels[idx, parent_level_idx])
        if true_label < 0 or parent_idx < 0:
            continue
        entry = parent_entries.get(parent_idx)
        if entry is None:
            continue
        child_ids = [int(item) for item in entry.get("child_ids", [])]
        if not child_ids:
            continue

        model_path = entry.get("model_path")
        if model_path is None:
            if len(child_ids) != 1:
                continue
            pred_global = child_ids[0]
        else:
            x = preprocessor(dataset.samples[idx])
            model = runtime.get_parent_model(
                level_name,
                parent_idx,
                child_ids=child_ids,
                model_path=model_path,
            )
            with torch.no_grad():
                logits = select_logits(model(x), head_name=level_name)
                pred_local = int(torch.softmax(logits, dim=1).argmax(1).item())
            pred_global = child_ids[pred_local]

        all_paths.append(dataset.samples[idx])
        all_labels.append(true_label)
        all_preds.append(int(pred_global))

    return _write_eval_outputs(result_dir, classes, all_paths, all_labels, all_preds)


def run_eval_cascade(exp_dir, target_level):
    """从顶层到目标层执行多层多模型级联验证集评估"""
    ctx = _load_eval_context(exp_dir, target_level=target_level)
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    level_order = ctx["level_order"]
    level_idx = dataset.head_name_to_idx[level_name]
    result_root = resolve_mode_result_root(ctx["input_context"].exp_dir, level_name, "cascade")
    result_dir = resolve_mode_result_dir(ctx["input_context"].exp_dir, "val", level_name, "cascade")

    runtime.build_level_model_paths(level_order)
    for item in level_order:
        runtime.ensure_parent_models(item, runtime.parent_to_children)
        validate_parent_split_hashes(
            ctx["input_context"].exp_dir,
            item,
            runtime.parent_models.get(item, {}),
        )

    classes = dataset.get_class_names(level_name)
    num_classes_by_level = {
        item: dataset.num_classes_by_level[item]
        for item in level_order
    }
    class_names_by_level = {
        item: dataset.get_class_names(item)
        for item in level_order
    }
    preprocessor = InputPreprocessor(ctx["config"], ctx["device"])

    print(f"\n[Val Split] Val samples: {len(ctx['val_idx'])}")
    all_paths, all_labels, all_preds = [], [], []
    labels = dataset.level_labels[:, level_idx]
    for idx in ctx["val_idx"]:
        true_label = int(labels[idx])
        if true_label < 0:
            continue
        x = preprocessor(dataset.samples[idx])
        with torch.no_grad():
            result = run_cascade_inference(
                runtime,
                x,
                level_order=level_order,
                target_level=level_name,
                num_classes_by_level=num_classes_by_level,
                class_names_by_level=class_names_by_level,
                parent_to_children=runtime.parent_to_children,
                fallback_to_previous=False,
            )
        if result is None:
            continue
        all_paths.append(dataset.samples[idx])
        all_labels.append(true_label)
        all_preds.append(int(result["pred_global"]))

    out = _write_eval_outputs(result_dir, classes, all_paths, all_labels, all_preds)
    used_runs = collect_used_runs(
        ctx["input_context"].exp_dir,
        runtime,
        level_order=level_order,
        target_level=level_name,
    )
    write_used_runs(
        result_root,
        mode="cascade",
        target_level=level_name,
        runs=used_runs,
    )
    return out
