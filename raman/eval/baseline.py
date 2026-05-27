import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import compute_classification_metrics
from .experiment import (
    load_experiment_context_with_dataset,
    resolve_mode_result_dir,
    resolve_split_dir,
)
from raman.tool.hierarchy import load_hierarchy_meta
from .report import (
    format_classification_report_text,
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    write_text,
)
from .runtime import build_experiment_runtime
from raman.data import RamanDataset
from raman.training import load_split_files


def _load_baseline_context(exp_dir, target_level=None):
    """加载 PCA+SVM baseline 所需的通用上下文"""
    input_context, config = load_experiment_context_with_dataset(exp_dir)
    dataset = RamanDataset(config.dataset_root, augment=False, config=config)
    target_level = target_level or input_context.input_level
    target_level = dataset._resolve_level_name(target_level, field_name="target_level")

    split_dir = resolve_split_dir(input_context.exp_dir)
    split = load_split_files(dataset, split_dir) if split_dir else None
    if split is None:
        raise FileNotFoundError(
            f"实验根缺少 train_split.json/val_split.json，无法运行 baseline：{input_context.exp_dir}"
        )
    train_idx, val_idx = split

    meta = load_hierarchy_meta(input_context.exp_dir) or {}
    runtime = build_experiment_runtime(
        input_context.exp_dir,
        None,
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
        "train_idx": np.array(sorted(train_idx)),
        "val_idx": np.array(sorted(val_idx)),
        "runtime": runtime,
    }

def _extract_features(dataset, indices, level_idx, use_all_channels, label_map=None, allowed_labels=None):
    """按索引提取特征和标签，并跳过无效标签样本"""
    x_list = []
    y_list = []
    skipped = 0
    allowed_labels = set(int(item) for item in allowed_labels) if allowed_labels else None

    for idx in indices:
        x, labels, _ = dataset[idx]
        y = int(labels[level_idx] if labels.ndim > 0 else labels)
        if y < 0:
            skipped += 1
            continue
        if allowed_labels is not None and y not in allowed_labels:
            skipped += 1
            continue

        feat = x.reshape(-1).numpy() if use_all_channels else x[0].numpy()
        x_list.append(feat)
        y_list.append(label_map.get(y, y) if label_map else y)

    if not x_list:
        raise RuntimeError("筛选后没有有效样本")

    return np.stack(x_list, axis=0), np.array(y_list, dtype=np.int64), skipped


def _filter_indices_by_parent(dataset, indices, level_name, parent_idx):
    """把样本索引筛到指定 parent 分支内"""
    parent_level = dataset.get_parent_level(level_name)
    if parent_level is None:
        return indices
    parent_level_idx = dataset.head_name_to_idx[parent_level]
    labels = dataset.level_labels[indices]
    mask = labels[:, parent_level_idx] == int(parent_idx)
    return indices[mask]


def _fit_predict_svm(
    dataset,
    train_idx,
    val_idx,
    level_name,
    *,
    use_all_channels,
    pca_n_components,
    svm_c,
    svm_kernel,
    svm_gamma,
    random_state,
    label_map=None,
    allowed_labels=None,
):
    """提取光谱特征，完成标准化、PCA、SVM 训练和验证预测"""
    level_idx = dataset.head_name_to_idx[level_name]
    x_train, y_train, skipped_train = _extract_features(
        dataset,
        train_idx,
        level_idx,
        use_all_channels,
        label_map=label_map,
        allowed_labels=allowed_labels,
    )
    x_val, y_val, skipped_val = _extract_features(
        dataset,
        val_idx,
        level_idx,
        use_all_channels,
        label_map=label_map,
        allowed_labels=allowed_labels,
    )

    print(f"[Info] 训练样本数: {len(y_train)} (跳过 {skipped_train})")
    print(f"[Info] 验证样本数: {len(y_val)} (跳过 {skipped_val})")

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_val_std = scaler.transform(x_val)

    pca = PCA(n_components=pca_n_components, random_state=random_state)
    x_train_pca = pca.fit_transform(x_train_std)
    x_val_pca = pca.transform(x_val_std)

    svm = SVC(C=svm_c, kernel=svm_kernel, gamma=svm_gamma)
    svm.fit(x_train_pca, y_train)
    y_pred = svm.predict(x_val_pca)
    return x_train_pca, y_train, y_val, y_pred, pca


def _write_baseline_outputs(result_dir, class_names, x_train_pca, y_train, y_val, y_pred, pca):
    """写出 PCA+SVM baseline 的指标和图表"""
    result_dir = os.fspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    labels = list(range(len(class_names)))
    metrics = compute_classification_metrics(y_val, y_pred, labels=labels)
    acc = metrics["accuracy"]
    print(f"\n[Baseline] 验证集 Accuracy: {acc * 100:.4f}%")

    report_dict = classification_report(
        y_val,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = format_classification_report_text(report_dict, class_names, acc)
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    if pca is None or x_train_pca is None:
        pca_text = "PCA components: per-parent\nExplained variance ratio: per-parent\n"
    else:
        pca_text = (
            f"PCA components: {x_train_pca.shape[1]}\n"
            "Explained variance ratio:\n"
            f"{np.array2string(pca.explained_variance_ratio_, precision=4)}\n"
        )
    metrics_text = f"Accuracy: {acc * 100:.4f}%\n{pca_text}\n{report_text}"
    write_text(os.path.join(result_dir, "metrics.txt"), metrics_text)
    save_confusion_matrix_csv(cm, class_names, os.path.join(result_dir, "confusion_matrix.csv"))
    save_confusion_matrix_figure(cm, class_names, os.path.join(result_dir, "confusion_matrix.png"))

    out_pca_png = os.path.join(result_dir, "pca_scatter.png")
    if pca is None or x_train_pca is None:
        print("[Info] per-parent baseline 聚合结果跳过 PCA 散点图")
    elif x_train_pca.shape[1] < 2:
        print("[Warn] PCA 维数小于 2，跳过散点图")
    else:
        plt.figure(figsize=(8, 6))
        for cls_idx, cls_name in enumerate(class_names):
            train_mask = y_train == cls_idx
            if not train_mask.any():
                continue
            plt.scatter(
                x_train_pca[train_mask, 0],
                x_train_pca[train_mask, 1],
                s=12,
                alpha=0.6,
                label=cls_name,
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA scatter (train only)")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_pca_png, dpi=300)
        plt.close()

    print("All BASELINE VAL results saved to:", result_dir)
    return result_dir


def _run_global_baseline(ctx, result_dir, **kwargs):
    """运行某一层全局类别空间的 PCA+SVM baseline"""
    dataset = ctx["dataset"]
    level_name = ctx["target_level"]
    class_names = dataset.get_class_names(level_name)
    result = _fit_predict_svm(
        dataset,
        ctx["train_idx"],
        ctx["val_idx"],
        level_name,
        **kwargs,
    )
    return _write_baseline_outputs(result_dir, class_names, *result)


def _run_parent_baseline(ctx, result_dir, parent_idx, **kwargs):
    """运行单个 parent 子类空间内的 PCA+SVM baseline"""
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    runtime.ensure_parent_models(level_name, runtime.parent_to_children)
    entry = runtime.parent_models.get(level_name, {}).get(int(parent_idx))
    if entry is None:
        raise ValueError(f"No parent entry for {level_name}, parent={parent_idx}")
    child_ids = [int(item) for item in entry.get("child_ids", [])]
    if len(child_ids) <= 1:
        raise ValueError(f"Parent {parent_idx} has only one child; no baseline model needed.")

    class_names_all = dataset.get_class_names(level_name)
    class_names = [class_names_all[child_id] for child_id in child_ids]
    label_map = {child_id: local_idx for local_idx, child_id in enumerate(child_ids)}
    train_idx = _filter_indices_by_parent(dataset, ctx["train_idx"], level_name, parent_idx)
    val_idx = _filter_indices_by_parent(dataset, ctx["val_idx"], level_name, parent_idx)
    result = _fit_predict_svm(
        dataset,
        train_idx,
        val_idx,
        level_name,
        label_map=label_map,
        allowed_labels=set(child_ids),
        **kwargs,
    )
    return _write_baseline_outputs(result_dir, class_names, *result)


def run_baseline_single_model(
    run_dir,
    level=None,
    *,
    use_all_channels=False,
    pca_n_components=0.95,
    svm_c=1.0,
    svm_kernel="rbf",
    svm_gamma="scale",
    random_state=42,
):
    """只针对传入 run_* 目录运行 PCA+SVM baseline"""
    ctx = _load_baseline_context(run_dir, target_level=level)
    input_context = ctx["input_context"]
    if not input_context.is_single_run:
        raise ValueError("run_baseline_single_model 必须传入具体 run_* 或 best/run_* 目录")

    result_dir = os.path.join(input_context.input_run_dir, "baseline_val_result")
    kwargs = {
        "use_all_channels": use_all_channels,
        "pca_n_components": pca_n_components,
        "svm_c": svm_c,
        "svm_kernel": svm_kernel,
        "svm_gamma": svm_gamma,
        "random_state": random_state,
    }
    if input_context.input_parent_idx is not None:
        return _run_parent_baseline(ctx, result_dir, input_context.input_parent_idx, **kwargs)
    return _run_global_baseline(ctx, result_dir, **kwargs)


def _run_level_parent_baseline(ctx, result_dir, **kwargs):
    """单层多 parent baseline：按真实 parent 分开训练，再汇总为全局指标"""
    dataset = ctx["dataset"]
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    level_idx = dataset.head_name_to_idx[level_name]
    parent_level = dataset.get_parent_level(level_name)
    parent_level_idx = dataset.head_name_to_idx[parent_level]
    class_names = dataset.get_class_names(level_name)

    all_labels, all_preds = [], []
    parent_entries = runtime.parent_models.get(level_name, {})
    for parent_idx, entry in sorted(parent_entries.items()):
        child_ids = [int(item) for item in entry.get("child_ids", [])]
        if not child_ids:
            continue

        train_idx = _filter_indices_by_parent(dataset, ctx["train_idx"], level_name, parent_idx)
        val_idx = _filter_indices_by_parent(dataset, ctx["val_idx"], level_name, parent_idx)
        if len(child_ids) == 1:
            labels = dataset.level_labels[val_idx, level_idx]
            mask = labels >= 0
            all_labels.extend(labels[mask].astype(int).tolist())
            all_preds.extend([child_ids[0]] * int(mask.sum()))
            continue

        label_map = {child_id: local_idx for local_idx, child_id in enumerate(child_ids)}
        _, _, y_val, y_pred, _ = _fit_predict_svm(
            dataset,
            train_idx,
            val_idx,
            level_name,
            label_map=label_map,
            allowed_labels=set(child_ids),
            **kwargs,
        )
        local_to_child = {local_idx: child_id for child_id, local_idx in label_map.items()}
        all_labels.extend([local_to_child[int(item)] for item in y_val])
        all_preds.extend([local_to_child[int(item)] for item in y_pred])

    return _write_baseline_outputs(
        result_dir,
        class_names,
        None,
        None,
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_preds, dtype=np.int64),
        None,
    )


def run_baseline_level_only(
    exp_dir,
    target_level,
    *,
    use_all_channels=False,
    pca_n_components=0.95,
    svm_c=1.0,
    svm_kernel="rbf",
    svm_gamma="scale",
    random_state=42,
):
    """只按目标层真实父类分发运行单层 PCA+SVM baseline"""
    ctx = _load_baseline_context(exp_dir, target_level=target_level)
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    result_dir = resolve_mode_result_dir(ctx["input_context"].exp_dir, "baseline", level_name, "level_only")
    kwargs = {
        "use_all_channels": use_all_channels,
        "pca_n_components": pca_n_components,
        "svm_c": svm_c,
        "svm_kernel": svm_kernel,
        "svm_gamma": svm_gamma,
        "random_state": random_state,
    }

    parent_level = ctx["dataset"].get_parent_level(level_name)
    if parent_level is None:
        out = _run_global_baseline(ctx, result_dir, **kwargs)
    else:
        runtime.ensure_parent_models(level_name, runtime.parent_to_children)
        parent_entries = runtime.parent_models.get(level_name, {})
        has_parent_model = any(entry.get("model_path") is not None for entry in parent_entries.values())
        all_single_child = bool(parent_entries) and all(
            len(entry.get("child_ids", [])) <= 1 for entry in parent_entries.values()
        )
        if has_parent_model or all_single_child:
            out = _run_level_parent_baseline(ctx, result_dir, **kwargs)
        else:
            out = _run_global_baseline(ctx, result_dir, **kwargs)

    return out


def _run_baseline_into_mode_dir(ctx, result_dir, **kwargs):
    """按目标层真实父类分发，输出到指定目录"""
    runtime = ctx["runtime"]
    level_name = ctx["target_level"]
    parent_level = ctx["dataset"].get_parent_level(level_name)
    if parent_level is None:
        return _run_global_baseline(ctx, result_dir, **kwargs)

    runtime.ensure_parent_models(level_name, runtime.parent_to_children)
    parent_entries = runtime.parent_models.get(level_name, {})
    has_parent_model = any(entry.get("model_path") is not None for entry in parent_entries.values())
    all_single_child = bool(parent_entries) and all(
        len(entry.get("child_ids", [])) <= 1 for entry in parent_entries.values()
    )
    if has_parent_model or all_single_child:
        return _run_level_parent_baseline(ctx, result_dir, **kwargs)
    return _run_global_baseline(ctx, result_dir, **kwargs)


def run_baseline_cascade(
    exp_dir,
    target_level,
    *,
    use_all_channels=False,
    pca_n_components=0.95,
    svm_c=1.0,
    svm_kernel="rbf",
    svm_gamma="scale",
    random_state=42,
):
    """在多层级联结果目录下运行目标层 PCA+SVM baseline"""
    ctx = _load_baseline_context(exp_dir, target_level=target_level)
    target_level, level_order = ctx["target_level"], []
    for level_name in ctx["dataset"].level_names:
        level_order.append(level_name)
        if level_name == target_level:
            break

    runtime = ctx["runtime"]
    runtime.build_level_model_paths(level_order)
    for level_name in level_order:
        runtime.ensure_parent_models(level_name, runtime.parent_to_children)

    result_dir = resolve_mode_result_dir(ctx["input_context"].exp_dir, "baseline", target_level, "cascade")
    out = _run_baseline_into_mode_dir(
        ctx,
        result_dir,
        use_all_channels=use_all_channels,
        pca_n_components=pca_n_components,
        svm_c=svm_c,
        svm_kernel=svm_kernel,
        svm_gamma=svm_gamma,
        random_state=random_state,
    )

    return out
