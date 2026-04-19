import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from raman.data import RamanDataset
from raman.eval.experiment import (
    load_experiment_with_dataset,
    load_hierarchy_meta,
    resolve_level_model_path,
)
from raman.eval.runtime import build_experiment_runtime
from raman.training import (
    build_label_map_np,
    split_by_lowest_level_ratio,
    load_split_files,
)
from .ig import (
    _effective_label_names,
    compute_ig_batches,
    compute_channel_importance_from_ig,
    compute_band_importance_from_ig,
    compute_class_band_importance_ig,
    compute_class_mean_spectrum,
    build_wavenumber_axis,
    _get_bad_bands,
    plot_band_importance_heatmap,
    save_topk_bands_csv,
    _plot_channel_importance,
)
from .gradcam import (
    collect_analyzable_layers,
    LayerGradCAMAnalyzer,
    merge_scores_by_group,
    _plot_layer_importance,
)
from .embedding import collect_embeddings_train_test, plot_embedding_hierarchical
from .se import log_seblock_summary

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

class HeatmapConfig:
    # 波段重要性热图（IG）相关配置
    num_batches: int = 10          # 采样多少个 batch
    steps: int = 32                # IG 积分步数
    max_per_class: int = 50        # 每类最多样本数
    row_norm: str = "max"          # 行归一化方式：max / sum / none
    use_train_loader: bool = True  # True=训练集，False=测试集
    topk_per_class: int = 5        # 每类 top-k 波段导出

class AnalysisOverrides:
    """统一收拢分析入口的覆盖项"""

    exp_dir: str | None = None
    mode: str = "single"
    analysis_level: str | None = None
    parent_idx: int | str | None = None
    inherit_missing_levels: bool = False
    fallback_to_single: bool = True

def _ensure_heatmap_cfg(cfg):
    # None 时返回默认热图配置，避免外部传空
    return cfg if cfg is not None else HeatmapConfig()

def _normalize_parent_idx(parent_idx):
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

def _build_analysis_tasks(
    exp_dir,
    analysis_level,
    head_index,
    full_dataset,
    level_models,
    parent_models,
    parent_idx_setting,
):
    """解析需要分析的模型任务（单模型或按 parent 拆分）"""
    parent_idx_setting = _normalize_parent_idx(parent_idx_setting)
    parent_entries = parent_models.get(analysis_level, {})
    tasks = []
    auto_all = False

    if parent_idx_setting is None:
        model_path = resolve_level_model_path(exp_dir, analysis_level, level_models)
        if os.path.exists(model_path):
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

def _build_task_loaders(
    task,
    config,
    full_dataset,
    analysis_level,
    head_index,
    train_idx_all,
    test_idx_all,
    base_train_dataset,
    base_test_dataset,
):
    """按 parent 过滤样本并构建 DataLoader"""
    parent_idx = task["parent_idx"]
    child_ids = task["child_ids"]
    inherit_missing = getattr(config, "inherit_missing_levels", False)

    train_idx = train_idx_all
    test_idx = test_idx_all

    if parent_idx is not None:
        parent_level = full_dataset.get_parent_level(analysis_level)
        if parent_level is None:
            raise ValueError("Top level has no parent; cannot use PARENT_IDX.")
        parent_level_idx = full_dataset.head_name_to_idx[parent_level]

        labels_train = full_dataset.level_labels[train_idx]
        labels_test = full_dataset.level_labels[test_idx]

        train_mask = (labels_train[:, parent_level_idx] == parent_idx)
        test_mask = (labels_test[:, parent_level_idx] == parent_idx)
        if not inherit_missing:
            train_mask = train_mask & (labels_train[:, head_index] >= 0)
            test_mask = test_mask & (labels_test[:, head_index] >= 0)

        train_idx = train_idx[train_mask]
        test_idx = test_idx[test_mask]

        label_map_np = build_label_map_np(
            child_ids,
            full_dataset.num_classes_by_level[analysis_level]
        )

        train_dataset = LabelMapDataset(base_train_dataset, head_index, label_map_np)
        test_dataset = LabelMapDataset(base_test_dataset, head_index, label_map_np)
    else:
        train_dataset = base_train_dataset
        test_dataset = base_test_dataset

    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False
    )

    if len(test_subset) == 0:
        test_loader = train_loader

    return train_loader, test_loader, train_subset, test_subset

def run_aggregate_analysis(
    exp_dir,
    config,
    full_dataset,
    analysis_level,
    head_index,
    tasks,
    train_idx_all,
    test_idx_all,
    base_train_dataset,
    base_test_dataset,
    runtime=None,
    device=None,
    meta=None,
    heatmap_cfg=None,
):
    """跨 parent 聚合分析：用样本数加权合并结果"""
    heatmap_cfg = _ensure_heatmap_cfg(heatmap_cfg)
    inherit_missing = getattr(config, "inherit_missing_levels", False)
    missing_tag = getattr(full_dataset, "MISSING_TAG", "__missing__")
    if inherit_missing:
        global_class_names = _effective_label_names(full_dataset, analysis_level, missing_tag)
        global_num_classes = len(global_class_names)
        global_name_to_idx = {n: i for i, n in enumerate(global_class_names)}
    else:
        global_num_classes = full_dataset.num_classes_by_level[analysis_level]
        global_class_names = full_dataset.class_names_by_level[head_index]
        global_name_to_idx = None

    # ---------------- 输出目录 ----------------
    analysis_dir = os.path.join(exp_dir, f"{analysis_level}_aggregate_analysis")
    fig_dir = os.path.join(analysis_dir, "figures")
    log_dir = os.path.join(analysis_dir, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "analysis_log.txt")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + os.linesep)

    log(f"Aggregate analysis for {analysis_level} over {len(tasks)} parents.")

    # ---------------- 模型与统计缓存 ----------------
    channel_names = [f"{config.norm_method}"]
    if config.smooth_use:
        channel_names.append("smooth")
    if config.d1_use:
        channel_names.append("d1")

    agg_channel = None
    agg_layer = None
    weight_total = 0
    layer_names = None

    band_total = None
    band_counts = np.zeros(global_num_classes, dtype=np.int64)
    mean_total = None
    mean_counts = np.zeros(global_num_classes, dtype=np.int64)

    if device is None:
        use_cuda = (config.use_gpu and torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda else "cpu")
    if runtime is None:
        runtime = build_experiment_runtime(
            exp_dir,
            device,
            config=config,
            meta=(meta or load_hierarchy_meta(exp_dir) or {}),
        )
    if not runtime.parent_to_children:
        runtime.parent_to_children = full_dataset.parent_to_children
    log(f"Using device: {device} (config.use_gpu={config.use_gpu}, cuda_available={torch.cuda.is_available()})")

    for task in tasks:
        parent_idx = task["parent_idx"]
        log(f"--- Parent {parent_idx} ---")

        train_loader, test_loader, train_subset, _ = _build_task_loaders(
            task,
            config,
            full_dataset,
            analysis_level,
            head_index,
            train_idx_all,
            test_idx_all,
            base_train_dataset,
            base_test_dataset,
        )
        if len(train_subset) == 0:
            log(f"Skip parent {parent_idx}: no samples after filtering.")
            continue

        model = runtime.get_parent_model(
            analysis_level,
            parent_idx,
            child_ids=task["child_ids"],
            model_path=task["model_path"],
        )

        # warmup forward
        sample_x, _, _ = next(iter(train_loader))
        sample_x = sample_x.to(device)
        _ = model(sample_x)

        heatmap_loader = train_loader if heatmap_cfg.use_train_loader else test_loader

        # ----- IG：先生成原始归因结果，再分别汇总通道/波段重要性 -----
        band_num_classes = global_num_classes if inherit_missing else task["num_classes"]
        ig_batches = compute_ig_batches(
            model,
            heatmap_loader,
            device,
            steps=heatmap_cfg.steps,
            num_batches=heatmap_cfg.num_batches,
            head_index=head_index,
            num_classes=band_num_classes,
            max_per_class=heatmap_cfg.max_per_class,
            level_name=analysis_level if inherit_missing else None,
            label_name_to_idx=global_name_to_idx,
            missing_tag=missing_tag,
        )
        ch_imp = compute_channel_importance_from_ig(ig_batches)
        band_importance, counts = compute_band_importance_from_ig(
            ig_batches,
            num_classes=band_num_classes,
            max_per_class=heatmap_cfg.max_per_class,
        )
        weight = len(train_subset)
        if agg_channel is None:
            agg_channel = ch_imp * weight
        else:
            agg_channel += ch_imp * weight

        # ----- layer importance -----
        analyzable, groups = collect_analyzable_layers(model)
        analyzer = LayerGradCAMAnalyzer(model, device)
        for name, layer in analyzable.items():
            analyzer.register_layer(name, layer)
        layer_scores = analyzer.run(
            train_loader,
            save_dir=None,
            num_batches=3,
            head_index=head_index
        )
        merged_scores = merge_scores_by_group(layer_scores, groups)
        if layer_names is None:
            layer_names = list(merged_scores.keys())
            agg_layer = {k: 0.0 for k in layer_names}
        for k in layer_names:
            agg_layer[k] += merged_scores.get(k, 0.0) * weight

        weight_total += weight

        # ----- mean spectrum -----
        mean_spectra, mean_ct = compute_class_mean_spectrum(
            heatmap_loader,
            device,
            num_classes=band_num_classes,
            head_index=head_index,
            max_per_class=heatmap_cfg.max_per_class,
            level_name=analysis_level if inherit_missing else None,
            label_name_to_idx=global_name_to_idx,
            missing_tag=missing_tag,
        )

        if band_total is None:
            band_total = np.zeros((global_num_classes, band_importance.shape[1]), dtype=np.float32)
            mean_total = np.zeros((global_num_classes, mean_spectra.shape[1]), dtype=np.float32)

        if inherit_missing:
            for idx in range(global_num_classes):
                c = int(counts[idx])
                if c > 0:
                    band_total[idx] += band_importance[idx] * c
                    band_counts[idx] += c
                mc = int(mean_ct[idx])
                if mc > 0:
                    mean_total[idx] += mean_spectra[idx] * mc
                    mean_counts[idx] += mc
        else:
            child_ids = task["child_ids"]
            for local_idx, global_idx in enumerate(child_ids):
                c = int(counts[local_idx])
                if c > 0:
                    band_total[global_idx] += band_importance[local_idx] * c
                    band_counts[global_idx] += c
                mc = int(mean_ct[local_idx])
                if mc > 0:
                    mean_total[global_idx] += mean_spectra[local_idx] * mc
                    mean_counts[global_idx] += mc

    # 单子类 parent：用上一级模型的重要性补齐
    parent_level = full_dataset.get_parent_level(analysis_level)
    if (not inherit_missing) and parent_level is not None and band_total is not None:
        parent_to_children = full_dataset.parent_to_children.get(analysis_level, {})
        single_child_map = {
            int(p): int(children[0])
            for p, children in parent_to_children.items()
            if len(children) == 1
        }

        if single_child_map:
            parent_model_path = runtime.build_level_model_paths([parent_level]).get(parent_level)

            if parent_model_path and os.path.exists(parent_model_path):
                parent_head_index = full_dataset.head_name_to_idx[parent_level]
                num_parent_classes = full_dataset.num_classes_by_level[parent_level]

                dummy_task = {"parent_idx": None, "child_ids": None}
                parent_train_loader, parent_test_loader, _, _ = _build_task_loaders(
                    dummy_task,
                    config,
                    full_dataset,
                    parent_level,
                    parent_head_index,
                    train_idx_all,
                    test_idx_all,
                    base_train_dataset,
                    base_test_dataset,
                )
                parent_loader = (
                    parent_train_loader if heatmap_cfg.use_train_loader else parent_test_loader
                )

                parent_model = runtime.get_level_model(
                    parent_level,
                    num_classes=num_parent_classes,
                )

                parent_importance, parent_counts = compute_class_band_importance_ig(
                    parent_model,
                    parent_loader,
                    device,
                    num_classes=num_parent_classes,
                    head_index=parent_head_index,
                    steps=heatmap_cfg.steps,
                    num_batches=heatmap_cfg.num_batches,
                    max_per_class=heatmap_cfg.max_per_class,
                )
                # parent level 的平均光谱（用于单子类 child 继承）
                parent_mean_spectra, parent_mean_ct = compute_class_mean_spectrum(
                    parent_loader,
                    device,
                    num_classes=num_parent_classes,
                    head_index=parent_head_index,
                    max_per_class=heatmap_cfg.max_per_class,
                )

                for p_idx, child_idx in single_child_map.items():
                    if child_idx < 0 or child_idx >= global_num_classes:
                        continue
                    if band_counts[child_idx] > 0:
                        continue
                    c = int(parent_counts[p_idx])
                    if c <= 0:
                        continue
                    # -------- band importance：继承 parent --------
                    band_total[child_idx] += parent_importance[p_idx] * c
                    band_counts[child_idx] += c

                    # -------- mean spectrum：继承 parent --------
                    mc = int(parent_mean_ct[p_idx])
                    if mc > 0:
                        mean_total[child_idx] += parent_mean_spectra[p_idx] * mc
                        mean_counts[child_idx] += mc
            else:
                log(f"Parent model not found for {parent_level}: {parent_model_path}")

    if inherit_missing and mean_total is not None:
        missing_mask = mean_counts == 0
        if missing_mask.any():
            full_train_subset = Subset(base_train_dataset, train_idx_all)
            full_test_subset = Subset(base_test_dataset, test_idx_all)
            full_loader = DataLoader(
                full_train_subset if heatmap_cfg.use_train_loader else full_test_subset,
                batch_size=config.batch_size,
                shuffle=False
            )
            global_mean, global_counts = compute_class_mean_spectrum(
                full_loader,
                device,
                num_classes=global_num_classes,
                head_index=head_index,
                max_per_class=heatmap_cfg.max_per_class,
                level_name=analysis_level,
                label_name_to_idx=global_name_to_idx,
                missing_tag=missing_tag,
            )
            for i in np.where(missing_mask)[0]:
                c = int(global_counts[i])
                if c <= 0:
                    continue
                mean_total[i] = global_mean[i] * c
                mean_counts[i] = c

        # Fill missing band importance by inheriting from parent-level per-parent models
        if band_total is not None:
            missing_band = band_counts == 0
            if missing_band.any():
                parent_level = full_dataset.get_parent_level(analysis_level)
                if parent_level is None:
                    log("Parent level not found; cannot inherit band importance.")
                else:
                    parent_models = runtime.ensure_parent_models(
                        parent_level,
                        full_dataset.parent_to_children,
                    )
                    if not parent_models:
                        log(f"No parent models for {parent_level}; cannot inherit band importance.")
                    else:
                        parent_level_idx = full_dataset.head_name_to_idx[parent_level]
                        parent_level_map = full_dataset.label_maps_by_level[parent_level_idx]
                        grand_parent = full_dataset.get_parent_level(parent_level)
                        if grand_parent is None:
                            log(f"{parent_level} has no parent; cannot select per-parent models.")
                        else:
                            grand_parent_idx = full_dataset.head_name_to_idx[grand_parent]
                            grand_parent_map = full_dataset.label_maps_by_level[grand_parent_idx]

                            class_to_parent_idx = {}
                            class_to_parent_level_id = {}
                            for sample_idx, hier in enumerate(full_dataset.hier_names):
                                class_name = hier.get(analysis_level)
                                parent_name = hier.get(parent_level)
                                gp_name = hier.get(grand_parent)
                                if class_name is None or parent_name is None or gp_name is None:
                                    continue
                                if class_name not in class_to_parent_idx:
                                    class_to_parent_idx[class_name] = grand_parent_map.get(gp_name)
                                    class_to_parent_level_id[class_name] = parent_level_map.get(parent_name)

                            parent_importance_cache = {}

                            for p_idx, entry in parent_models.items():
                                model_path = entry.get("model_path")
                                child_ids = entry.get("child_ids", [])
                                if model_path is None or not child_ids:
                                    continue
                                if not os.path.isabs(model_path):
                                    model_path = os.path.join(exp_dir, model_path)
                                if not os.path.exists(model_path):
                                    log(f"Parent model missing: {model_path}")
                                    continue

                                task = {"parent_idx": int(p_idx), "child_ids": child_ids}
                                parent_train_loader, parent_test_loader, _, _ = _build_task_loaders(
                                    task,
                                    config,
                                    full_dataset,
                                    parent_level,
                                    parent_level_idx,
                                    train_idx_all,
                                    test_idx_all,
                                    base_train_dataset,
                                    base_test_dataset,
                                )
                                parent_loader = (
                                    parent_train_loader if heatmap_cfg.use_train_loader else parent_test_loader
                                )

                                parent_model = runtime.get_parent_model(
                                    parent_level,
                                    int(p_idx),
                                    child_ids=child_ids,
                                    model_path=model_path,
                                )

                                parent_importance, _ = compute_class_band_importance_ig(
                                    parent_model,
                                    parent_loader,
                                    device,
                                    num_classes=len(child_ids),
                                    head_index=parent_level_idx,
                                    steps=heatmap_cfg.steps,
                                    num_batches=heatmap_cfg.num_batches,
                                    max_per_class=heatmap_cfg.max_per_class,
                                )
                                child_to_local = {int(cid): i for i, cid in enumerate(child_ids)}
                                parent_importance_cache[int(p_idx)] = (parent_importance, child_to_local)

                            for i in np.where(missing_band)[0]:
                                class_name = global_class_names[i]
                                p_idx = class_to_parent_idx.get(class_name)
                                parent_level_id = class_to_parent_level_id.get(class_name)
                                if p_idx is None or parent_level_id is None:
                                    continue
                                cached = parent_importance_cache.get(int(p_idx))
                                if cached is None:
                                    continue
                                parent_importance, child_to_local = cached
                                local_idx = child_to_local.get(int(parent_level_id))
                                if local_idx is None:
                                    continue
                                c = int(max(mean_counts[i], 1))
                                band_total[i] = parent_importance[local_idx] * c
                                band_counts[i] = c

    if weight_total == 0:
        log("No samples available for aggregate analysis.")
        log_file.close()
        return

    # ----- finalize aggregate channel/layer -----
    agg_channel = agg_channel / float(weight_total)
    agg_channel = agg_channel / (agg_channel.sum() + 1e-8)
    _plot_channel_importance(
        agg_channel,
        channel_names,
        os.path.join(fig_dir, "channel_importance_IG_aggregate.png")
    )
    log(f"Saved aggregate channel importance: {os.path.join(fig_dir, 'channel_importance_IG_aggregate.png')}")

    agg_layer = {k: v / float(weight_total) for k, v in agg_layer.items()}
    layer_total = sum(agg_layer.values())
    if layer_total > 0:
        agg_layer = {k: v / layer_total for k, v in agg_layer.items()}
    _plot_layer_importance(
        agg_layer,
        os.path.join(fig_dir, "layer_importance_aggregate.png")
    )
    log(f"Saved aggregate layer importance: {os.path.join(fig_dir, 'layer_importance_aggregate.png')}")

    # ----- finalize aggregate band importance -----
    valid_counts = np.maximum(band_counts[:, None], 1)
    band_avg = band_total / valid_counts
    mean_avg = mean_total / np.maximum(mean_counts[:, None], 1)

    wavenumbers = build_wavenumber_axis(band_avg.shape[1], config)
    bad_bands = _get_bad_bands(config)
    heatmap_path = os.path.join(fig_dir, "band_importance_heatmap_aggregate.png")
    plot_band_importance_heatmap(
        band_avg,
        band_counts,
        global_class_names,
        wavenumbers,
        heatmap_path,
        row_norm=heatmap_cfg.row_norm,
        mean_spectra=mean_avg,
        bad_bands=bad_bands,
    )
    log(f"Saved aggregate band importance heatmap: {heatmap_path}")

    topk_path = os.path.join(
        fig_dir, f"band_top{heatmap_cfg.topk_per_class}_per_class_aggregate.csv"
    )
    save_topk_bands_csv(
        band_avg,
        global_class_names,
        wavenumbers,
        top_k=heatmap_cfg.topk_per_class,
        save_path=topk_path,
        row_norm=heatmap_cfg.row_norm,
    )
    log(f"Saved aggregate band top-k CSV: {topk_path}")

    log("Note: Embedding plots are skipped in aggregate mode (different parent models).")
    log_file.close()

def run_single_analysis(
    exp_dir,
    config,
    full_dataset,
    analysis_level,
    head_index,
    task,
    train_idx_all,
    test_idx_all,
    base_train_dataset,
    base_test_dataset,
    runtime=None,
    device=None,
    heatmap_cfg=None,
):
    """单模型分析：Grad-CAM / IG / embedding / 波段热图"""
    heatmap_cfg = _ensure_heatmap_cfg(heatmap_cfg)
    parent_idx = task["parent_idx"]
    num_classes = task["num_classes"]
    class_names = task["class_names"]
    model_path = task["model_path"]
    tag = task["tag"]

    # ---------------- 输出目录 ----------------
    analysis_dir = os.path.join(exp_dir, f"{tag}_analysis")
    fig_dir = os.path.join(analysis_dir, "figures")
    log_dir = os.path.join(analysis_dir, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "analysis_log.txt")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + os.linesep)

    if parent_idx is None:
        log(f"Analysis target: {analysis_level}")
    else:
        log(f"Analysis target: {analysis_level} (parent={parent_idx})")

    train_loader, test_loader, _, _ = _build_task_loaders(
        task,
        config,
        full_dataset,
        analysis_level,
        head_index,
        train_idx_all,
        test_idx_all,
        base_train_dataset,
        base_test_dataset,
    )

    # ---------------- 模型 ----------------
    if device is None:
        use_cuda = (config.use_gpu and torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda else "cpu")
    if runtime is None:
        runtime = build_experiment_runtime(exp_dir, device, config=config)
    if not runtime.parent_to_children:
        runtime.parent_to_children = full_dataset.parent_to_children
    log(
        f"Using device: {device} (config.use_gpu={config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})"
    )

    if parent_idx is None:
        model = runtime.load_single_level_model(
            analysis_level,
            num_classes=num_classes,
        )
    else:
        model = runtime.get_parent_model(
            analysis_level,
            parent_idx,
            child_ids=task["child_ids"],
            model_path=model_path,
        )

    # ------------------------------------------------------------
    # 执行一次前向传播：
    # - 初始化模型内部状态
    # - 确保后续 Grad-CAM / hook 正常注册与触发
    # ------------------------------------------------------------
    sample_x, _, _ = next(iter(train_loader))
    sample_x = sample_x.to(device)
    _ = model(sample_x)

    # 使用同一轮 Integrated Gradients 归因，同时评估输入通道和类别波段的重要性
    log("")
    log("=== Computing input channel importance and band importance ===")
    channel_names = [f"{config.norm_method}"]
    if config.smooth_use:
        channel_names.append("smooth")
    if config.d1_use:
        channel_names.append("d1")

    log(f"Using channel names: {channel_names}")

    # 波段热图使用哪一侧数据，这里也同步决定通道重要性的统计样本
    heatmap_loader = train_loader if heatmap_cfg.use_train_loader else test_loader
    inherit_missing = getattr(config, "inherit_missing_levels", False)

    if inherit_missing:
        missing_tag = getattr(full_dataset, "MISSING_TAG", "__missing__")
        heatmap_class_names = _effective_label_names(full_dataset, analysis_level, missing_tag)
        heatmap_name_to_idx = {n: i for i, n in enumerate(heatmap_class_names)}
        heatmap_num_classes = len(heatmap_class_names)
    else:
        missing_tag = getattr(full_dataset, "MISSING_TAG", "__missing__")
        heatmap_class_names = class_names
        heatmap_name_to_idx = None
        heatmap_num_classes = num_classes

    ig_batches = compute_ig_batches(
        model,
        heatmap_loader,
        device,
        steps=heatmap_cfg.steps,
        num_batches=heatmap_cfg.num_batches,
        head_index=head_index,
        num_classes=heatmap_num_classes,
        max_per_class=heatmap_cfg.max_per_class,
        level_name=analysis_level if inherit_missing else None,
        label_name_to_idx=heatmap_name_to_idx,
        missing_tag=missing_tag,
    )
    importance = compute_channel_importance_from_ig(ig_batches)
    band_importance, counts = compute_band_importance_from_ig(
        ig_batches,
        num_classes=heatmap_num_classes,
        max_per_class=heatmap_cfg.max_per_class,
    )

    _plot_channel_importance(
        importance,
        channel_names,
        os.path.join(fig_dir, "channel_importance_IG.png")
    )

    log(f"Input channel importance: {importance}")

    # 多层 Grad-CAM 级联 Layer-wise Importance 分析
    log("")
    log("=== Running Multi-layer Grad-CAM Analysis ===")
    analyzable, groups = collect_analyzable_layers(model)
    analyzer = LayerGradCAMAnalyzer(model, device)

    for name, layer in analyzable.items():
        analyzer.register_layer(name, layer)

    layer_scores = analyzer.run(
        train_loader,
        save_dir=fig_dir,
        num_batches=3,
        head_index=head_index
    )

    merged_scores = merge_scores_by_group(layer_scores, groups)

    log("")
    log("=== Layer Importance (merged by stage) ===")
    for k, v in merged_scores.items():
        log(f"{k:30s}: {v:.4f}")

    # SE 模块分析
    if config.se_use:
        log_seblock_summary(runtime, model_path, log)

    # Embedding 可视化
    embed_method = str(config.embedding_method).lower()
    embed_tag = embed_method.replace("-", "").replace("_", "")

    embed_levels = [analysis_level]
    if full_dataset.head_names:
        top_level = full_dataset.head_names[0]
        if top_level not in embed_levels:
            embed_levels.append(top_level)

    if inherit_missing:
        feats, hier_labels, label_names = collect_embeddings_train_test(
            model,
            train_loader,
            test_loader,
            device,
            dataset=full_dataset,
            level_names=embed_levels,
            return_label_names=True,
        )
    else:
        feats, hier_labels = collect_embeddings_train_test(
            model,
            train_loader,
            test_loader,
            device,
            dataset=full_dataset,
            level_names=embed_levels,
        )
        label_names = None

    plot_embedding_hierarchical(
        feats,
        hier_labels=hier_labels,
        save_path=os.path.join(fig_dir, f"{embed_tag}_hier_train_test.png"),
        method=embed_method,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        tsne_perplexity=config.tsne_perplexity,
        tsne_iter=config.tsne_iter,
        label_names=label_names,
    )

    # 波段重要性热图（按类别）
    log("")
    log("=== Computing band importance heatmap ===")

    mean_spectra, _ = compute_class_mean_spectrum(
        heatmap_loader,
        device,
        num_classes=heatmap_num_classes,
        head_index=head_index,
        max_per_class=heatmap_cfg.max_per_class,
        level_name=analysis_level if inherit_missing else None,
        label_name_to_idx=heatmap_name_to_idx,
        missing_tag=missing_tag,
    )

    wavenumbers = build_wavenumber_axis(band_importance.shape[1], config)
    bad_bands = _get_bad_bands(config)
    heatmap_path = os.path.join(fig_dir, "band_importance_heatmap.png")
    plot_band_importance_heatmap(
        band_importance,
        counts,
        heatmap_class_names,
        wavenumbers,
        heatmap_path,
        row_norm=heatmap_cfg.row_norm,
        mean_spectra=mean_spectra,
        bad_bands=bad_bands,
    )
    log(f"Saved band importance heatmap: {heatmap_path}")

    topk_path = os.path.join(
        fig_dir, f"band_top{heatmap_cfg.topk_per_class}_per_class.csv"
    )
    save_topk_bands_csv(
        band_importance,
        heatmap_class_names,
        wavenumbers,
        top_k=heatmap_cfg.topk_per_class,
        save_path=topk_path,
        row_norm=heatmap_cfg.row_norm,
    )
    log(f"Saved band top-k CSV: {topk_path}")

    log_file.close()

def build_analysis_context(
    exp_dir,
    analysis_level,
    parent_idx,
    inherit_missing_levels=False,
):
    """构建通用上下文（数据集 / 划分 / 任务列表）"""
    exp_dir, config = load_experiment_with_dataset(exp_dir)

    config.inherit_missing_levels = bool(inherit_missing_levels)

    full_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config
    )

    analysis_level = full_dataset._resolve_level_name(
        analysis_level,
        field_name="analysis_level",
    )
    head_index = full_dataset.head_name_to_idx[analysis_level]

    meta = load_hierarchy_meta(exp_dir) or {}
    device = torch.device(
        "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
    )
    runtime = build_experiment_runtime(exp_dir, device, config=config, meta=meta)
    if not runtime.parent_to_children:
        runtime.parent_to_children = full_dataset.parent_to_children
    runtime.ensure_parent_models(analysis_level, runtime.parent_to_children)
    level_models = runtime.build_level_model_paths([analysis_level])
    parent_models = runtime.parent_models

    tasks, auto_all = _build_analysis_tasks(
        exp_dir,
        analysis_level,
        head_index,
        full_dataset,
        level_models,
        parent_models,
        parent_idx,
    )

    split = load_split_files(full_dataset, exp_dir)
    if split is not None:
        train_idx, test_idx = split
    else:
        split_level = getattr(config, "split_level", None) or "leaf"
        train_idx, test_idx = split_by_lowest_level_ratio(
            full_dataset,
            lowest_level=split_level,
            train_ratio=config.train_split,
            seed=config.seed
        )
    train_idx_all = np.array(sorted(train_idx))
    test_idx_all = np.array(sorted(test_idx))

    base_train_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config
    )
    base_test_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config
    )

    return {
        "exp_dir": exp_dir,
        "config": config,
        "full_dataset": full_dataset,
        "analysis_level": analysis_level,
        "head_index": head_index,
        "tasks": tasks,
        "train_idx_all": train_idx_all,
        "test_idx_all": test_idx_all,
        "base_train_dataset": base_train_dataset,
        "base_test_dataset": base_test_dataset,
        "auto_all": auto_all,
        "meta": meta,
        "device": device,
        "runtime": runtime,
    }

def run_analysis_pipeline(overrides=None, heatmap_cfg=None):
    """统一分析入口：按 mode 切换单模型或聚合分析"""
    overrides = overrides or AnalysisOverrides()
    if not overrides.exp_dir:
        raise ValueError("analyze 需要显式传入 exp_dir")

    mode = str(overrides.mode).lower()
    if mode not in ("single", "aggregate"):
        raise ValueError(f"未知分析模式：{overrides.mode}，可选值为：single / aggregate")

    ctx = build_analysis_context(
        overrides.exp_dir,
        overrides.analysis_level,
        overrides.parent_idx,
        inherit_missing_levels=overrides.inherit_missing_levels,
    )

    if mode == "single":
        if ctx["auto_all"]:
            print(
                f"No global model for {ctx['analysis_level']}; "
                f"running per-parent analysis for {len(ctx['tasks'])} parents."
            )
        elif len(ctx["tasks"]) > 1:
            print(f"Running per-parent analysis for {len(ctx['tasks'])} parents.")

        for task in ctx["tasks"]:
            run_single_analysis(
                ctx["exp_dir"],
                ctx["config"],
                ctx["full_dataset"],
                ctx["analysis_level"],
                ctx["head_index"],
                task,
                ctx["train_idx_all"],
                ctx["test_idx_all"],
                ctx["base_train_dataset"],
                ctx["base_test_dataset"],
                runtime=ctx["runtime"],
                device=ctx["device"],
                heatmap_cfg=heatmap_cfg,
            )
        return

    parent_tasks = [t for t in ctx["tasks"] if t["parent_idx"] is not None]
    if not parent_tasks:
        if overrides.fallback_to_single:
            print("Aggregate fallback: no parent models found; running single-model analysis.")
            if ctx["auto_all"]:
                print(
                    f"No global model for {ctx['analysis_level']}; "
                    f"running per-parent analysis for {len(ctx['tasks'])} parents."
                )
            elif len(ctx["tasks"]) > 1:
                print(f"Running per-parent analysis for {len(ctx['tasks'])} parents.")

            for task in ctx["tasks"]:
                run_single_analysis(
                    ctx["exp_dir"],
                    ctx["config"],
                    ctx["full_dataset"],
                    ctx["analysis_level"],
                    ctx["head_index"],
                    task,
                    ctx["train_idx_all"],
                    ctx["test_idx_all"],
                    ctx["base_train_dataset"],
                    ctx["base_test_dataset"],
                    runtime=ctx["runtime"],
                    device=ctx["device"],
                    heatmap_cfg=heatmap_cfg,
                )
        else:
            print("Aggregate analysis skipped: no parent models found.")
        return

    run_aggregate_analysis(
        ctx["exp_dir"],
        ctx["config"],
        ctx["full_dataset"],
        ctx["analysis_level"],
        ctx["head_index"],
        parent_tasks,
        ctx["train_idx_all"],
        ctx["test_idx_all"],
        ctx["base_train_dataset"],
        ctx["base_test_dataset"],
        runtime=ctx["runtime"],
        device=ctx["device"],
        meta=ctx["meta"],
        heatmap_cfg=heatmap_cfg,
    )
