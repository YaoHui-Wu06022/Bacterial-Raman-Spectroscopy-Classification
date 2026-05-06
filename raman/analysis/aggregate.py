import os

import numpy as np
import torch

from raman.eval.experiment import load_hierarchy_meta
from raman.eval.runtime import build_experiment_runtime

from .gradcam import (
    LayerGradCAMAnalyzer,
    _plot_layer_importance,
    collect_analyzable_layers,
    merge_scores_by_group,
)
from .ig import (
    _effective_label_names,
    _get_bad_bands,
    _plot_channel_importance,
    build_wavenumber_axis,
    compute_band_importance_from_ig,
    compute_channel_importance_from_ig,
    compute_class_band_importance_ig,
    compute_class_mean_spectrum,
    compute_ig_batches,
    plot_band_importance_heatmap,
    save_topk_bands_csv,
)
from .tasks import build_task_loaders


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

        train_loader, test_loader, train_subset, _ = build_task_loaders(
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
                parent_train_loader, parent_test_loader, _, _ = build_task_loaders(
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
                                parent_train_loader, parent_test_loader, _, _ = build_task_loaders(
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

