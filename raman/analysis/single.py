import os

import torch

from raman.eval.runtime import build_experiment_runtime

from .embedding import collect_embeddings_train_test, plot_embedding_hierarchical
from .gradcam import (
    LayerGradCAMAnalyzer,
    _plot_layer_importance,
    collect_analyzable_layers,
    merge_scores_by_group,
)
from .ig import (
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
from .se import log_seblock_summary
from .tasks import build_task_loaders


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

    train_loader, test_loader, _, _ = build_task_loaders(
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

