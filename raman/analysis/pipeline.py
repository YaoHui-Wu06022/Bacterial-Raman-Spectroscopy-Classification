import os
from dataclasses import dataclass
import numpy as np
import torch

from raman.data import RamanDataset
from raman.eval.experiment import (
    load_experiment_with_dataset,
    load_hierarchy_meta,
)
from raman.eval.runtime import build_experiment_runtime
from raman.training import (
    split_by_lowest_level_ratio,
    load_split_files,
)
from .aggregate import run_aggregate_analysis
from .level import run_level_analysis
from .tasks import build_analysis_tasks


@dataclass
class HeatmapConfig:
    # 波段重要性热图（IG）相关配置
    num_batches: int = 10          # 采样多少个 batch
    steps: int = 32                # IG 积分步数
    max_per_class: int = 50        # 每类最多样本数
    row_norm: str = "max"          # 行归一化方式：max / sum / none
    use_train_loader: bool = True  # True=训练集，False=测试集
    topk_per_class: int = 5        # 每类 top-k 波段导出

@dataclass
class AnalysisOverrides:
    """统一收拢分析入口的覆盖项"""

    exp_dir: str | None = None
    mode: str = "single"
    analysis_level: str | None = None
    parent_idx: int | str | None = None
    inherit_missing_levels: bool = False
    fallback_to_single: bool = True

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

    tasks, auto_all = build_analysis_tasks(
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
    heatmap_cfg = heatmap_cfg if heatmap_cfg is not None else HeatmapConfig()

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
            run_level_analysis(
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
                run_level_analysis(
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
