import os
from dataclasses import dataclass

import numpy as np
import torch

from raman.data import RamanDataset
from raman.eval.experiment import (
    collect_used_runs,
    load_experiment_context_with_dataset,
    load_hierarchy_meta,
    resolve_mode_result_dir,
    resolve_mode_result_root,
    resolve_split_dir,
    validate_parent_split_hashes,
    write_used_runs,
)
from raman.eval.runtime import build_experiment_runtime
from raman.training import load_split_files

from .aggregate import run_aggregate_analysis
from .level import run_level_analysis
from .tasks import build_analysis_tasks, normalize_parent_idx


@dataclass
class HeatmapConfig:
    """解释性热图的采样参数"""

    num_batches: int = 10
    steps: int = 32
    max_per_class: int = 50
    row_norm: str = "max"
    use_train_loader: bool = True
    topk_per_class: int = 5


def _resolve_level_order(dataset, target_level):
    """解析目标层级，并返回从 level_1 到目标层的顺序"""
    target_level = dataset._resolve_level_name(target_level, field_name="target_level")
    if target_level not in dataset.level_names:
        raise ValueError(
            f"未知 target_level: {target_level}，可选值：{dataset.level_names}"
        )
    stop_idx = dataset.level_names.index(target_level) + 1
    return target_level, list(dataset.level_names[:stop_idx])


def _load_analysis_context(exp_dir, target_level=None, inherit_missing_levels=False):
    """加载分析入口共用的实验配置、数据集、split 和 runtime"""
    input_context, config = load_experiment_context_with_dataset(exp_dir)
    config.inherit_missing_levels = bool(inherit_missing_levels)

    full_dataset = RamanDataset(config.dataset_root, augment=False, config=config)
    target_level = target_level or input_context.input_level
    target_level, level_order = _resolve_level_order(full_dataset, target_level)
    head_index = full_dataset.head_name_to_idx[target_level]

    split_dir = resolve_split_dir(input_context.exp_dir)
    split = load_split_files(full_dataset, split_dir) if split_dir else None
    if split is None:
        raise FileNotFoundError(
            f"实验根缺少 train_split.json/val_split.json，无法运行分析：{input_context.exp_dir}"
        )
    train_idx, val_idx = split

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
        runtime.parent_to_children = full_dataset.parent_to_children

    base_train_dataset = RamanDataset(config.dataset_root, augment=False, config=config)
    base_val_dataset = RamanDataset(config.dataset_root, augment=False, config=config)

    return {
        "input_context": input_context,
        "config": config,
        "full_dataset": full_dataset,
        "target_level": target_level,
        "level_order": level_order,
        "head_index": head_index,
        "train_idx": np.array(sorted(train_idx)),
        "val_idx": np.array(sorted(val_idx)),
        "base_train_dataset": base_train_dataset,
        "base_val_dataset": base_val_dataset,
        "device": device,
        "runtime": runtime,
        "meta": meta,
    }


def _prepare_target_models(ctx, levels_for_runtime):
    """预解析需要用到的全局模型和目标层 parent 模型"""
    runtime = ctx["runtime"]
    target_level = ctx["target_level"]
    runtime.build_level_model_paths(levels_for_runtime)
    runtime.ensure_parent_models(target_level, runtime.parent_to_children)
    return runtime


def _build_tasks(ctx, parent_idx_setting):
    """根据全局模型或 parent 子模型构造分析任务"""
    runtime = ctx["runtime"]
    return build_analysis_tasks(
        ctx["input_context"].exp_dir,
        ctx["target_level"],
        ctx["head_index"],
        ctx["full_dataset"],
        runtime.level_model_paths,
        runtime.parent_models,
        parent_idx_setting,
    )


def _run_single_task(ctx, task, analysis_dir, heatmap_cfg):
    """运行单个模型的 Grad-CAM、IG 和 embedding 分析"""
    run_level_analysis(
        ctx["input_context"].exp_dir,
        ctx["config"],
        ctx["full_dataset"],
        ctx["target_level"],
        ctx["head_index"],
        task,
        ctx["train_idx"],
        ctx["val_idx"],
        ctx["base_train_dataset"],
        ctx["base_val_dataset"],
        runtime=ctx["runtime"],
        device=ctx["device"],
        heatmap_cfg=heatmap_cfg,
        analysis_dir=os.fspath(analysis_dir),
    )
    return os.fspath(analysis_dir)


def _run_aggregate_tasks(ctx, tasks, analysis_dir, heatmap_cfg):
    """运行多个 parent 子模型的聚合分析"""
    run_aggregate_analysis(
        ctx["input_context"].exp_dir,
        ctx["config"],
        ctx["full_dataset"],
        ctx["target_level"],
        ctx["head_index"],
        tasks,
        ctx["train_idx"],
        ctx["val_idx"],
        ctx["base_train_dataset"],
        ctx["base_val_dataset"],
        runtime=ctx["runtime"],
        device=ctx["device"],
        meta=ctx["meta"],
        heatmap_cfg=heatmap_cfg,
        analysis_dir=os.fspath(analysis_dir),
    )
    return os.fspath(analysis_dir)


def _run_target_analysis(ctx, analysis_dir, heatmap_cfg, parent_idx_setting=None):
    """按目标层模型形态选择单模型分析或 parent 聚合分析"""
    parent_idx_setting = normalize_parent_idx(parent_idx_setting)
    if parent_idx_setting == "all":
        parent_entries = ctx["runtime"].parent_models.get(ctx["target_level"], {})
        has_parent_model = any(
            entry.get("model_path") is not None
            for entry in parent_entries.values()
        )
        # 顶层、无 parent 条目或全是单子类直通时，回退为全局单模型分析
        if (
            ctx["full_dataset"].get_parent_level(ctx["target_level"]) is None
            or not parent_entries
            or not has_parent_model
        ):
            parent_idx_setting = None
    tasks, _ = _build_tasks(ctx, parent_idx_setting)
    if not tasks:
        raise ValueError(f"没有可分析的模型：level={ctx['target_level']}")

    if len(tasks) == 1:
        return _run_single_task(ctx, tasks[0], analysis_dir, heatmap_cfg)
    return _run_aggregate_tasks(ctx, tasks, analysis_dir, heatmap_cfg)


def run_analysis_single_model(
    run_dir,
    level=None,
    *,
    parent_idx=None,
    inherit_missing_levels=False,
    heatmap_cfg=None,
):
    """单测入口：只分析传入的 run_* 模型，结果写入该 run 下"""
    heatmap_cfg = heatmap_cfg or HeatmapConfig()
    ctx = _load_analysis_context(
        run_dir,
        target_level=level,
        inherit_missing_levels=inherit_missing_levels,
    )
    input_context = ctx["input_context"]
    if not input_context.is_single_run:
        raise ValueError("run_analysis_single_model 必须传入具体 run_* 或 best/run_* 目录")

    requested_parent = normalize_parent_idx(parent_idx)
    if requested_parent == "all":
        raise ValueError("单测入口只允许分析一个模型，parent_idx 不能为 'all'")
    if input_context.input_parent_idx is not None:
        if requested_parent is not None and requested_parent != input_context.input_parent_idx:
            raise ValueError(
                "parent_idx 与 run_* 所属 parent 不一致："
                f"{requested_parent} != {input_context.input_parent_idx}"
            )
        requested_parent = input_context.input_parent_idx

    _prepare_target_models(ctx, [ctx["target_level"]])
    analysis_dir = os.path.join(input_context.input_run_dir, "analysis_result")
    return _run_target_analysis(
        ctx,
        analysis_dir,
        heatmap_cfg,
        parent_idx_setting=requested_parent,
    )


def run_analysis_level_only(
    exp_dir,
    target_level,
    *,
    parent_idx="all",
    inherit_missing_levels=False,
    heatmap_cfg=None,
):
    """单层多模型入口：只分析目标层模型，parent 子模型存在时做聚合"""
    heatmap_cfg = heatmap_cfg or HeatmapConfig()
    ctx = _load_analysis_context(
        exp_dir,
        target_level=target_level,
        inherit_missing_levels=inherit_missing_levels,
    )
    _prepare_target_models(ctx, [ctx["target_level"]])
    validate_parent_split_hashes(
        ctx["input_context"].exp_dir,
        ctx["target_level"],
        ctx["runtime"].parent_models.get(ctx["target_level"], {}),
    )

    result_root = resolve_mode_result_root(
        ctx["input_context"].exp_dir,
        ctx["target_level"],
        "level_only",
    )
    analysis_dir = resolve_mode_result_dir(
        ctx["input_context"].exp_dir,
        "analysis",
        ctx["target_level"],
        "level_only",
    )
    out = _run_target_analysis(
        ctx,
        analysis_dir,
        heatmap_cfg,
        parent_idx_setting=parent_idx,
    )

    used_runs = collect_used_runs(
        ctx["input_context"].exp_dir,
        ctx["runtime"],
        level_order=[ctx["target_level"]],
        target_level=ctx["target_level"],
    )
    write_used_runs(
        result_root,
        mode="level_only",
        target_level=ctx["target_level"],
        runs=used_runs,
    )
    return out


def run_analysis_cascade(
    exp_dir,
    target_level,
    *,
    parent_idx="all",
    inherit_missing_levels=False,
    heatmap_cfg=None,
):
    """多层多模型入口：按级联结果目录记录目标层分析结果"""
    heatmap_cfg = heatmap_cfg or HeatmapConfig()
    ctx = _load_analysis_context(
        exp_dir,
        target_level=target_level,
        inherit_missing_levels=inherit_missing_levels,
    )
    _prepare_target_models(ctx, ctx["level_order"])
    for level_name in ctx["level_order"]:
        ctx["runtime"].ensure_parent_models(level_name, ctx["runtime"].parent_to_children)
        validate_parent_split_hashes(
            ctx["input_context"].exp_dir,
            level_name,
            ctx["runtime"].parent_models.get(level_name, {}),
        )

    result_root = resolve_mode_result_root(
        ctx["input_context"].exp_dir,
        ctx["target_level"],
        "cascade",
    )
    analysis_dir = resolve_mode_result_dir(
        ctx["input_context"].exp_dir,
        "analysis",
        ctx["target_level"],
        "cascade",
    )
    out = _run_target_analysis(
        ctx,
        analysis_dir,
        heatmap_cfg,
        parent_idx_setting=parent_idx,
    )

    used_runs = collect_used_runs(
        ctx["input_context"].exp_dir,
        ctx["runtime"],
        level_order=ctx["level_order"],
        target_level=ctx["target_level"],
    )
    write_used_runs(
        result_root,
        mode="cascade",
        target_level=ctx["target_level"],
        runs=used_runs,
    )
    return out
