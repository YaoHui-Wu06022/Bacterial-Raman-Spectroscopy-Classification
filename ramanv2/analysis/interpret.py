"""模型归因与 embedding 分析的任务编排。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ramanv2.analysis.attribution_report import save_task_reports, write_aggregate_reports
from ramanv2.analysis.embedding import write_train_val_umap
from ramanv2.analysis.integrated_gradients import (
    collect_task_inputs,
    compute_integrated_gradients,
    select_balanced_task_sample_indices,
)
from ramanv2.analysis.layer_attribution import (
    compute_layer_attribution,
    merge_layer_attribution_scores,
)
from ramanv2.core.run_context import (
    ANALYSIS_LOG_FILE_NAME,
    ANALYSIS_RESULT_DIR_NAME,
    UMAP_FIGURE_NAME,
)
from ramanv2.evaluation.context import (
    EvaluationContext,
    load_evaluation_context,
    resolve_level_name,
    resolve_run_entry,
)
from ramanv2.evaluation.level_summary import (
    LEVEL_SUMMARY_DIR_NAME,
    assert_multi_model_level,
)
from ramanv2.modeling.predictor import load_predictor
from ramanv2.modeling.run_manifest import has_model_path, resolve_entry_path
from ramanv2.spectra.axis import expected_wavenumbers


@dataclass(frozen=True)
class AnalysisTask:
    """一个模型槽位的分析范围。"""

    level_name: str
    parent_id: int | None
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    slot_dir: str
    entry: Mapping[str, Any]
    train_indices: np.ndarray
    validation_indices: np.ndarray


def interpret_slot(
    source_dir: str,
    level_name: str,
    device: str | None = None,
    attribution_batch_count: int | None = None,
    umap_enable: bool = True,
) -> Path:
    """分析一个模型槽位，结果写入槽位 analysis_result。"""
    context = load_evaluation_context(source_dir)
    level = resolve_level_name(context, level_name)
    task = build_slot_task(context, level)
    return _run_slot_analysis(
        context,
        task,
        device,
        attribution_batch_count,
        umap_enable,
    )


def umap_slot(
    source_dir: str,
    level_name: str,
    device: str | None = None,
) -> Path:
    """只为一个模型槽位重新生成 Train/Val 联合 UMAP。"""
    context = load_evaluation_context(source_dir)
    level = resolve_level_name(context, level_name)
    task = build_slot_task(context, level)
    target_device = _resolve_device(device)
    predictor = load_predictor(task.slot_dir, target_device, task.level_name)
    model = predictor.load_model(task.level_name, task.entry, task.parent_id)
    output_path = (
        Path(task.slot_dir)
        / ANALYSIS_RESULT_DIR_NAME
        / "figures"
        / UMAP_FIGURE_NAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_task_umap(context, task, model, target_device, output_path)
    return output_path


def interpret_level(
    source_dir: str,
    level_name: str,
    device: str | None = None,
    attribution_batch_count: int | None = None,
    umap_enable: bool = True,
) -> Path:
    """聚合一个层级多个模型的归因，写入 level_summary/analysis_result。"""
    assert_multi_model_level(source_dir, level_name, "层级归因汇总")
    context = load_evaluation_context(source_dir)
    level = resolve_level_name(context, level_name)
    tasks = build_level_tasks(context, level)
    return _run_level_analysis(
        context,
        tasks,
        device,
        attribution_batch_count,
        umap_enable,
    )


def build_slot_task(context: EvaluationContext, level_name: str) -> AnalysisTask:
    """从明确模型槽位构建唯一的分析任务。"""
    run_entry = resolve_run_entry(context)
    if run_entry.level_name != level_name:
        raise ValueError(
            f"模型槽位属于 {run_entry.level_name}，不能按 {level_name} 分析"
        )
    return _build_task(
        context,
        level_name,
        run_entry.parent_id,
        run_entry.values,
        run_entry.run_dir,
    )


def build_level_tasks(context: EvaluationContext, level_name: str) -> list[AnalysisTask]:
    """构建一个层级下所有已完成训练模型的分析任务。"""
    global_entry = (context.meta.get("level_models") or {}).get(level_name)
    if has_model_path(global_entry):
        slot_dir = resolve_entry_path(context.experiment_dir, global_entry, "run_dir")
        return [_build_task(context, level_name, None, global_entry, slot_dir)]

    parent_to_children = (
        (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    )
    entries = (context.meta.get("parent_models") or {}).get(level_name) or {}
    tasks: list[AnalysisTask] = []
    for parent_text in sorted(parent_to_children, key=lambda item: int(item)):
        parent_id = int(parent_text)
        children = parent_to_children.get(parent_text) or []
        entry = entries.get(str(parent_id))
        if not has_model_path(entry):
            if len(children) > 1:
                raise FileNotFoundError(
                    f"{level_name} 父类 {parent_id} 缺少子模型，无法生成层级归因"
                )
            continue
        slot_dir = resolve_entry_path(context.experiment_dir, entry, "run_dir")
        tasks.append(_build_task(context, level_name, parent_id, entry, slot_dir))
    if not tasks:
        raise ValueError(f"没有可分析的模型：level={level_name}")
    return tasks


def _build_task(
    context,
    level_name,
    parent_id,
    entry,
    slot_dir,
) -> AnalysisTask:
    """按任务类别空间筛选固定 train/val 索引。"""
    class_ids = tuple(
        range(context.dataset_index.num_classes_by_level[level_name])
        if parent_id is None
        else (int(item) for item in entry.get("child_ids") or [])
    )
    if not class_ids:
        raise ValueError(f"{level_name} 缺少可分析类别")
    level_index = context.dataset_index.head_name_to_idx[level_name]
    parent_index = (
        None
        if parent_id is None
        else context.dataset_index.head_name_to_idx[
            context.dataset_index.get_parent_level(level_name)
        ]
    )

    def select(indices):
        labels = context.dataset_index.level_labels[indices]
        mask = np.isin(labels[:, level_index], class_ids)
        if parent_index is not None:
            mask &= labels[:, parent_index] == parent_id
        return indices[mask]

    all_names = context.dataset_index.get_class_names(level_name)
    return AnalysisTask(
        level_name=level_name,
        parent_id=parent_id,
        class_ids=class_ids,
        class_names=tuple(all_names[index] for index in class_ids),
        slot_dir=str(slot_dir),
        entry=entry,
        train_indices=select(context.train_indices),
        validation_indices=select(context.validation_indices),
    )


def _run_slot_analysis(
    context,
    task,
    device_value: str | None,
    attribution_batch_count: int | None,
    umap_enable: bool,
) -> Path:
    """执行单模型分析并写入槽位 analysis_result。"""
    device = _resolve_device(device_value)
    output_dir = Path(task.slot_dir) / ANALYSIS_RESULT_DIR_NAME
    figure_dir, log_dir = _prepare_analysis_dirs(output_dir)
    summaries = [
        _analyze_task(
            context,
            task,
            device,
            figure_dir,
            collect_embedding=umap_enable,
            write_reports_enable=True,
            attribution_batch_count=attribution_batch_count,
        )
    ]
    write_aggregate_reports(
        summaries,
        figure_dir,
        _resolve_channel_names(context.input_spec),
        expected_wavenumbers(context.config.input),
        context.config.input.bad_bands,
        context.config.analysis.row_norm,
        context.config.analysis.separate_class_plots_use,
    )
    se_lines = write_se_summary([task], log_dir / "se_summary.txt")
    _write_analysis_log(
        summaries,
        log_dir / ANALYSIS_LOG_FILE_NAME,
        tasks=[task],
        mode="slot",
        level_name=task.level_name,
        device=device,
        execution_config=context.config.execution,
        channel_names=_resolve_channel_names(context.input_spec),
        se_lines=se_lines,
        figure_dir=figure_dir,
        separate_class_plots_enable=context.config.analysis.separate_class_plots_use,
    )
    return output_dir


def _run_level_analysis(
    context,
    tasks,
    device_value: str | None,
    attribution_batch_count: int | None,
    umap_enable: bool,
) -> Path:
    """聚合一个层级多个模型的归因并写入 level_summary/analysis_result。"""
    device = _resolve_device(device_value)
    level_name = tasks[0].level_name
    output_dir = (
        context.experiment_dir
        / level_name
        / LEVEL_SUMMARY_DIR_NAME
        / ANALYSIS_RESULT_DIR_NAME
    )
    figure_dir, log_dir = _prepare_analysis_dirs(output_dir)
    task_report_enable = len(tasks) == 1
    summaries = [
        _analyze_task(
            context,
            task,
            device,
            figure_dir,
            collect_embedding=umap_enable and len(tasks) == 1,
            write_reports_enable=task_report_enable,
            attribution_batch_count=attribution_batch_count,
        )
        for task in tasks
    ]
    if context.config.analysis.inherit_missing_levels_use:
        summaries.extend(
            _inherit_single_child_summaries(
                context,
                level_name,
                device,
                figure_dir,
                attribution_batch_count,
            )
        )
    write_aggregate_reports(
        summaries,
        figure_dir,
        _resolve_channel_names(context.input_spec),
        expected_wavenumbers(context.config.input),
        context.config.input.bad_bands,
        context.config.analysis.row_norm,
        context.config.analysis.separate_class_plots_use,
    )
    se_lines = write_se_summary(tasks, log_dir / "se_summary.txt")
    _write_analysis_log(
        summaries,
        log_dir / ANALYSIS_LOG_FILE_NAME,
        tasks=tasks,
        mode="level",
        level_name=level_name,
        device=device,
        execution_config=context.config.execution,
        channel_names=_resolve_channel_names(context.input_spec),
        se_lines=se_lines,
        figure_dir=figure_dir,
        separate_class_plots_enable=context.config.analysis.separate_class_plots_use,
    )
    return output_dir


def _analyze_task(
    context,
    task,
    device,
    figure_dir: Path,
    collect_embedding: bool,
    write_reports_enable: bool,
    attribution_batch_count: int | None = None,
) -> dict:
    """执行一个模型任务的 IG、层归因、图表和可选 UMAP。"""
    predictor = load_predictor(task.slot_dir, device, task.level_name)
    model = predictor.load_model(task.level_name, task.entry, task.parent_id)
    config = context.config.analysis
    sample_count = len(
        task.train_indices if config.attribution_split == "train" else task.validation_indices
    )
    batch_size = context.config.training.batch_size
    selected_batch_count = (
        config.attribution_batch_count
        if attribution_batch_count is None
        else int(attribution_batch_count)
    )
    limit = min(sample_count, max(1, selected_batch_count) * batch_size)
    selected_indices = select_balanced_task_sample_indices(
        context,
        task,
        config.attribution_split,
        limit,
        config.max_per_class,
    )
    inputs, labels = collect_task_inputs(
        context,
        task,
        config.attribution_split,
        device,
        selected_indices,
    )
    result = compute_integrated_gradients(
        model,
        inputs,
        labels,
        batch_size,
        config.ig_steps,
        config.max_per_class,
        len(task.class_ids),
    )
    layer_scores = compute_layer_attribution(model, inputs[:batch_size], labels[:batch_size])
    tag = (
        task.level_name
        if task.parent_id is None
        else f"{task.level_name}_parent_{task.parent_id}"
    )
    if write_reports_enable:
        save_task_reports(
            figure_dir,
            task.class_names,
            result.channel_importance,
            result.band_importance,
            result.mean_spectra,
            layer_scores,
            _resolve_channel_names(context.input_spec),
            expected_wavenumbers(context.config.input),
            context.config.input.bad_bands,
            config.row_norm,
            config.separate_class_plots_use,
        )
    if collect_embedding:
        _save_task_umap(context, task, model, device, figure_dir / UMAP_FIGURE_NAME)
    return {
        "class_ids": task.class_ids,
        "class_names": task.class_names,
        "channel": result.channel_importance,
        "band": result.band_importance,
        "counts": result.sample_counts,
        "mean": result.mean_spectra,
        "mean_counts": result.mean_counts,
        "weight": result.sample_count,
        "layer": layer_scores,
        "log": f"{tag}: samples={result.sample_count}, classes={len(task.class_ids)}",
    }


def _inherit_single_child_summaries(
    context,
    level_name: str,
    device,
    figure_dir: Path,
    attribution_batch_count: int | None,
) -> list[dict]:
    """将单子类父分支的父层模型归因映射到唯一子类。"""
    parent_level = context.dataset_index.get_parent_level(level_name)
    mapping = (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    single_children = {
        int(parent_id): int(children[0])
        for parent_id, children in mapping.items()
        if len(children) == 1
    }
    if parent_level is None or not single_children:
        return []
    parent_tasks = build_level_tasks(context, parent_level)
    summaries_by_slot: dict[str, dict] = {}
    class_names = context.dataset_index.get_class_names(level_name)
    inherited: list[dict] = []
    for parent_id, child_id in single_children.items():
        parent_task = next(
            (task for task in parent_tasks if parent_id in task.class_ids),
            None,
        )
        if parent_task is None:
            raise FileNotFoundError(
                f"层级归因汇总需要 {parent_level} 中包含父类 {parent_id} 的模型"
            )
        parent_summary = summaries_by_slot.get(parent_task.slot_dir)
        if parent_summary is None:
            parent_summary = _analyze_task(
                context,
                parent_task,
                device,
                figure_dir,
                collect_embedding=False,
                write_reports_enable=False,
                attribution_batch_count=attribution_batch_count,
            )
            summaries_by_slot[parent_task.slot_dir] = parent_summary
        row = next(
            (
                index
                for index, class_id in enumerate(parent_summary["class_ids"])
                if int(class_id) == parent_id
            ),
            None,
        )
        if row is None:
            continue
        inherited.append(
            {
                "class_ids": (child_id,),
                "class_names": (class_names[child_id],),
                "channel": parent_summary["channel"],
                "band": parent_summary["band"][row : row + 1],
                "counts": parent_summary["counts"][row : row + 1],
                "mean": parent_summary["mean"][row : row + 1],
                "mean_counts": parent_summary["mean_counts"][row : row + 1],
                "weight": int(parent_summary["counts"][row]),
                "layer": parent_summary["layer"],
                "log": (
                    f"{level_name} child={child_id}: inherited from "
                    f"{parent_level} parent={parent_id}"
                ),
            }
        )
    return inherited


def _save_task_umap(context, task, model, device, output_path: Path) -> None:
    """收集同一模型的 Train/Val 输入并生成联合 UMAP。"""
    train_inputs, train_labels = collect_task_inputs(context, task, "train", device)
    validation_inputs, validation_labels = collect_task_inputs(context, task, "val", device)
    config = context.config.analysis
    write_train_val_umap(
        model,
        train_inputs,
        validation_inputs,
        train_labels,
        validation_labels,
        task.class_names,
        output_path,
        config.umap_neighbors,
        config.umap_min_dist,
        context.config.execution.seed,
    )


def _resolve_channel_names(input_spec) -> tuple[str, ...]:
    """按模型输入通道顺序生成归因图的显示名称。"""
    names = [input_spec.norm_method]
    if input_spec.smooth_enable:
        names.append("smooth")
    if input_spec.d1_enable:
        names.append("d1")
    return tuple(names)


def _prepare_analysis_dirs(output_dir: Path) -> tuple[Path, Path]:
    """创建分析产物目录并返回图表与日志目录。"""
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, log_dir


def _resolve_device(device_value: str | None) -> torch.device:
    """解析分析使用的运行设备。"""
    return torch.device(device_value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _write_analysis_log(
    summaries,
    output_path: Path,
    tasks,
    mode: str,
    level_name: str,
    device: torch.device,
    execution_config,
    channel_names: tuple[str, ...],
    se_lines: list[str],
    figure_dir: Path,
    separate_class_plots_enable: bool,
) -> None:
    """按稳定段落写入通道、层、SE 与波段归因摘要日志。"""
    if mode == "level" and len(summaries) > 1:
        _write_aggregate_analysis_log(
            summaries,
            output_path,
            tasks,
            level_name,
            device,
            execution_config,
            figure_dir,
        )
        return
    channel = _aggregate_summary_values(summaries, "channel")
    layer_scores = _aggregate_layer_scores(summaries)
    band_count = len({name for summary in summaries for name in summary["class_names"]})
    lines = [
        f"Analysis target: {level_name}",
        "Using device: "
        f"{device.type} (config.use_gpu={execution_config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})",
        "",
        "=== Computing input channel importance and band importance ===",
        f"Using channel names: {list(channel_names[: len(channel)])}",
        f"Input channel importance: {np.array2string(channel)}",
        "",
        "=== Running Multi-layer Grad-CAM Analysis ===",
        "",
        "=== Layer Importance (merged by stage) ===",
    ]
    lines.extend(f"{name:<30}: {value:.4f}" for name, value in layer_scores.items())
    if se_lines:
        lines.extend(["", "===== SE Module Summary (Compact) =====", *se_lines])
    lines.extend(
        [
            "",
            "=== Computing band importance heatmap ===",
            "Saved band importance heatmap figures: "
            f"{band_count if separate_class_plots_enable else 1}",
            "Saved full-spectrum band importance CSV: "
            f"{figure_dir / 'band_importance_per_class.csv'}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aggregate_analysis_log(
    summaries,
    output_path: Path,
    tasks,
    level_name: str,
    device: torch.device,
    execution_config,
    figure_dir: Path,
) -> None:
    """写入层级级多模型归因汇总日志。"""
    lines = [
        f"Aggregate analysis for {level_name} over {len(tasks)} models.",
        "Using device: "
        f"{device.type} (config.use_gpu={execution_config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})",
    ]
    lines.extend(f"--- Model {task.parent_id} ---" for task in tasks)
    inherited_lines = [
        summary["log"]
        for summary in summaries
        if "inherited from" in summary.get("log", "")
    ]
    if inherited_lines:
        lines.extend(["", "Inherited single-child attribution:", *inherited_lines])
    lines.extend(
        [
            "",
            "Saved aggregate channel importance: "
            f"{figure_dir / 'channel_importance_IG_aggregate.png'}",
            "Saved aggregate layer importance: "
            f"{figure_dir / 'layer_importance_aggregate.png'}",
            "Saved aggregate band importance heatmap: "
            f"{figure_dir / 'band_importance_heatmap_aggregate.png'}",
            "Saved aggregate band importance CSV: "
            f"{figure_dir / 'band_importance_per_class_aggregate.csv'}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_summary_values(summaries, field_name: str) -> np.ndarray:
    """按分析样本量加权聚合摘要数组，用于整体日志。"""
    weights = np.asarray([summary["weight"] for summary in summaries], dtype=float)
    values = sum(summary[field_name] * weight for summary, weight in zip(summaries, weights))
    return values / max(float(weights.sum()), 1.0)


def _aggregate_layer_scores(summaries) -> dict[str, float]:
    """按分析样本量加权聚合各任务的层归因分数。"""
    weights = np.asarray([summary["weight"] for summary in summaries], dtype=float)
    scores: dict[str, float] = {}
    for summary, weight in zip(summaries, weights):
        for name, value in summary["layer"].items():
            scores[name] = scores.get(name, 0.0) + value * weight
    total = max(float(weights.sum()), 1.0)
    normalized = {name: value / total for name, value in scores.items()}
    return merge_layer_attribution_scores(normalized)


def write_se_summary(tasks, output_path: Path) -> list[str]:
    """汇总各分析模型槽位的 SE sidecar，并写入独立摘要文件。"""
    lines = build_se_summary_lines(tasks)
    if lines:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def build_se_summary_lines(tasks) -> list[str]:
    """读取 SE 通道缩放统计并整理为分析日志行。"""
    lines: list[str] = []
    for task in tasks:
        stats_path = Path(task.slot_dir) / "se_stats.pt"
        if not stats_path.is_file():
            continue
        stats = torch.load(stats_path, map_location="cpu")
        for name, values in stats.items():
            lines.append(
                f"{name}: mean={float(values['channel_mean'].mean()):.4f}, "
                f"std={float(values['channel_std'].mean()):.4f}, "
                f"min={float(values['channel_min'].min()):.4f}, "
                f"max={float(values['channel_max'].max()):.4f}, "
                f"samples={int(values['sample_count'])}"
            )
    return lines


__all__ = ["interpret_level", "interpret_slot", "umap_slot"]
