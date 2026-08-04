"""模型归因与 embedding 分析的任务编排。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ramanv2.analysis.context import load_analysis_context, resolve_analysis_level
from ramanv2.analysis.embedding import save_train_val_umap
from ramanv2.analysis.integrated_gradients import collect_task_inputs, compute_integrated_gradients
from ramanv2.analysis.layer_attribution import (
    compute_layer_attribution,
    merge_layer_attribution_scores,
)
from ramanv2.analysis.report import save_task_reports, write_aggregate_reports
from ramanv2.analysis.se_summary import write_se_summary
from ramanv2.analysis.task import build_parent_tasks, build_run_task
from ramanv2.inference.predictor import load_predictor
from ramanv2.spectra.axis import expected_wavenumbers


def run_interpret_run(source_dir: str, level_name: str, device: str | None = None) -> Path:
    """分析一个明确 global 或 parent 模型 run。"""
    context = load_analysis_context(source_dir)
    level = resolve_analysis_level(context, level_name)
    return _run_tasks(context, [build_run_task(context, level)], "run", device)


def run_interpret_parent_routed(
    source_dir: str,
    level_name: str,
    parent: str | None = None,
    device: str | None = None,
) -> Path:
    """分析目标层全部或指定 parent 子模型并聚合归因。"""
    context = load_analysis_context(source_dir)
    level = resolve_analysis_level(context, level_name)
    return _run_tasks(context, build_parent_tasks(context, level, parent), "parent-routed", device)


def _run_tasks(context, tasks, mode: str, device_value: str | None) -> Path:
    """执行单模型或多 parent 模型分析，并写入既定产物目录。"""
    device = torch.device(device_value or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = _resolve_output_dir(context, tasks[0], mode)
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    task_report_enable = mode == "run" or len(tasks) == 1
    summaries = [
        _analyze_task(
            context,
            task,
            device,
            figure_dir,
            collect_embedding=len(tasks) == 1,
            write_reports_enable=task_report_enable,
        )
        for task in tasks
    ]
    if mode == "parent-routed" and context.config.analysis.inherit_missing_levels_use:
        summaries.extend(_inherit_single_child_summaries(context, tasks[0].level_name, device, figure_dir))
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
        log_dir / "analysis_log.txt",
        tasks,
        mode,
        tasks[0].level_name,
        device,
        context.config.execution,
        _resolve_channel_names(context.input_spec),
        se_lines,
        figure_dir,
        context.config.analysis.separate_class_plots_use,
    )
    if mode == "parent-routed":
        _write_used_runs(tasks, output_dir.parent / "used_runs.json", tasks[0].level_name)
    return output_dir


def _resolve_output_dir(context, task, mode: str) -> Path:
    """保持单 run 与 parent-routed 的既有分析结果目录规则。"""
    if mode == "run":
        return Path(task.run_dir) / "analysis_result"
    return context.experiment_dir / task.level_name / "level_only_result" / "analysis_result"


def _analyze_task(
    context,
    task,
    device,
    figure_dir: Path,
    collect_embedding: bool,
    write_reports_enable: bool,
) -> dict:
    """执行一个模型任务的 IG、层归因、图表和可选 UMAP。"""
    predictor = load_predictor(task.run_dir, device, task.level_name)
    model = predictor.load_model(task.level_name, task.entry, task.parent_id)
    config = context.config.analysis
    inputs, labels = collect_task_inputs(context, task, config.attribution_split, device)
    batch_size = context.config.training.batch_size
    limit = min(len(inputs), max(1, config.attribution_batch_count) * batch_size)
    inputs, labels = inputs[:limit], labels[:limit]
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
    tag = task.level_name if task.parent_id is None else f"{task.level_name}_parent_{task.parent_id}"
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
        _save_task_umap(context, task, model, device, figure_dir / "umap_hier_train_val.png")
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


def _save_task_umap(context, task, model, device, output_path: Path) -> None:
    """收集同一模型的 Train/Val 输入并生成联合 UMAP。"""
    train_inputs, train_labels = collect_task_inputs(context, task, "train", device)
    validation_inputs, validation_labels = collect_task_inputs(context, task, "val", device)
    config = context.config.analysis
    save_train_val_umap(
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


def _inherit_single_child_summaries(context, level_name: str, device, figure_dir: Path) -> list[dict]:
    """将单子类父分支的上级模型归因映射到唯一子类。"""
    parent_level = context.dataset_index.get_parent_level(level_name)
    mapping = (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    single_children = {
        int(parent_id): int(children[0])
        for parent_id, children in mapping.items()
        if len(children) == 1
    }
    if parent_level is None or not single_children:
        return []
    try:
        parent_task = build_parent_tasks(context, parent_level, None)[0]
    except (FileNotFoundError, ValueError):
        return []
    parent_summary = _analyze_task(
        context,
        parent_task,
        device,
        figure_dir,
        collect_embedding=False,
        write_reports_enable=False,
    )
    inherited = []
    class_names = context.dataset_index.get_class_names(level_name)
    for local_row, parent_id in enumerate(parent_summary["class_ids"]):
        child_id = single_children.get(parent_id)
        if child_id is None:
            continue
        inherited.append(
            {
                "class_ids": (child_id,),
                "class_names": (class_names[child_id],),
                "channel": parent_summary["channel"],
                "band": parent_summary["band"][local_row : local_row + 1],
                "counts": parent_summary["counts"][local_row : local_row + 1],
                "mean": parent_summary["mean"][local_row : local_row + 1],
                "mean_counts": parent_summary["mean_counts"][local_row : local_row + 1],
                "weight": int(parent_summary["counts"][local_row]),
                "layer": parent_summary["layer"],
                "log": f"{level_name} child={child_id}: inherited from {parent_level} parent={parent_id}",
            }
        )
    return inherited


def _write_used_runs(tasks, output_path: Path, level_name: str) -> None:
    """记录 parent-routed 分析实际使用的模型 run。"""
    runs = {str(task.parent_id): task.entry.get("run_dir") for task in tasks}
    output_path.write_text(
        json.dumps({"mode": "parent-routed", "target_level": level_name, "runs": runs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    if mode == "parent-routed" and len(tasks) > 1:
        _write_aggregate_analysis_log(
            summaries,
            output_path,
            tasks,
            level_name,
            device,
            execution_config,
            figure_dir,
            separate_class_plots_enable,
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
    separate_class_plots_enable: bool,
) -> None:
    """写入多父类模型的聚合分析日志。"""
    band_count = len({name for summary in summaries for name in summary["class_names"]})
    lines = [
        f"Aggregate analysis for {level_name} over {len(tasks)} parents.",
        "Using device: "
        f"{device.type} (config.use_gpu={execution_config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})",
    ]
    lines.extend(f"--- Parent {task.parent_id} ---" for task in tasks)
    lines.extend(
        [
            "Saved aggregate channel importance: "
            f"{figure_dir / 'channel_importance_IG_aggregate.png'}",
            "Saved aggregate layer importance: "
            f"{figure_dir / 'layer_importance_aggregate.png'}",
            "Saved aggregate band importance heatmap figures: "
            f"{band_count if separate_class_plots_enable else 1}",
            "Saved aggregate full-spectrum band importance CSV: "
            f"{figure_dir / 'band_importance_per_class_aggregate.csv'}",
            "Note: Embedding plots are skipped in aggregate mode (different parent models).",
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
