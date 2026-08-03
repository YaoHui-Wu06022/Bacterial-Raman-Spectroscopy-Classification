"""模型归因与 embedding 分析的任务编排。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from raman_temp.analysis.context import load_analysis_context, resolve_analysis_level
from raman_temp.analysis.embedding import save_train_val_umap
from raman_temp.analysis.integrated_gradients import collect_task_inputs, compute_integrated_gradients
from raman_temp.analysis.layer_attribution import compute_layer_attribution
from raman_temp.analysis.report import save_task_reports, write_aggregate_reports
from raman_temp.analysis.se_summary import write_se_summary
from raman_temp.analysis.task import build_parent_tasks, build_run_task
from raman_temp.inference.predictor import load_predictor


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
    summaries = [
        _analyze_task(context, task, device, figure_dir, collect_embedding=len(tasks) == 1)
        for task in tasks
    ]
    if mode == "parent-routed" and context.config.analysis.inherit_missing_levels_use:
        summaries.extend(_inherit_single_child_summaries(context, tasks[0].level_name, device, figure_dir))
    write_aggregate_reports(summaries, figure_dir, context.config.analysis.row_norm)
    write_se_summary(tasks, log_dir / "se_summary.txt")
    _write_analysis_log(summaries, log_dir / "analysis_log.txt")
    if mode == "parent-routed":
        _write_used_runs(tasks, output_dir.parent / "used_runs.json", tasks[0].level_name)
    return output_dir


def _resolve_output_dir(context, task, mode: str) -> Path:
    """保持单 run 与 parent-routed 的既有分析结果目录规则。"""
    if mode == "run":
        return Path(task.run_dir) / "analysis_result"
    return context.experiment_dir / task.level_name / "level_only_result" / "analysis_result"


def _analyze_task(context, task, device, figure_dir: Path, collect_embedding: bool) -> dict:
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
    save_task_reports(
        figure_dir,
        tag,
        task.class_names,
        result.channel_importance,
        result.band_importance,
        result.mean_spectra,
        layer_scores,
        config.row_norm,
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
    train_inputs, _train_labels = collect_task_inputs(context, task, "train", device)
    validation_inputs, _validation_labels = collect_task_inputs(context, task, "val", device)
    config = context.config.analysis
    save_train_val_umap(
        model,
        train_inputs,
        validation_inputs,
        output_path,
        config.umap_neighbors,
        config.umap_min_dist,
        context.config.execution.seed,
    )


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
    parent_summary = _analyze_task(context, parent_task, device, figure_dir, False)
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


def _write_analysis_log(summaries, output_path: Path) -> None:
    """写入每个分析或继承任务的样本与类别摘要。"""
    output_path.write_text(
        "\n".join(summary["log"] for summary in summaries) + "\n",
        encoding="utf-8",
    )
