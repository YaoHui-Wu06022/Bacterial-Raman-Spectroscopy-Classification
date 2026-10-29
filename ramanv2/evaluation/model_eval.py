"""深度模型验证集评估。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ramanv2.common.run_manifest import write_run_manifest
from ramanv2.core.runtime import resolve_device
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.evaluation.context import (
    EvaluationContext,
    RunEntry,
    build_used_runs,
    load_evaluation_context,
    resolve_entry_path,
    resolve_level_name,
    resolve_result_dir,
    resolve_run_entry,
)
from ramanv2.evaluation.report import write_model_report
from ramanv2.evaluation.scope import resolve_run_class_ids
from ramanv2.inference.predictor import Predictor, load_predictor
from ramanv2.inference.routing import predict_routed_labels


def evaluate_model_run(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
) -> Path:
    """评估一个明确全局模型或父类子模型 run。"""
    context = load_evaluation_context(source_dir)
    run_entry = resolve_run_entry(context)
    level_name = resolve_level_name(context, level_name)
    if run_entry.level_name != level_name:
        raise ValueError(
            f"run 属于 {run_entry.level_name}，不能按 {level_name} 评估"
        )
    target_device = resolve_device(device)
    predictor = load_predictor(run_entry.run_dir, target_device, level_name)
    result_dir = resolve_result_dir(context, level_name, "model", "run", run_entry)
    class_names, paths, labels, predictions = _predict_run(context, predictor, run_entry)
    return write_model_report(result_dir, class_names, paths, labels, predictions)


def evaluate_model_parent_routed(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
) -> Path:
    """使用真实父类标签选择目标层模型，评估该层分类能力。"""
    context = load_evaluation_context(source_dir)
    level_name = resolve_level_name(context, level_name)
    target_device = resolve_device(device)
    result_dir = resolve_result_dir(context, level_name, "model", "parent-routed")
    global_entry = (context.meta.get("level_models") or {}).get(level_name)
    if isinstance(global_entry, dict) and global_entry.get("model_path"):
        run_dir = resolve_entry_path(context.experiment_dir, global_entry, "run_dir")
        predictor = load_predictor(run_dir, target_device, level_name)
        run_entry = RunEntry(level_name, None, global_entry, run_dir)
        class_names, paths, labels, predictions = _predict_run(context, predictor, run_entry)
        used_runs = {level_name: build_used_runs(global_entry)}
    else:
        class_names, paths, labels, predictions, used_runs = _predict_parent_routed(
            context,
            level_name,
            target_device,
        )
    output_dir = write_model_report(result_dir, class_names, paths, labels, predictions)
    write_run_manifest(output_dir.parent / "used_runs.json", "parent-routed", level_name, used_runs)
    return output_dir


def evaluate_model_cascade(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
) -> Path:
    """从顶层到目标层执行端到端级联验证集评估。"""
    context = load_evaluation_context(source_dir)
    level_name = resolve_level_name(context, level_name)
    target_device = resolve_device(device)
    predictor = load_predictor(context.experiment_dir, target_device, level_name)
    preprocessor = InputPreprocessor(context.input_spec, target_device)
    level_index = context.dataset_index.head_name_to_idx[level_name]
    class_names = context.dataset_index.get_class_names(level_name)
    paths: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    for sample_index in context.validation_indices:
        true_label = int(context.dataset_index.level_labels[sample_index, level_index])
        if true_label < 0:
            continue
        inputs = preprocessor.preprocess_intensity(
            context.dataset_index.get_raw_intensity(int(sample_index))
        )
        prediction = predictor.predict_tensor(inputs, top_k=1)[0]
        paths.append(str(context.dataset_index.samples[sample_index]))
        labels.append(true_label)
        predictions.append(prediction.class_id)
    result_dir = resolve_result_dir(context, level_name, "model", "cascade")
    output_dir = write_model_report(result_dir, class_names, paths, labels, predictions)
    write_run_manifest(
        output_dir.parent / "used_runs.json",
        "cascade",
        level_name,
        predictor.build_used_runs(),
    )
    return output_dir


def _predict_run(
    context: EvaluationContext,
    predictor: Predictor,
    run_entry: RunEntry,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """在一个 run 的有效验证样本上收集局部类别空间预测。"""
    dataset_index = context.dataset_index
    level_index = dataset_index.head_name_to_idx[run_entry.level_name]
    class_names = predictor.resolve_target_class_names()
    class_ids = resolve_run_class_ids(dataset_index, run_entry)
    local_by_global = {class_id: local_id for local_id, class_id in enumerate(class_ids)}
    preprocessor = InputPreprocessor(context.input_spec, predictor.device)
    paths: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    for sample_index in context.validation_indices:
        global_label = int(dataset_index.level_labels[sample_index, level_index])
        if global_label not in local_by_global:
            continue
        inputs = preprocessor.preprocess_intensity(
            dataset_index.get_raw_intensity(int(sample_index))
        )
        prediction = predictor.predict_tensor(inputs, top_k=1)[0]
        paths.append(str(dataset_index.samples[sample_index]))
        labels.append(local_by_global[global_label])
        predictions.append(local_by_global[prediction.class_id])
    return class_names, paths, labels, predictions


def _predict_parent_routed(
    context: EvaluationContext,
    level_name: str,
    device: torch.device,
) -> tuple[list[str], list[str], list[int], list[int], dict[str, Any]]:
    """按真实父类将验证样本路由至对应子模型并还原全局类别标识。"""
    dataset_index = context.dataset_index
    parent_level = dataset_index.get_parent_level(level_name)
    if parent_level is None:
        raise ValueError(f"{level_name} 没有父层且缺少全局模型")
    level_index = dataset_index.head_name_to_idx[level_name]
    parent_index = dataset_index.head_name_to_idx[parent_level]
    entries = (context.meta.get("parent_models") or {}).get(level_name) or {}
    mapping = (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    predictors: dict[int, Predictor] = {}
    used_runs: dict[str, Any] = {}
    preprocessor = InputPreprocessor(context.input_spec, device)

    def load_parent_predictor(parent_id: int) -> Predictor:
        entry = entries.get(str(parent_id))
        if not isinstance(entry, dict) or not entry.get("model_path"):
            raise FileNotFoundError(f"{level_name} 缺少 parent={parent_id} 的子模型")
        predictor = load_predictor(
            resolve_entry_path(context.experiment_dir, entry, "run_dir"),
            device,
            level_name,
        )
        used_runs[str(parent_id)] = build_used_runs(entry)
        return predictor

    def predict_for_parent(predictor: Predictor, sample_index: int) -> int:
        inputs = preprocessor.preprocess_intensity(
            dataset_index.get_raw_intensity(int(sample_index))
        )
        return predictor.predict_tensor(inputs, top_k=1)[0].class_id

    validation_indices = context.validation_indices
    results = predict_routed_labels(
        validation_indices,
        dataset_index.level_labels[validation_indices, level_index],
        dataset_index.level_labels[validation_indices, parent_index],
        mapping,
        predictors,
        load_parent_predictor,
        predict_for_parent,
    )
    paths = [str(dataset_index.samples[index]) for index, _, _ in results]
    labels = [label for _, label, _ in results]
    predictions = [prediction for _, _, prediction in results]
    return dataset_index.get_class_names(level_name), paths, labels, predictions, {level_name: used_runs}


