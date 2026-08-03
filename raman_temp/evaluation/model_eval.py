"""深度模型验证集评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from raman_temp.data.input import InputPreprocessor
from raman_temp.evaluation.context import (
    EvaluationContext,
    RunEntry,
    build_used_runs,
    load_evaluation_context,
    resolve_entry_path,
    resolve_level_name,
    resolve_result_dir,
    resolve_run_entry,
)
from raman_temp.evaluation.report import write_model_report
from raman_temp.inference.predictor import Predictor, load_predictor


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
    target_device = _resolve_device(device)
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
    target_device = _resolve_device(device)
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
    _write_used_runs(output_dir.parent, "parent-routed", level_name, used_runs)
    return output_dir


def evaluate_model_cascade(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
) -> Path:
    """从顶层到目标层执行端到端级联验证集评估。"""
    context = load_evaluation_context(source_dir)
    level_name = resolve_level_name(context, level_name)
    target_device = _resolve_device(device)
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
    _write_used_runs(output_dir.parent, "cascade", level_name, predictor.build_used_runs())
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
    class_ids = _resolve_run_class_ids(dataset_index, run_entry)
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
    paths: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    preprocessor = InputPreprocessor(context.input_spec, device)
    for sample_index in context.validation_indices:
        true_label = int(dataset_index.level_labels[sample_index, level_index])
        parent_id = int(dataset_index.level_labels[sample_index, parent_index])
        if true_label < 0 or parent_id < 0:
            continue
        child_ids = [int(item) for item in mapping.get(str(parent_id), [])]
        if true_label not in child_ids:
            continue
        entry = entries.get(str(parent_id))
        if len(child_ids) == 1:
            predicted_label = child_ids[0]
        else:
            if not isinstance(entry, dict) or not entry.get("model_path"):
                raise FileNotFoundError(
                    f"{level_name} 缺少 parent={parent_id} 的子模型"
                )
            predictor = predictors.get(parent_id)
            if predictor is None:
                run_dir = resolve_entry_path(context.experiment_dir, entry, "run_dir")
                predictor = load_predictor(run_dir, device, level_name)
                predictors[parent_id] = predictor
                used_runs[str(parent_id)] = build_used_runs(entry)
            inputs = preprocessor.preprocess_intensity(
                dataset_index.get_raw_intensity(int(sample_index))
            )
            predicted_label = predictor.predict_tensor(inputs, top_k=1)[0].class_id
        paths.append(str(dataset_index.samples[sample_index]))
        labels.append(true_label)
        predictions.append(predicted_label)
    return dataset_index.get_class_names(level_name), paths, labels, predictions, {level_name: used_runs}


def _resolve_run_class_ids(dataset_index, run_entry: RunEntry) -> list[int]:
    """返回单 run 输出位置对应的全局类别标识。"""
    if run_entry.parent_id is None:
        return list(range(dataset_index.num_classes_by_level[run_entry.level_name]))
    return [int(item) for item in run_entry.values.get("child_ids") or []]


def _resolve_device(value: str | None) -> torch.device:
    """按显式参数或 CUDA 可用性选择评估设备。"""
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_used_runs(
    result_root: Path,
    mode: str,
    level_name: str,
    used_runs: Mapping[str, Any],
) -> None:
    """写入实验级模型评估使用的模型 run 清单。"""
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "used_runs.json").write_text(
        json.dumps(
            {"mode": mode, "target_level": level_name, "runs": used_runs},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
