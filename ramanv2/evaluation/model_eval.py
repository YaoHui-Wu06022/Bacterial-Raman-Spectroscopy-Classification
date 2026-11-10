"""深度模型验证集评估。"""

from __future__ import annotations

from pathlib import Path

from ramanv2.common.device import resolve_device
from ramanv2.core.run_context import VAL_RESULT_DIR_NAME
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.evaluation.context import (
    EvaluationContext,
    RunEntry,
    load_evaluation_context,
    resolve_level_name,
    resolve_run_class_ids,
    resolve_run_entry,
)
from ramanv2.evaluation.metrics_report import write_model_report
from ramanv2.modeling.predictor import Predictor, load_predictor


def evaluate_model_run(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
) -> Path:
    """评估一个模型槽位在其类别范围内的验证集表现。"""
    context = load_evaluation_context(source_dir)
    run_entry = resolve_run_entry(context)
    level_name = resolve_level_name(context, level_name)
    if run_entry.level_name != level_name:
        raise ValueError(
            f"模型槽位属于 {run_entry.level_name}，不能按 {level_name} 评估"
        )
    target_device = resolve_device(device)
    predictor = load_predictor(run_entry.run_dir, target_device, level_name)
    result_dir = run_entry.run_dir / VAL_RESULT_DIR_NAME
    class_names, paths, labels, predictions = _predict_run(context, predictor, run_entry)
    return write_model_report(result_dir, class_names, paths, labels, predictions)


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


