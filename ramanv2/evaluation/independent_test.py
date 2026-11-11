"""独立测试集评估：按模型槽位的类别范围筛选并写入槽位结果。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from ramanv2.common.device import resolve_device
from ramanv2.common.file_ops import read_json
from ramanv2.common.naming import build_natural_key
from ramanv2.core.run_context import METRICS_FILE_NAME, TEST_RESULT_DIR_NAME
from ramanv2.data.dataset_io import load_arc_intensity
from ramanv2.data.profiles import resolve_selection
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.evaluation.context import (
    load_evaluation_context,
    resolve_run_class_ids,
    resolve_run_entry,
)
from ramanv2.evaluation.metrics_report import write_model_report
from ramanv2.evaluation.summary_text import build_result_summary
from ramanv2.inference.spectrum_input import (
    build_inference_preprocessor,
    list_spectrum_paths,
    sample_folder_spectra,
)
from ramanv2.inference.test_labels import build_folder_summary, build_test_folder_labels
from ramanv2.modeling.predictor import Predictor, load_predictor


TEST_SAMPLE_CSV_NAME = "test_eval_results.csv"


def resolve_preprocessed_test_dir(
    profile_id: str,
    project_root: Path | str | None = None,
    dataset_id: str | None = None,
) -> Path:
    """返回 profile 预处理后的独立测试集目录 ``dataset/<profile>/test``。"""
    selection = resolve_selection(dataset_id, profile_id, project_root)
    return selection.dataset_dir / selection.profile.root_test


def evaluate_test_set(
    source_dir: Path | str,
    device: str | None = None,
    test_dir: Path | str | None = None,
) -> dict[str, Any]:
    """评估一个模型槽位在其类别范围内的独立测试表现。"""
    context = load_evaluation_context(source_dir)
    run_entry = resolve_run_entry(context)
    level_name = run_entry.level_name
    target_device = resolve_device(device)
    random.seed(context.config.execution.seed)
    test_path = (
        resolve_preprocessed_test_dir(
            context.config.dataset.profile_id,
            dataset_id=context.config.dataset.dataset_id,
        )
        if test_dir is None
        else Path(test_dir).resolve()
    )
    if not test_path.is_dir():
        raise FileNotFoundError(f"缺少独立测试集目录：{test_path}")

    folder_names, sample_paths, sample_folders = _scan_test_set(test_path)
    level_class_names = context.dataset_index.get_class_names(level_name)
    label_by_folder, _rows = build_test_folder_labels(
        folder_names,
        context.meta,
        level_name,
        level_class_names,
    )
    labels = np.asarray(
        [label_by_folder.get(name, -1) for name in sample_folders],
        dtype=np.int64,
    )
    class_ids = resolve_run_class_ids(context.dataset_index, run_entry)
    local_by_global = {
        int(class_id): local_id for local_id, class_id in enumerate(class_ids)
    }
    indices = np.flatnonzero(np.isin(labels, list(local_by_global)))
    if indices.size == 0:
        raise RuntimeError(f"测试集没有属于 {run_entry.run_dir} 类别范围的样本")

    predictor = load_predictor(run_entry.run_dir, target_device, level_name)
    preprocessor = build_inference_preprocessor(context.input_spec, predictor.device)
    predictions = _predict_indices(predictor, preprocessor, sample_paths, indices)
    truth = [local_by_global[int(labels[int(index)])] for index in indices]
    predicted = [local_by_global[int(item)] for item in predictions]
    class_names = predictor.resolve_target_class_names()
    folders = [sample_folders[int(index)] for index in indices]
    folder_metrics = compute_folder_metrics(folders, truth, predicted, class_names)

    result_dir = run_entry.run_dir / TEST_RESULT_DIR_NAME
    write_model_report(
        result_dir,
        class_names,
        [str(sample_paths[int(index)]) for index in indices],
        truth,
        predicted,
        sample_name=TEST_SAMPLE_CSV_NAME,
        extra_summary={"folder": folder_metrics},
        metric_class_ids=sorted(set(truth)),
    )
    metrics = read_json(result_dir / METRICS_FILE_NAME)
    summary = metrics["summary"]
    _write_summary_text(result_dir, level_name, metrics)
    _maybe_generate_level_inference(context, level_name, test_path, target_device)
    return {
        "run_dir": str(run_entry.run_dir),
        "level_name": level_name,
        "parent_id": run_entry.parent_id,
        "test_dir": str(test_path),
        "result_dir": str(result_dir),
        "sample_count": int(indices.size),
        "folder_count": int(folder_metrics["folder_count"]),
        "folder_correct": int(folder_metrics["folder_correct"]),
        "folder_accuracy": float(folder_metrics["folder_accuracy"]),
        "spectrum_accuracy": float(summary["accuracy"]),
        "macro_f1": float(summary["macro_f1"]),
        "macro_recall": float(summary["macro_recall"]),
        "support": int(summary["support"]),
        "class_count": len(class_names),
        "covered_class_count": sum(
            1 for item in metrics["classes"] if int(item["support"]) > 0
        ),
    }


def _write_summary_text(
    result_dir: Path,
    level_name: str,
    metrics: dict[str, Any],
) -> None:
    """按项目统一格式写入独立测试摘要。"""
    (Path(result_dir) / "summary.txt").write_text(
        build_result_summary(metrics, level_name, "model"),
        encoding="utf-8",
    )


def compute_folder_metrics(
    folders: list[str],
    truth: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """按文件夹多数预测计算文件夹级准确率，复用独立推理的文件夹汇总口径。"""
    predictions_by_folder: dict[str, list[dict[str, str]]] = {}
    truth_by_folder: dict[str, int] = {}
    for folder, expected, predicted in zip(folders, truth, predictions):
        predictions_by_folder.setdefault(folder, []).append(
            {"top1_label": class_names[int(predicted)]}
        )
        truth_by_folder.setdefault(folder, int(expected))
    correct = 0
    for folder, folder_predictions in predictions_by_folder.items():
        row = build_folder_summary(
            folder,
            class_names[truth_by_folder[folder]],
            class_names,
            folder_predictions,
        )
        correct += int(bool(row["folder_correct"]))
    total = len(predictions_by_folder)
    return {
        "folder_count": total,
        "folder_correct": correct,
        "folder_accuracy": correct / total if total else 0.0,
    }


def _maybe_generate_level_inference(
    context: Any,
    level_name: str,
    test_path: Path,
    device: Any,
) -> None:
    """test_result 生成后补齐已完备层级的级联推理；未补齐时由 helper 跳过。"""
    from ramanv2.inference.folder_inference import run_experiment_inference

    random.seed(context.config.execution.seed)
    levels = tuple((context.meta or {}).get("head_names") or (level_name,))
    try:
        results = run_experiment_inference(
            context.experiment_dir,
            test_path,
            levels,
            device=device,
            seed=context.config.execution.seed,
        )
    except FileNotFoundError as error:
        print(f"[Inference] 跳过级联推理：{error}")
        return
    for name, result_dir in results.items():
        print(f"[Inference] {name} -> {result_dir}")


def _scan_test_set(test_path: Path) -> tuple[list[str], list[Path], list[str]]:
    """按自然序扫描扁平测试集，返回文件夹名、抽样光谱路径与逐谱所属文件夹。

    每个文件夹按 cell 分组随机抽取最多 ``INFERENCE_SAMPLES_PER_FOLDER`` 条，
    与样本推理保持一致的独立测试口径。
    """
    folder_names: list[str] = []
    sample_paths: list[Path] = []
    sample_folders: list[str] = []
    for folder_dir in sorted(
        (path for path in test_path.iterdir() if path.is_dir()),
        key=lambda path: build_natural_key(path.name),
    ):
        paths = list_spectrum_paths(folder_dir)
        if not paths:
            continue
        sampled = sample_folder_spectra([(path.name, path) for path in paths])
        folder_names.append(folder_dir.name)
        sample_paths.extend(path for _, path in sampled)
        sample_folders.extend([folder_dir.name] * len(sampled))
    if not sample_paths:
        raise RuntimeError(f"测试目录中没有 .arc_data 光谱：{test_path}")
    return folder_names, sample_paths, sample_folders


def _predict_indices(
    predictor: Predictor,
    preprocessor: InputPreprocessor,
    sample_paths: list[Path],
    indices: np.ndarray,
) -> list[int]:
    """逐条预测指定样本，返回全局类别索引。

    测试集已由 ``build_test`` 预处理到统一网格，这里直接沿用训练侧的
    ``InputPreprocessor``，不再重复基线、裁剪或插值。
    """
    predictions: list[int] = []
    for sample_index in indices:
        inputs = preprocessor.preprocess_intensity(
            load_arc_intensity(sample_paths[int(sample_index)])
        )
        predictions.append(predictor.predict_tensor(inputs, top_k=1)[0].class_id)
    return predictions


__all__ = [
    "TEST_RESULT_DIR_NAME",
    "TEST_SAMPLE_CSV_NAME",
    "compute_folder_metrics",
    "evaluate_test_set",
    "resolve_preprocessed_test_dir",
]
