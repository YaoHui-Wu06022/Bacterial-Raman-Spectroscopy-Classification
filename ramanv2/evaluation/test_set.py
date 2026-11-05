"""独立测试集评估与汇总。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ramanv2.common.device import resolve_device
from ramanv2.common.file_ops import read_json, write_json
from ramanv2.common.naming import build_natural_key
from ramanv2.data.dataset_io import load_arc_intensity
from ramanv2.data.profiles import resolve_dataset_dir, resolve_profile
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.evaluation.context import (
    EvaluationContext,
    load_evaluation_context,
    resolve_entry_path,
)
from ramanv2.evaluation.metrics_report import write_model_report
from ramanv2.inference.spectrum_input import sample_folder_spectra
from ramanv2.modeling.predictor import Predictor, load_predictor
from ramanv2.modeling.routing import predict_routed_labels
from ramanv2.inference.spectrum_input import build_inference_preprocessor, list_spectrum_paths
from ramanv2.inference.test_labels import build_folder_summary, build_test_folder_labels


TEST_RESULT_DIR_NAME = "test_result"
TEST_SAMPLE_CSV_NAME = "test_eval_results.csv"
SUMMARY_JSON_NAME = "test_summary.json"
SUMMARY_TEXT_NAME = "test_summary.txt"
WEAK_CLASS_COUNT = 5
DEFAULT_TEST_LEVEL = "level_1"


def resolve_preprocessed_test_dir(
    profile_id: str,
    project_root: Path | str | None = None,
) -> Path:
    """返回 profile 预处理后的独立测试集目录 ``dataset/<profile>/test``。"""
    profile = resolve_profile(profile_id)
    return resolve_dataset_dir(profile, project_root) / profile.root_test


def evaluate_test_set(
    experiment_dir: Path | str,
    device: str | None = None,
    test_dir: Path | str | None = None,
    *,
    level_name: str = DEFAULT_TEST_LEVEL,
) -> dict[str, Any]:
    """在独立测试集上评估实验内指定层级的模型，并写出汇总。

    缺省只评估 ``level_1``；需要检查其他层级时显式传 ``level_name``。
    测试目录应是 ``build_test`` 预处理后的 ``dataset/<profile>/test``；缺省
    时按实验配置的 profile 解析。只评估 ``level_name`` 指定的层级：该层有
    全局模型时用 ``model``，只有父类子模型时用 ``parent-routed``，若该层是
    最深层再追加端到端 ``cascade``。每个测试文件夹按 cell 分组随机抽取最多
    ``INFERENCE_SAMPLES_PER_FOLDER`` 条光谱（同一 cell 至多一条），与样本推理
    口径一致。测试样本按文件夹前缀对齐实验类别空间，对不上的整组计入忽略
    清单，不参与指标计算。
    """
    context = load_evaluation_context(experiment_dir)
    target_device = resolve_device(device)
    target_level = context.dataset_index.resolve_level_name(level_name)
    test_path = (
        resolve_preprocessed_test_dir(context.config.dataset.profile_id)
        if test_dir is None
        else Path(test_dir).resolve()
    )
    if not test_path.is_dir():
        raise FileNotFoundError(f"缺少独立测试集目录：{test_path}")
    folder_names, sample_paths, sample_folders = _scan_test_set(test_path)

    level_labels: dict[str, np.ndarray] = {}
    level_ignored_folders: dict[str, list[str]] = {}
    required_levels = [target_level]
    parent_level = context.dataset_index.get_parent_level(target_level)
    if parent_level is not None:
        required_levels.append(parent_level)
    for level in required_levels:
        labels, ignored_folders = _map_test_level_labels(
            context, folder_names, sample_folders, level
        )
        level_labels[level] = labels
        level_ignored_folders[level] = ignored_folders

    records = _evaluate_test_level(
        context,
        sample_paths,
        sample_folders,
        level_labels,
        target_level,
        target_device,
    )
    # 端到端级联只在所选层级即最深层且使用父类子模型时才有独立意义。
    deepest_level = context.dataset_index.level_names[-1]
    if (
        target_level == deepest_level
        and _has_parent_models(context, target_level)
        and not _has_global_model(context, target_level)
    ):
        records.append(
            _evaluate_test_cascade(
                context,
                sample_paths,
                sample_folders,
                level_labels[target_level],
                target_level,
                target_device,
            )
        )
    if not records:
        raise RuntimeError(f"{target_level} 没有可用于独立测试的模型")

    summary_path = _write_test_summary(
        context,
        test_path,
        len(sample_paths),
        len(folder_names),
        {target_level: int((level_labels[target_level] < 0).sum())},
        {target_level: level_ignored_folders[target_level]},
        records,
        target_level,
    )
    return {
        "summary_path": str(summary_path),
        "test_dir": str(test_path),
        "sample_count": len(sample_paths),
        "folder_count": len(folder_names),
        "level_name": target_level,
        "overall": select_deployment_record(records),
        "records": records,
    }


def select_deployment_record(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """选择代表实际部署效果的一条记录：级联优先，否则取最深层记录。"""
    for record in records:
        if record["mode"] == "cascade":
            return record
    if not records:
        return None
    deepest_level = records[-1]["level_name"]
    for record in reversed(records):
        if record["level_name"] == deepest_level:
            return record
    return records[-1]


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


def _has_global_model(context: EvaluationContext, level_name: str) -> bool:
    """判断层级是否训练了覆盖全类别的全局模型。"""
    entry = (context.meta.get("level_models") or {}).get(level_name)
    return isinstance(entry, dict) and bool(entry.get("model_path"))


def _has_parent_models(context: EvaluationContext, level_name: str) -> bool:
    """判断层级是否训练了任一父类子模型。"""
    entries = (context.meta.get("parent_models") or {}).get(level_name) or {}
    return any(
        isinstance(entry, dict) and bool(entry.get("model_path"))
        for entry in entries.values()
    )


def _evaluate_test_level(
    context: EvaluationContext,
    sample_paths: list[Path],
    sample_folders: list[str],
    level_labels: dict[str, np.ndarray],
    level_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    """按层级元数据选择可用评估模式：全局模型或真实父类路由。"""
    if _has_global_model(context, level_name):
        return [
            _evaluate_test_global_model(
                context,
                sample_paths,
                sample_folders,
                level_labels[level_name],
                level_name,
                device,
            )
        ]
    if _has_parent_models(context, level_name):
        return [
            _evaluate_test_parent_routed(
                context,
                sample_paths,
                sample_folders,
                level_labels,
                level_name,
                device,
            )
        ]
    return []


def _evaluate_test_global_model(
    context: EvaluationContext,
    sample_paths: list[Path],
    sample_folders: list[str],
    labels: np.ndarray,
    level_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """用全局模型直接预测测试集。"""
    entry = (context.meta.get("level_models") or {}).get(level_name)
    run_dir = resolve_entry_path(context.experiment_dir, entry, "run_dir")
    return _evaluate_test_predictor(
        context,
        load_predictor(run_dir, device, level_name),
        sample_paths,
        sample_folders,
        labels,
        level_name,
        "model",
    )


def _evaluate_test_parent_routed(
    context: EvaluationContext,
    sample_paths: list[Path],
    sample_folders: list[str],
    level_labels: dict[str, np.ndarray],
    level_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """按真实父类把测试样本路由到对应子模型，评估子模型本身的能力。"""
    parent_level = context.dataset_index.get_parent_level(level_name)
    if parent_level is None:
        raise ValueError(f"{level_name} 没有父层，无法按真实父类路由")
    labels = level_labels[level_name]
    parent_labels = level_labels[parent_level]
    entries = (context.meta.get("parent_models") or {}).get(level_name) or {}
    mapping = (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    preprocessor = build_inference_preprocessor(context.input_spec, device)
    predictors: dict[int, Predictor] = {}

    def load_parent_predictor(parent_id: int) -> Predictor:
        entry = entries.get(str(parent_id))
        if not isinstance(entry, dict) or not entry.get("model_path"):
            raise FileNotFoundError(f"{level_name} 缺少 parent={parent_id} 的子模型")
        return load_predictor(
            resolve_entry_path(context.experiment_dir, entry, "run_dir"),
            device,
            level_name,
        )

    def predict_for_parent(predictor: Predictor, sample_index: int) -> int:
        inputs = preprocessor.preprocess_intensity(
            load_arc_intensity(sample_paths[sample_index])
        )
        return predictor.predict_tensor(inputs, top_k=1)[0].class_id

    results = predict_routed_labels(
        range(len(sample_paths)),
        labels,
        parent_labels,
        mapping,
        predictors,
        load_parent_predictor,
        predict_for_parent,
    )
    if not results:
        raise RuntimeError(f"测试集没有可评估的 {level_name} 父类路由样本")
    paths = [str(sample_paths[index]) for index, _, _ in results]
    folders = [sample_folders[index] for index, _, _ in results]
    truth = [label for _, label, _ in results]
    predictions = [prediction for _, _, prediction in results]
    class_names = context.dataset_index.get_class_names(level_name)
    folder_metrics = compute_folder_metrics(folders, truth, predictions, class_names)
    result_dir = _resolve_test_result_dir(context, level_name, "parent-routed")
    write_model_report(
        result_dir,
        class_names,
        paths,
        truth,
        predictions,
        sample_name=TEST_SAMPLE_CSV_NAME,
    )
    return _build_record(level_name, "parent-routed", result_dir, folder_metrics)


def _evaluate_test_cascade(
    context: EvaluationContext,
    sample_paths: list[Path],
    sample_folders: list[str],
    labels: np.ndarray,
    level_name: str,
    device: torch.device,
) -> dict[str, Any]:
    """从顶层模型开始端到端预测测试集，衡量实际部署效果。"""
    return _evaluate_test_predictor(
        context,
        load_predictor(context.experiment_dir, device, level_name),
        sample_paths,
        sample_folders,
        labels,
        level_name,
        "cascade",
    )


def _evaluate_test_predictor(
    context: EvaluationContext,
    predictor: Predictor,
    sample_paths: list[Path],
    sample_folders: list[str],
    labels: np.ndarray,
    level_name: str,
    mode: str,
) -> dict[str, Any]:
    """用给定预测器评估测试集，写出 model 或 cascade 结果。"""
    indices = np.flatnonzero(labels >= 0)
    if indices.size == 0:
        raise RuntimeError(f"测试集没有可评估的 {level_name} 样本")
    preprocessor = build_inference_preprocessor(context.input_spec, predictor.device)
    predictions = _predict_indices(predictor, preprocessor, sample_paths, indices)
    truth = labels[indices].tolist()
    folders = [sample_folders[int(index)] for index in indices]
    class_names = context.dataset_index.get_class_names(level_name)
    folder_metrics = compute_folder_metrics(folders, truth, predictions, class_names)
    result_dir = _resolve_test_result_dir(context, level_name, mode)
    write_model_report(
        result_dir,
        class_names,
        [str(sample_paths[index]) for index in indices],
        truth,
        predictions,
        sample_name=TEST_SAMPLE_CSV_NAME,
    )
    return _build_record(level_name, mode, result_dir, folder_metrics)


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


def _map_test_level_labels(
    context: EvaluationContext,
    folder_names: list[str],
    sample_folders: list[str],
    level_name: str,
) -> tuple[np.ndarray, list[str]]:
    """把测试文件夹前缀映射到实验的全局类别索引，未归属记为 -1。

    返回逐谱标签和整组被忽略的文件夹名。忽略是文件夹级的：一个来源组
    要么整体参与评估，要么整体不参与，避免同组样本跨标签空间撕裂。
    """
    class_names = context.dataset_index.get_class_names(level_name)
    label_by_folder, rows = build_test_folder_labels(
        folder_names, context.meta, level_name, class_names
    )
    labels = np.asarray(
        [label_by_folder.get(name, -1) for name in sample_folders], dtype=np.int64
    )
    ignored = sorted(
        (str(row["folder"]) for row in rows if not row["selected"]),
        key=build_natural_key,
    )
    return labels, ignored


def _resolve_test_result_dir(
    context: EvaluationContext,
    level_name: str,
    mode: str,
) -> Path:
    """返回测试集评估结果目录：<experiment>/<level>/test_result/<mode>。"""
    return context.experiment_dir / level_name / TEST_RESULT_DIR_NAME / mode


def _build_record(
    level_name: str,
    mode: str,
    result_dir: Path,
    folder_metrics: dict[str, Any],
) -> dict[str, Any]:
    """从结果目录的 metrics.json 提取一条汇总记录。"""
    metrics = read_json(result_dir / "metrics.json")
    summary = metrics["summary"]
    scored_classes = sorted(
        (item for item in metrics["classes"] if int(item["support"]) > 0),
        key=lambda item: float(item["f1_score"]),
    )
    covered_class_count = sum(
        1 for item in metrics["classes"] if int(item["support"]) > 0
    )
    return {
        "level_name": level_name,
        "mode": mode,
        "result_dir": str(result_dir),
        "spectrum_accuracy": float(summary["accuracy"]),
        "accuracy": float(summary["accuracy"]),
        "folder_accuracy": float(folder_metrics["folder_accuracy"]),
        "folder_correct": int(folder_metrics["folder_correct"]),
        "folder_count": int(folder_metrics["folder_count"]),
        "macro_f1": float(summary["macro_f1"]),
        "macro_recall": float(summary["macro_recall"]),
        "support": int(summary["support"]),
        "class_count": len(metrics["classes"]),
        "covered_class_count": covered_class_count,
        "weak_classes": [
            {
                "name": item["name"],
                "f1_score": float(item["f1_score"]),
                "recall": float(item["recall"]),
                "support": int(item["support"]),
            }
            for item in scored_classes[:WEAK_CLASS_COUNT]
        ],
    }


def _write_test_summary(
    context: EvaluationContext,
    test_path: Path,
    sample_count: int,
    folder_count: int,
    ignored_sample_counts: dict[str, int],
    ignored_folders: dict[str, list[str]],
    records: list[dict[str, Any]],
    target_level: str,
) -> Path:
    """写出独立测试集汇总的 JSON 与可读文本。"""
    payload = {
        "experiment_dir": str(context.experiment_dir),
        "test_dir": str(test_path),
        "target_level": target_level,
        "test_sample_count": int(sample_count),
        "test_folder_count": int(folder_count),
        "ignored_sample_counts": ignored_sample_counts,
        "ignored_folders": ignored_folders,
        "overall": select_deployment_record(records),
        "records": records,
    }
    json_path = context.experiment_dir / SUMMARY_JSON_NAME
    write_json(json_path, payload)
    text_path = context.experiment_dir / SUMMARY_TEXT_NAME
    text_path.write_text(_format_test_summary(payload), encoding="utf-8")
    return text_path


def _format_test_summary(payload: dict[str, Any]) -> str:
    """把汇总数据排版为固定宽度的可读表格。"""
    ignored_text = ", ".join(
        f"{level_name}={count}"
        for level_name, count in payload["ignored_sample_counts"].items()
    )
    ignored_folder_total = {
        level_name: len(names)
        for level_name, names in payload.get("ignored_folders", {}).items()
    }
    ignored_folder_text = ", ".join(
        f"{level_name}={count}" for level_name, count in ignored_folder_total.items()
    )
    lines = [
        "独立测试集评估汇总",
        f"experiment_dir = {payload['experiment_dir']}",
        f"target_level = {payload.get('target_level', '')}",
        f"test_dir = {payload['test_dir']}",
        f"test_sample_count = {payload['test_sample_count']}",
        f"test_folder_count = {payload.get('test_folder_count', 0)}",
        f"ignored_samples = {ignored_text or '0'}",
        f"ignored_folders = {ignored_folder_text or '0'}",
    ]
    overall = payload.get("overall") or select_deployment_record(payload["records"])
    if overall is not None:
        lines.extend([
            "",
            "整体结果（部署口径）",
            "=====================================",
            f"target_level = {overall['level_name']}  mode = {overall['mode']}",
            f"Spectrum Accuracy = {overall['spectrum_accuracy'] * 100:.4f}%  (support={overall['support']})",
            f"Folder Accuracy   = {overall['folder_accuracy'] * 100:.4f}%  "
            f"({overall['folder_correct']}/{overall['folder_count']})",
            f"Class Coverage    = {overall['covered_class_count']}/{overall['class_count']}",
            f"Macro F1          = {overall['macro_f1'] * 100:.4f}%",
            f"Macro Recall      = {overall['macro_recall'] * 100:.4f}%",
            "=====================================",
        ])
    lines.extend([
        "",
        f"{'target_level':<14}{'mode':<16}{'spectrum_acc':>13}{'folder_acc':>12}"
        f"{'support':>9}{'folders':>9}{'macro_f1':>10}{'macro_recall':>14}",
    ])
    for record in payload["records"]:
        lines.append(
            f"{record['level_name']:<14}{record['mode']:<16}"
            f"{record['spectrum_accuracy'] * 100:>12.4f}%"
            f"{record['folder_accuracy'] * 100:>11.4f}%"
            f"{record['support']:>9d}{record['folder_count']:>9d}"
            f"{record['macro_f1'] * 100:>9.4f}%"
            f"{record['macro_recall'] * 100:>13.4f}%"
        )
    for level_name, names in payload.get("ignored_folders", {}).items():
        if names:
            lines.extend(["", f"{level_name} 未匹配的测试文件夹：{', '.join(names)}"])
    weak_lines = [
        f"{record['level_name']}/{record['mode']}: "
        + "; ".join(
            f"{item['name']} f1={item['f1_score'] * 100:.2f}% "
            f"recall={item['recall'] * 100:.2f}% support={item['support']}"
            for item in record["weak_classes"]
        )
        for record in payload["records"]
        if record["weak_classes"]
    ]
    if weak_lines:
        lines.extend(["", f"最弱类别（按 F1 升序取前 {WEAK_CLASS_COUNT}）"])
        lines.extend(weak_lines)
    return "\n".join(lines) + "\n"


__all__ = [
    "compute_folder_metrics",
    "evaluate_test_set",
    "resolve_preprocessed_test_dir",
    "select_deployment_record",
]
