"""独立测试文件夹推理的完整编排入口。"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from ramanv2.core.paths import DATASET_ROOT
from ramanv2.core.runtime import resolve_device
from ramanv2.common.naming import build_natural_key, parse_folder_prefix
from ramanv2.data.io import load_arc_intensity
from ramanv2.data.profiles import (
    get_dataset_dir,
    get_profile,
    resolve_test_root,
    resolve_training_dir,
)
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.evaluation.context import (
    EvaluationContext,
    load_evaluation_context,
    resolve_entry_path,
)
from ramanv2.evaluation.report import write_model_report
from ramanv2.inference.directory import list_spectrum_paths, resolve_input_dirs
from ramanv2.inference.cache import CachedFolder, load_test_cache
from ramanv2.inference.labels import (
    build_expected_label_lookup,
    build_folder_summary,
    build_test_folder_labels,
    build_test_input_selection,
)
from ramanv2.inference.predictor import Predictor, load_predictor
from ramanv2.inference.routing import predict_routed_labels
from ramanv2.common.run_manifest import write_run_manifest
from ramanv2.inference.report import (
    plot_folder_spectra,
    write_file_report,
    write_summary_json,
    write_summary_report,
)
from ramanv2.inference.spectra import (
    build_inference_preprocessor,
    preprocess_spectrum_path,
)


def run_independent_inference(
    source_dir: Path | str,
    level_name: int | str,
    *,
    predictor: Predictor | None = None,
    model_run_dir: Path | str | None = None,
    input_dir: Path | str | None = None,
    one_dir: Path | str | None = None,
    top_k: int = 3,
    device: torch.device | str | None = None,
    evaluate_enable: bool = True,
    plot_train_mean_enable: bool = False,
    cached_test_root: Path | str | None = None,
    folder_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> Path:
    """运行独立文件夹推理并发布 `test_result/` 产物目录。"""
    target_device = predictor.device if predictor is not None else resolve_device(device)
    if predictor is None:
        predictor = load_predictor(
            source_dir,
            target_device,
            level_name,
            model_run_dir=model_run_dir,
        )
    preprocessor = build_inference_preprocessor(predictor.input_spec, target_device)
    cached_folders: list[CachedFolder] | None = None
    test_dir = _resolve_test_dir(predictor, input_dir)
    if cached_test_root is not None:
        if folder_names is None:
            raise ValueError("使用测试缓存时必须指定文件夹")
        selected_names = sorted({str(name) for name in folder_names})
        cached_folders = load_test_cache(cached_test_root, selected_names, predictor)
        input_dirs: list[Path] = []
    else:
        input_dirs = resolve_input_dirs(test_dir, one_dir)
    expected_lookup = build_expected_label_lookup(predictor.meta, predictor.predict_level)
    available_names = (
        [folder.folder_name for folder in cached_folders]
        if cached_folders is not None
        else [path.name for path in input_dirs]
    )
    selected_names, selection_rows = build_test_input_selection(
        available_names,
        predictor.meta,
        predictor.predict_level,
        predictor.resolve_target_class_names(),
    )
    if folder_names is not None:
        selected_names = {str(name) for name in folder_names}
        for row in selection_rows:
            row["selected"] = row["folder"] in selected_names
        if cached_folders is None:
            input_dirs = [path for path in input_dirs if path.name in selected_names]
    else:
        input_dirs = [path for path in input_dirs if path.name in selected_names]
    if not selected_names:
        raise FileNotFoundError(f"没有属于当前模型标签空间的推理文件夹：{test_dir}")
    if cached_folders is None and folder_names is not None and not input_dirs:
        raise FileNotFoundError(f"没有找到选定的推理文件夹：{sorted(selected_names)}")
    target_dir = _resolve_result_dir(predictor)
    temp_dir = target_dir.parent / f".{target_dir.name}_building_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True)
    try:
        write_input_selection(temp_dir / "input_selection.csv", selection_rows)
        rows = _run_folder_predictions(
            input_dirs,
            temp_dir,
            predictor,
            preprocessor,
            top_k,
            evaluate_enable,
            plot_train_mean_enable,
            expected_lookup,
            cached_folders,
        )
        summary = write_summary_report(temp_dir / "summary.txt", rows, evaluate_enable)
        write_summary_json(temp_dir / "summary.json", summary)
        write_run_manifest(
            temp_dir / "used_runs.json",
            "single_run" if predictor.run_dir is not None else "cascade",
            predictor.predict_level,
            predictor.build_used_runs(),
        )
        _publish_result_dir(temp_dir, target_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(f"[Saved] independent test results -> {target_dir}")
    return target_dir


def _run_folder_predictions(
    input_dirs: list[Path],
    output_dir: Path,
    predictor: Predictor,
    preprocessor,
    top_k: int,
    evaluate_enable: bool,
    plot_train_mean_enable: bool,
    expected_lookup: dict[str, str],
    cached_folders: list[CachedFolder] | None = None,
) -> list[dict[str, Any]]:
    """遍历所有输入文件夹，写入逐谱结果并返回有效汇总行。"""
    train_mean_bank = (
        _build_train_mean_bank(predictor, preprocessor)
        if plot_train_mean_enable
        else {}
    )
    class_names = predictor.resolve_target_class_names()
    rows: list[dict[str, Any]] = []
    if cached_folders is None:
        folder_inputs = [
            (folder_dir.name, [(path.name, path) for path in list_spectrum_paths(folder_dir)])
            for folder_dir in input_dirs
        ]
    else:
        folder_inputs = [
            (folder.folder_name, [(item.file_name, item.path) for item in folder.files])
            for folder in cached_folders
        ]
    for folder_name, spectrum_items in folder_inputs:
        row = _run_single_folder(
            folder_name,
            spectrum_items,
            output_dir,
            predictor,
            preprocessor,
            class_names,
            expected_lookup if evaluate_enable else {},
            evaluate_enable,
            top_k,
            train_mean_bank,
        )
        if row is not None:
            rows.append(row)
    return rows


def write_input_selection(output_path: Path, rows: list[dict[str, str | bool]]) -> None:
    """写入测试文件夹的标签匹配与筛选结果，供于追溯。"""
    fields = (
        "folder",
        "species_prefix",
        "target_level",
        "expected_label",
        "expected_in_model",
        "selected",
        "reason",
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_single_folder(
    folder_name: str,
    spectrum_items: list[tuple[str, Path]],
    output_dir: Path,
    predictor: Predictor,
    preprocessor,
    class_names: list[str],
    expected_lookup: dict[str, str],
    evaluate_enable: bool,
    top_k: int,
    train_mean_bank: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    """预测一个文件夹的全部光谱，并保存文本和对照图。"""
    predictions: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    for file_name, spectrum_path in spectrum_items:
        if spectrum_path.suffix.lower() == ".pt":
            payload = torch.load(spectrum_path, map_location="cpu")
            if not isinstance(payload, dict) or not isinstance(payload.get("input"), torch.Tensor):
                raise ValueError(f"测试缓存内容无效：{spectrum_path}")
            inputs = payload["input"]
        else:
            inputs = preprocess_spectrum_path(
                spectrum_path,
                preprocessor,
                predictor.input_config.bad_bands,
            )
        top_predictions = predictor.predict_tensor(inputs, top_k)
        predictions.append(
            {
                "file": file_name,
                "predictions": [
                    {
                        "label": item.label,
                        "probability": item.probability,
                        "class_id": item.class_id,
                    }
                    for item in top_predictions
                ],
                "top1_label": top_predictions[0].label,
            }
        )
        signals.append(inputs[0, 0].detach().cpu().numpy().astype(np.float32, copy=False))
    if not predictions:
        return None
    expected_label = (
        expected_lookup.get(parse_folder_prefix(folder_name, uppercase_enable=True))
        if evaluate_enable
        else None
    )
    row = build_folder_summary(folder_name, expected_label, class_names, predictions)
    row["file_predictions"] = predictions
    folder_output_dir = output_dir / folder_name
    folder_output_dir.mkdir(parents=True)
    evaluation_row = row if evaluate_enable and row["expected_in_model"] else None
    write_file_report(
        folder_output_dir / f"{folder_name}_file.txt",
        folder_name,
        predictions,
        evaluation_row,
    )
    values = np.stack(signals, axis=0)
    wavenumbers = np.linspace(
        predictor.input_config.cut_min,
        predictor.input_config.cut_max,
        values.shape[1],
        dtype=np.float32,
    )
    plot_folder_spectra(
        folder_output_dir / "spectra.png",
        folder_name,
        values,
        wavenumbers,
        predictor.input_config.bad_bands,
        row["expected_label"] if evaluation_row is not None else None,
        row["predicted_label"],
        train_mean_bank,
    )
    return row


def _resolve_test_dir(predictor: Predictor, input_dir: Path | str | None) -> Path:
    """解析显式输入目录；缺省时始终使用全局 ``dataset/init_test``。"""
    if input_dir is not None:
        return Path(input_dir).resolve()
    return resolve_test_root()


def _resolve_result_dir(predictor: Predictor) -> Path:
    """按实验或 run 模式解析基准一致的 `test_result/` 目录。"""
    if predictor.run_dir is not None:
        return predictor.run_dir / "test_result"
    return predictor.experiment_dir / predictor.predict_level / "test_result"


def _build_train_mean_bank(predictor: Predictor, preprocessor) -> dict[str, np.ndarray]:
    """按目标层级汇总训练集每个类别的均值输入谱。"""
    train_dir = resolve_training_dir(
        predictor.profile_id,
        DATASET_ROOT.parent,
        fallback_to_init_enable=False,
    )
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练目录不存在：{train_dir}")
    values_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    target_depth = int(predictor.predict_level.removeprefix("level_"))
    for spectrum_path in sorted(train_dir.rglob("*.arc_data")):
        relative_parts = spectrum_path.relative_to(train_dir).parts[:-1]
        if len(relative_parts) < target_depth:
            continue
        label = "/".join(relative_parts[:target_depth])
        inputs = preprocess_spectrum_path(
            spectrum_path,
            preprocessor,
            predictor.input_config.bad_bands,
        )
        values_by_label[label].append(
            inputs[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        )
    return {
        label: np.mean(np.stack(values, axis=0), axis=0)
        for label, values in values_by_label.items()
        if values
    }


def _publish_result_dir(temp_dir: Path, target_dir: Path) -> None:
    """将完整推理产物发布到目标目录，并保留此前结果副本。"""
    backup_dir = None
    if target_dir.exists():
        backup_dir = target_dir.parent / f"{target_dir.name}_previous_{uuid4().hex[:8]}"
        target_dir.replace(backup_dir)
    try:
        temp_dir.replace(target_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise


TEST_RESULT_DIR_NAME = "test_result"
TEST_SAMPLE_CSV_NAME = "test_eval_results.csv"
SUMMARY_JSON_NAME = "test_summary.json"
SUMMARY_TEXT_NAME = "test_summary.txt"
WEAK_CLASS_COUNT = 5
DEFAULT_TEST_LEVEL = "level_1"


def resolve_master_test_dir(project_root: Path | str | None = None) -> Path:
    """返回全局测试母本 ``dataset/init_test``（未预处理）。"""
    return resolve_test_root(project_root)


def resolve_preprocessed_test_dir(
    profile_id: str,
    project_root: Path | str | None = None,
) -> Path:
    """返回 profile 预处理后的独立测试集目录 ``dataset/<profile>/test``。"""
    profile = get_profile(profile_id)
    return get_dataset_dir(profile, project_root) / profile.root_test


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
    最深层再追加端到端 ``cascade``。测试样本按文件夹前缀对齐实验类别空间，
    对不上的整组计入忽略清单，不参与指标计算。
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
    """按自然序扫描扁平测试集，返回文件夹名、光谱路径与逐谱所属文件夹。"""
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
        folder_names.append(folder_dir.name)
        sample_paths.extend(paths)
        sample_folders.extend([folder_dir.name] * len(paths))
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
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
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
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
    "resolve_master_test_dir",
    "resolve_preprocessed_test_dir",
    "run_independent_inference",
    "select_deployment_record",
]
