"""独立文件夹推理的完整编排入口。"""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from ramanv2.common.device import resolve_device
from ramanv2.common.file_ops import publish_directory
from ramanv2.common.hierarchy import load_hierarchy_meta
from ramanv2.common.paths import find_ancestor_dir
from ramanv2.core.run_context import (
    CASCADE_MODE_DIR_NAME,
    HIERARCHY_META_FILE_NAME,
    INFERENCE_RESULT_DIR_NAME,
    LEVEL_SUMMARY_DIR_NAME,
    METRICS_FILE_NAME,
    MODEL_FILE_NAME,
    PARENT_KNOWN_MODE_DIR_NAME,
    TEST_RESULT_DIR_NAME,
)
from ramanv2.data.hierarchy_config import HierarchySpec, resolve_hierarchy_from_meta
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.profiles import resolve_selection
from ramanv2.evaluation.summary_text import build_result_summary
from ramanv2.inference.folder_report import (
    plot_folder_spectra,
    write_file_report,
    write_summary_json,
    write_summary_report,
)
from ramanv2.inference.input_cache import CachedFolder, load_test_cache
from ramanv2.modeling.predictor import Predictor, load_predictor
from ramanv2.modeling.run_manifest import has_model_path, resolve_entry_path
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.bad_bands import build_valid_mask
from ramanv2.inference.spectrum_input import (
    build_inference_preprocessor,
    list_spectrum_paths,
    preprocess_spectrum_path,
    resolve_input_dirs,
    sample_folder_spectra,
)
from ramanv2.inference.test_labels import (
    build_folder_summary,
    build_test_input_selection,
)


SUMMARY_TEXT_NAME = "summary.txt"
SUMMARY_JSON_NAME = "summary.json"
SPECTRA_FIGURE_NAME = "spectra.png"

def run_independent_inference(
    source_dir: Path | str,
    level_name: int | str,
    *,
    predictor: Predictor | None = None,
    input_dir: Path | str | None = None,
    one_dir: Path | str | None = None,
    top_k: int = 3,
    device: torch.device | str | None = None,
    evaluate_enable: bool = True,
    plot_train_mean_enable: bool = False,
    cached_test_root: Path | str | None = None,
    folder_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> Path:
    """运行层级级联样本推理并发布 `inference_result/` 产物目录。

    每个文件夹按 cell 分组随机抽取最多 ``INFERENCE_SAMPLES_PER_FOLDER`` 条，
    同一 cell 至多一条。
    """
    target_device = predictor.device if predictor is not None else resolve_device(device)
    if predictor is None:
        experiment_dir = find_ancestor_dir(source_dir, HIERARCHY_META_FILE_NAME)
        predictor = load_predictor(experiment_dir, target_device, level_name)
    elif predictor.run_dir is not None:
        raise ValueError("inference_result 只记录级联预测，请从实验根加载模型")
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
    expected_hierarchy = resolve_hierarchy_from_meta(predictor.meta)
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
            expected_hierarchy,
            cached_folders,
        )
        summary = write_summary_report(temp_dir / SUMMARY_TEXT_NAME, rows, evaluate_enable)
        write_summary_json(temp_dir / SUMMARY_JSON_NAME, summary)
        publish_directory(temp_dir, target_dir)
        merge_level_test_inference_summary(
            predictor.experiment_dir,
            predictor.meta,
            predictor.predict_level,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(f"[Saved] cascade inference results -> {target_dir}")
    return target_dir


def run_experiment_inference(
    source_dir: Path | str,
    input_dir: Path | str,
    levels: tuple[str, ...] = ("level_1", "level_2"),
    *,
    device: torch.device | str | None = None,
    seed: int | None = 42,
) -> dict[str, Path]:
    """按层级生成级联 inference_result；跳过未训练完整的层级。"""
    experiment_dir = find_ancestor_dir(source_dir, HIERARCHY_META_FILE_NAME)
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    if meta is None:
        raise FileNotFoundError(f"缺少 {HIERARCHY_META_FILE_NAME}：{experiment_dir}")
    test_dir = Path(input_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"测试目录不存在：{test_dir}")
    head_names = tuple(meta.get("head_names") or [])
    results: dict[str, Path] = {}
    for level_name in levels:
        if level_name not in head_names:
            print(f"[Inference] 跳过 {level_name}：实验未训练该层级")
            continue
        level_index = head_names.index(level_name)
        prerequisites = head_names[:level_index]
        if any(
            not _level_models_ready(meta, experiment_dir, prerequisite)
            for prerequisite in prerequisites
        ):
            print(f"[Inference] 跳过 {level_name}：上级模型未就绪")
            continue
        if not _level_models_ready(meta, experiment_dir, level_name):
            print(f"[Inference] 跳过 {level_name}：模型槽位未补齐")
            continue
        if _is_direct_pass_level(meta, level_name):
            print(f"[Inference] 跳过 {level_name}：单类层级直通，无模型产物")
            continue
        if seed is not None:
            random.seed(seed)
        results[level_name] = run_independent_inference(
            experiment_dir,
            level_name,
            input_dir=test_dir,
            device=device,
        )
    return results


def _level_models_ready(
    meta: dict[str, Any],
    experiment_dir: Path,
    level_name: str,
) -> bool:
    """判断一个层级的全局模型或全部多子类父模型是否已落盘。"""
    global_entry = (meta.get("level_models") or {}).get(level_name)
    if has_model_path(global_entry):
        return _entry_model_exists(experiment_dir, global_entry)
    parent_map = (meta.get("parent_to_children") or {}).get(level_name) or {}
    parent_models = (meta.get("parent_models") or {}).get(level_name) or {}
    if not parent_map:
        return _is_direct_pass_level(meta, level_name)
    for parent_text, children in parent_map.items():
        if len(children) <= 1:
            continue
        entry = parent_models.get(str(parent_text))
        if not has_model_path(entry):
            return False
        if not _entry_model_exists(experiment_dir, entry):
            return False
    return True


def _is_direct_pass_level(meta: dict[str, Any], level_name: str) -> bool:
    """判断层级是否只有一个类别且没有该层模型，可直接透传。"""
    class_names = (meta.get("class_names_by_level") or {}).get(level_name) or []
    if len(class_names) != 1:
        return False
    global_entry = (meta.get("level_models") or {}).get(level_name)
    if has_model_path(global_entry):
        return False
    parent_entries = (meta.get("parent_models") or {}).get(level_name) or {}
    return not any(has_model_path(entry) for entry in parent_entries.values())


def _entry_model_exists(experiment_dir: Path, entry: Any) -> bool:
    slot_dir = resolve_entry_path(experiment_dir, entry, "run_dir")
    return (slot_dir / MODEL_FILE_NAME).is_file()


def _run_folder_predictions(
    input_dirs: list[Path],
    output_dir: Path,
    predictor: Predictor,
    preprocessor,
    top_k: int,
    evaluate_enable: bool,
    plot_train_mean_enable: bool,
    expected_hierarchy: HierarchySpec,
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
            sample_folder_spectra(spectrum_items),
            output_dir,
            predictor,
            preprocessor,
            class_names,
            expected_hierarchy if evaluate_enable else None,
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
    expected_hierarchy: HierarchySpec | None,
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
    expected_label = None
    if evaluate_enable and expected_hierarchy is not None:
        try:
            expected_label = expected_hierarchy.expected_label(folder_name, predictor.predict_level)
        except ValueError:
            expected_label = None
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
    wavenumbers = build_wn_ref(
        predictor.input_config.cut_min,
        predictor.input_config.cut_max,
        predictor.input_config.target_points,
    ).astype(np.float32)
    valid_mask = build_valid_mask(wavenumbers, predictor.input_config.bad_bands)
    if valid_mask is not None:
        wavenumbers = wavenumbers[valid_mask]
    if wavenumbers.size != values.shape[1]:
        raise ValueError("推理绘图轴长度与模型输入不一致")
    plot_folder_spectra(
        folder_output_dir / SPECTRA_FIGURE_NAME,
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
    """解析显式输入目录；缺省时使用实验所属数据集的 init_test。"""
    if input_dir is not None:
        return Path(input_dir).resolve()
    return resolve_selection(predictor.dataset_id, predictor.profile_id).test_root


def resolve_cascade_test_result_dir(
    experiment_dir: Path | str,
    meta: dict[str, Any],
    level_name: str,
) -> Path:
    """返回从头级联模式的测试结果目录。"""
    level_dir = Path(experiment_dir) / level_name
    global_entry = (meta.get("level_models") or {}).get(level_name)
    if has_model_path(global_entry):
        return level_dir / TEST_RESULT_DIR_NAME
    return level_dir / LEVEL_SUMMARY_DIR_NAME / CASCADE_MODE_DIR_NAME / TEST_RESULT_DIR_NAME


def resolve_parent_known_test_result_dir(
    experiment_dir: Path | str,
    meta: dict[str, Any],
    level_name: str,
) -> Path:
    """返回已知上层模式的测试结果目录。"""
    return (
        Path(experiment_dir)
        / level_name
        / LEVEL_SUMMARY_DIR_NAME
        / PARENT_KNOWN_MODE_DIR_NAME
        / TEST_RESULT_DIR_NAME
    )


def resolve_level_inference_dir(
    experiment_dir: Path | str,
    meta: dict[str, Any],
    level_name: str,
) -> Path:
    """返回挂在从头级联 test_result 下的 inference_result 目录。"""
    return resolve_cascade_test_result_dir(experiment_dir, meta, level_name) / INFERENCE_RESULT_DIR_NAME


def merge_level_test_inference_summary(
    experiment_dir: Path | str,
    meta: dict[str, Any],
    level_name: str,
) -> None:
    """把模型测试指标与级联文件夹表合并为外层 test_result/summary.txt。"""
    test_result_dir = resolve_cascade_test_result_dir(experiment_dir, meta, level_name)
    metrics_path = test_result_dir / METRICS_FILE_NAME
    if not metrics_path.is_file():
        return
    inference_dir = resolve_level_inference_dir(experiment_dir, meta, level_name)
    summary_json = inference_dir / SUMMARY_JSON_NAME
    if not summary_json.is_file():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    global_entry = (meta.get("level_models") or {}).get(level_name)
    mode = "model" if has_model_path(global_entry) else CASCADE_MODE_DIR_NAME
    (test_result_dir / SUMMARY_TEXT_NAME).write_text(
        build_result_summary(
            metrics,
            level_name,
            mode,
            inference_payload=payload,
        ),
        encoding="utf-8",
    )
    (inference_dir / SUMMARY_TEXT_NAME).unlink(missing_ok=True)


def _resolve_result_dir(predictor: Predictor) -> Path:
    """返回层级级联推理结果目录。"""
    return resolve_level_inference_dir(
        predictor.experiment_dir,
        predictor.meta,
        predictor.predict_level,
    )


def _build_train_mean_bank(predictor: Predictor, preprocessor) -> dict[str, np.ndarray]:
    """按目标层级汇总训练集每个类别的均值输入谱。"""
    selection = resolve_selection(predictor.dataset_id, predictor.profile_id)
    train_dir = selection.dataset_dir / selection.profile.root_train_clean
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练目录不存在：{train_dir}")
    values_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    index = DatasetIndex(
        train_dir,
        hierarchy=resolve_hierarchy_from_meta(predictor.meta),
        load_intensity_enable=False,
    )
    for sample_index, spectrum_path in enumerate(index.samples):
        label = index.get_level_key(sample_index, predictor.predict_level)
        if label is None:
            continue
        inputs = preprocess_spectrum_path(
            Path(spectrum_path),
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


__all__ = [
    "merge_level_test_inference_summary",
    "resolve_cascade_test_result_dir",
    "resolve_level_inference_dir",
    "resolve_parent_known_test_result_dir",
    "run_experiment_inference",
    "run_independent_inference",
]