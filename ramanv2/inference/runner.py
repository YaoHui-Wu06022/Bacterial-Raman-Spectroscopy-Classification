"""独立测试文件夹推理的完整编排入口。"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from ramanv2.core.paths import DATASET_ROOT
from ramanv2.common.naming import parse_test_folder_prefix
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.inference.directory import list_spectrum_paths, resolve_input_dirs
from ramanv2.inference.labels import build_expected_label_lookup, build_folder_summary
from ramanv2.inference.predictor import Predictor, load_predictor
from ramanv2.inference.report import (
    plot_folder_spectra,
    write_file_report,
    write_summary_report,
    write_used_runs,
)
from ramanv2.inference.spectra import (
    build_inference_preprocessor,
    preprocess_spectrum_path,
)


def run_independent_inference(
    source_dir: Path | str,
    level_name: int | str,
    *,
    input_dir: Path | str | None = None,
    one_dir: Path | str | None = None,
    top_k: int = 3,
    device: torch.device | str | None = None,
    evaluate_enable: bool = True,
    plot_train_mean_enable: bool = False,
    skip_transferred_enable: bool = False,
    transfer_manifest_path: Path | str | None = None,
) -> Path:
    """运行独立文件夹推理并发布 `test_result/` 产物目录。"""
    target_device = _resolve_device(device)
    predictor = load_predictor(source_dir, target_device, level_name)
    preprocessor = build_inference_preprocessor(predictor.input_spec, target_device)
    test_dir = _resolve_test_dir(predictor, input_dir)
    input_dirs = resolve_input_dirs(test_dir, one_dir)
    if not input_dirs:
        raise FileNotFoundError(f"推理输入目录没有子文件夹：{test_dir}")
    target_dir = _resolve_result_dir(predictor)
    temp_dir = target_dir.parent / f".{target_dir.name}_building_{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True)
    try:
        rows = _run_folder_predictions(
            input_dirs,
            temp_dir,
            predictor,
            preprocessor,
            top_k,
            evaluate_enable,
            plot_train_mean_enable,
            _load_skip_lookup(
                _resolve_manifest_path(predictor, transfer_manifest_path)
                if skip_transferred_enable
                else None
            ),
        )
        write_summary_report(temp_dir / "summary.txt", rows, evaluate_enable)
        write_used_runs(
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
    skip_lookup: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """遍历所有输入文件夹，写入逐谱结果并返回有效汇总行。"""
    expected_lookup = (
        build_expected_label_lookup(predictor.meta, predictor.predict_level)
        if evaluate_enable
        else {}
    )
    train_mean_bank = (
        _build_train_mean_bank(predictor, preprocessor)
        if plot_train_mean_enable
        else {}
    )
    class_names = predictor.resolve_target_class_names()
    rows: list[dict[str, Any]] = []
    skipped_rows: list[str] = []
    for folder_dir in input_dirs:
        row, skipped = _run_single_folder(
            folder_dir,
            output_dir,
            predictor,
            preprocessor,
            class_names,
            expected_lookup,
            evaluate_enable,
            top_k,
            train_mean_bank,
            skip_lookup,
        )
        if row is not None:
            rows.append(row)
        skipped_rows.extend(skipped)
    if skipped_rows:
        (output_dir / "skipped_transferred_samples.txt").write_text(
            "\n".join(skipped_rows) + "\n",
            encoding="utf-8",
        )
    return rows


def _run_single_folder(
    folder_dir: Path,
    output_dir: Path,
    predictor: Predictor,
    preprocessor,
    class_names: list[str],
    expected_lookup: dict[str, str],
    evaluate_enable: bool,
    top_k: int,
    train_mean_bank: dict[str, np.ndarray],
    skip_lookup: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """预测一个文件夹的全部光谱，并保存文本和对照图。"""
    predictions: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    skipped_rows: list[str] = []
    skip_files = skip_lookup.get(folder_dir.name, set())
    for spectrum_path in list_spectrum_paths(folder_dir):
        if spectrum_path.name in skip_files:
            skipped_rows.append(f"{folder_dir.name}/{spectrum_path.name}")
            continue
        inputs = preprocess_spectrum_path(
            spectrum_path,
            preprocessor,
            predictor.input_config.bad_bands,
        )
        top_predictions = predictor.predict_tensor(inputs, top_k)
        predictions.append(
            {
                "file": spectrum_path.name,
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
        return None, skipped_rows
    expected_label = (
        expected_lookup.get(parse_test_folder_prefix(folder_dir.name))
        if evaluate_enable
        else None
    )
    row = build_folder_summary(folder_dir.name, expected_label, class_names, predictions)
    folder_output_dir = output_dir / folder_dir.name
    folder_output_dir.mkdir(parents=True)
    write_file_report(
        folder_output_dir / f"{folder_dir.name}_file.txt",
        folder_dir.name,
        predictions,
        row if evaluate_enable else None,
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
        folder_dir.name,
        values,
        wavenumbers,
        predictor.input_config.bad_bands,
        row["expected_label"] if evaluate_enable else None,
        row["predicted_label"],
        train_mean_bank,
    )
    return row, skipped_rows


def _resolve_test_dir(predictor: Predictor, input_dir: Path | str | None) -> Path:
    """解析显式输入目录或 profile 对应的独立测试目录。"""
    if input_dir is not None:
        return Path(input_dir).resolve()
    profile = get_profile(predictor.profile_id)
    if profile.profile_id == "alldata":
        return get_dataset_dir(get_profile("test"), DATASET_ROOT.parent) / "init"
    return get_dataset_dir(profile, DATASET_ROOT.parent) / profile.root_test


def _resolve_result_dir(predictor: Predictor) -> Path:
    """按实验或 run 模式解析基准一致的 `test_result/` 目录。"""
    if predictor.run_dir is not None:
        return predictor.run_dir / "test_result"
    return predictor.experiment_dir / predictor.predict_level / "test_result"


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """选择用户指定设备，缺省时优先使用可用 CUDA。"""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_manifest_path(
    predictor: Predictor,
    manifest_path: Path | str | None,
) -> Path:
    """解析迁移样本清单的显式路径或测试菌默认清单路径。"""
    if manifest_path is not None:
        return Path(manifest_path).resolve()
    return get_dataset_dir(get_profile("test"), DATASET_ROOT.parent) / "test_transfer_manifest.csv"


def _load_skip_lookup(manifest_path: Path | None) -> dict[str, set[str]]:
    """读取测试菌迁移清单，建立需跳过的源文件集合。"""
    if manifest_path is None:
        return {}
    if not manifest_path.is_file():
        raise FileNotFoundError(f"迁移清单不存在：{manifest_path}")
    lookup: dict[str, set[str]] = defaultdict(set)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            folder = (row.get("source_folder") or "").strip()
            filename = (row.get("source_file") or "").strip()
            if folder and filename:
                lookup[folder].add(filename)
    return dict(lookup)


def _build_train_mean_bank(predictor: Predictor, preprocessor) -> dict[str, np.ndarray]:
    """按目标层级汇总训练集每个类别的均值输入谱。"""
    profile = get_profile(predictor.profile_id)
    train_dir = get_dataset_dir(profile, DATASET_ROOT.parent) / profile.root_train_clean
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
