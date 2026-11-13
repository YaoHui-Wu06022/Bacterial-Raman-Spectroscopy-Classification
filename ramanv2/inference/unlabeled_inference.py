"""实际样本推理：按日期现场仿射并输出每个样本文件夹的无标签预测。"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.device import resolve_device
from ramanv2.common.file_ops import publish_directory, write_json
from ramanv2.common.naming import SPECIAL_FOLDER_NAMES, build_natural_key
from ramanv2.data.calibration.affine import analyze_date
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.preprocess import build_preprocess_kwargs
from ramanv2.inference.spectrum_input import (
    build_inference_preprocessor,
    list_spectrum_paths,
)
from ramanv2.modeling.predictor import Predictor, load_predictor
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.preprocess import preprocess_single_spectrum


INFERENCE_DATA_DIR_NAME = "inference_data"
UNLABELED_RESULT_DIR_NAME = "result"
AFFINE_VALID_STATUSES = frozenset({"ready", "limited_replicates"})


def resolve_inference_data_root(dataset_dir: Path | str) -> Path:
    """返回数据集的 inference_data 根目录。"""
    return Path(dataset_dir).resolve() / INFERENCE_DATA_DIR_NAME


def list_inference_dates(dataset_dir: Path | str) -> list[Path]:
    """按自然序列出 inference_data 下的日期目录。"""
    root = resolve_inference_data_root(dataset_dir)
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path.name != UNLABELED_RESULT_DIR_NAME
        ),
        key=lambda path: build_natural_key(path.name),
    )


def resolve_date_result_dir(dataset_dir: Path | str, date_name: str) -> Path:
    """返回一个日期的实际推理结果目录。"""
    return (
        resolve_inference_data_root(dataset_dir)
        / UNLABELED_RESULT_DIR_NAME
        / date_name
    )


def run_unlabeled_inference(
    experiment_dir: Path | str,
    level_name: str,
    date_dir: Path | str,
    *,
    predictor: Predictor | None = None,
    device: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """对 inference_data 的一个日期执行无标签实际推理。

    日期级仿射由当天小球现场拟合；只接受 ready / limited_replicates。
    每个 CS 文件夹使用全部光谱，文件夹结果取 top-1 多数票，不做准确率评估。
    """
    source = Path(date_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"推理日期目录不存在：{source}")
    result_root = source.parent / UNLABELED_RESULT_DIR_NAME
    target_device = predictor.device if predictor is not None else resolve_device(device)
    if predictor is None:
        predictor = load_predictor(experiment_dir, target_device, level_name)
    elif predictor.run_dir is not None:
        raise ValueError("实际推理请从实验根加载级联模型")

    affine = analyze_date(source, result_root)
    status = str(affine.get("status", ""))
    if status not in AFFINE_VALID_STATUSES:
        raise RuntimeError(
            f"{source.name} 小球仿射未通过验收：{status or 'unknown'}；"
            f"详情见 {result_root / source.name / 'shift' / 'affine.json'}"
        )
    scale = affine.get("scale")
    offset = affine.get("offset")
    if scale is None or offset is None:
        raise RuntimeError(f"{source.name} 仿射参数缺失")

    folders = _sample_folders(source)
    if not folders:
        raise RuntimeError(f"{source} 中没有可推理的样本文件夹")
    input_config = predictor.input_config
    reference_axis = build_wn_ref(
        input_config.cut_min,
        input_config.cut_max,
        input_config.target_points,
    )
    options = build_preprocess_kwargs(
        predictor.profile_id,
        input_config,
        DataBuildConfig(),
        reference_axis,
    )
    preprocessor = build_inference_preprocessor(predictor.input_spec, predictor.device)

    result_dir = result_root / source.name
    temporary = result_root / f".{source.name}_building_{uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        shift_dir = result_dir / "shift"
        if shift_dir.is_dir():
            shutil.copytree(shift_dir, temporary / "shift")
        folder_reports = []
        for folder in folders:
            report = _predict_folder(
                folder,
                scale=float(scale),
                offset=float(offset),
                predictor=predictor,
                preprocessor=preprocessor,
                options=options,
                top_k=max(int(top_k), 1),
            )
            _write_folder_outputs(temporary / folder.name, report)
            folder_reports.append(
                {
                    key: report[key]
                    for key in (
                        "folder",
                        "status",
                        "predicted_label",
                        "majority_count",
                        "total_count",
                        "failed_count",
                        "majority_ratio",
                        "mean_top1_probability",
                        "label_votes",
                    )
                }
            )
        summary = {
            "date": source.name,
            "experiment_dir": str(Path(experiment_dir).resolve()),
            "target_level": str(level_name),
            "model_class_count": len(predictor.resolve_target_class_names()),
            "top_k": max(int(top_k), 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "affine": {
                key: affine.get(key)
                for key in ("status", "scale", "offset", "valid_repeat_count", "errors")
            },
            "folder_count": len(folder_reports),
            "succeeded_folder_count": sum(
                1 for report in folder_reports if report["status"] == "ok"
            ),
            "failed_folder_count": sum(
                1 for report in folder_reports if report["status"] != "ok"
            ),
            "result_dir": str(result_dir),
            "folders": folder_reports,
        }
        _write_summary(temporary, summary)
        publish_directory(temporary, result_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _sample_folders(date_dir: Path) -> list[Path]:
    """列出日期下含光谱的样本文件夹，跳过小球和流水线保留目录。"""
    result = []
    for path in sorted(date_dir.iterdir(), key=lambda item: build_natural_key(item.name)):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.name in SPECIAL_FOLDER_NAMES:
            continue
        if not list_spectrum_paths(path):
            continue
        result.append(path)
    return result


def _predict_folder(
    folder: Path,
    *,
    scale: float,
    offset: float,
    predictor: Predictor,
    preprocessor,
    options: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """推理一个样本文件夹的全部光谱，并聚合 top-1 多数票。"""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in list_spectrum_paths(folder):
        try:
            wavenumbers, intensity = read_arc_data(path)
            corrected_axis = np.asarray(wavenumbers, dtype=np.float64) * scale + offset
            output_axis, spectrum, _stats = preprocess_single_spectrum(
                corrected_axis,
                intensity,
                **options,
            )
            if output_axis is None or spectrum is None:
                raise ValueError("预处理未产生有效光谱")
            inputs = preprocessor.preprocess_intensity(spectrum)
            predictions = predictor.predict_tensor(inputs, top_k=top_k)
            rows.append(
                {
                    "file": path.name,
                    "predictions": [
                        {
                            "label": prediction.label,
                            "probability": float(prediction.probability),
                            "class_id": int(prediction.class_id),
                        }
                        for prediction in predictions
                    ],
                }
            )
        except Exception as error:
            failures.append({"file": path.name, "error": str(error)})

    votes: dict[str, dict[str, float | int]] = {}
    for row in rows:
        top = row["predictions"][0]
        item = votes.setdefault(
            top["label"],
            {"count": 0, "probability_sum": 0.0},
        )
        item["count"] = int(item["count"]) + 1
        item["probability_sum"] = float(item["probability_sum"]) + top["probability"]
    ranked = sorted(
        votes.items(),
        key=lambda item: (
            -int(item[1]["count"]),
            -float(item[1]["probability_sum"]) / int(item[1]["count"]),
            item[0],
        ),
    )
    predicted_label = ranked[0][0] if ranked else ""
    majority_count = int(ranked[0][1]["count"]) if ranked else 0
    total_count = len(rows)
    mean_probability = (
        sum(row["predictions"][0]["probability"] for row in rows) / total_count
        if total_count
        else 0.0
    )
    return {
        "folder": folder.name,
        "status": "ok" if rows else "failed",
        "predicted_label": predicted_label,
        "majority_count": majority_count,
        "total_count": total_count,
        "failed_count": len(failures),
        "majority_ratio": majority_count / total_count if total_count else 0.0,
        "mean_top1_probability": mean_probability,
        "label_votes": [
            {
                "label": label,
                "count": int(item["count"]),
                "mean_probability": float(item["probability_sum"]) / int(item["count"]),
            }
            for label, item in ranked
        ],
        "rows": rows,
        "failures": failures,
    }


def _write_folder_outputs(folder_dir: Path, report: dict[str, Any]) -> None:
    """写入一个 CS 文件夹的逐谱预测与可读摘要。"""
    folder_dir.mkdir(parents=True, exist_ok=True)
    max_count = max(
        (len(row["predictions"]) for row in report["rows"]),
        default=0,
    )
    header = ["file", "status", "error"]
    for index in range(max_count):
        header.extend([f"top{index + 1}_label", f"top{index + 1}_probability"])
    with (folder_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in report["rows"]:
            values = [row["file"], "ok", ""]
            for prediction in row["predictions"]:
                values.extend([prediction["label"], f"{prediction['probability']:.6f}"])
            values.extend([""] * (len(header) - len(values)))
            writer.writerow(values)
        for failure in report["failures"]:
            values = [failure["file"], "failed", failure["error"]]
            values.extend([""] * (len(header) - len(values)))
            writer.writerow(values)
    lines = [
        "样本文件夹推理结果",
        f"folder = {report['folder']}",
        f"status = {report['status']}",
        f"predicted_label = {report['predicted_label']}",
        f"majority = {report['majority_count']}/{report['total_count']}",
        f"majority_ratio = {report['majority_ratio'] * 100:.2f}%",
        f"mean_top1_probability = {report['mean_top1_probability'] * 100:.2f}%",
        f"failed_spectra = {report['failed_count']}",
        "",
        "label votes:",
    ]
    lines.extend(
        f"  {item['label']}: {item['count']}  "
        f"(mean p={item['mean_probability'] * 100:.2f}%)"
        for item in report["label_votes"]
    )
    (folder_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(result_dir: Path, summary: dict[str, Any]) -> None:
    """写入日期级汇总 JSON、CSV 和文本。"""
    write_json(result_dir / "summary.json", summary)
    columns = [
        "folder",
        "status",
        "predicted_label",
        "majority_count",
        "total_count",
        "majority_ratio",
        "mean_top1_probability",
        "failed_count",
    ]
    with (result_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for folder in summary["folders"]:
            writer.writerow(folder)
    affine = summary["affine"]
    lines = [
        "实际样本推理汇总",
        f"date = {summary['date']}",
        f"experiment_dir = {summary['experiment_dir']}",
        f"target_level = {summary['target_level']}",
        f"model_class_count = {summary['model_class_count']}",
        f"affine_status = {affine.get('status')}",
        f"affine_scale = {affine.get('scale')}",
        f"affine_offset = {affine.get('offset')}",
        f"folders = {summary['folder_count']}",
        f"succeeded_folders = {summary['succeeded_folder_count']}",
        f"failed_folders = {summary['failed_folder_count']}",
        "",
        f"{'folder':<20}{'status':<10}{'predicted_label':<28}"
        f"{'majority':>10}{'total':>8}{'mean_p':>10}{'failed':>8}",
    ]
    for folder in summary["folders"]:
        lines.append(
            f"{folder['folder']:<20}{folder['status']:<10}"
            f"{folder['predicted_label']:<28}"
            f"{folder['majority_count']:>10}{folder['total_count']:>8}"
            f"{folder['mean_top1_probability'] * 100:>9.2f}%"
            f"{folder['failed_count']:>8}"
        )
    (result_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "UNLABELED_RESULT_DIR_NAME",
    "INFERENCE_DATA_DIR_NAME",
    "list_inference_dates",
    "resolve_date_result_dir",
    "resolve_inference_data_root",
    "run_unlabeled_inference",
]
