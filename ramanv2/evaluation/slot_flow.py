"""单模型槽位的验证集模型与 PCA-SVM 基线评估。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramanv2.evaluation.result_text import build_result_summary
from ramanv2.experiments.run_manifest import resolve_slot_target
from ramanv2.evaluation.model_eval import evaluate_model_run
from ramanv2.evaluation.pca_svm import BaselineSpec, evaluate_baseline_run


@dataclass(frozen=True)
class SlotEvaluation:
    """一个模型槽位的模型、基线结果目录与指标；未请求的一侧为 None。"""

    slot_dir: Path
    level_name: str
    parent_id: int | None
    model_result_dir: Path | None
    baseline_result_dir: Path | None
    model_metrics: dict[str, Any] | None
    baseline_metrics: dict[str, Any] | None


def evaluate_slot_bundle(
    slot_dir: Path | str,
    *,
    include_model: bool = True,
    include_baseline: bool = True,
    level_name: str | None = None,
    device: str | None = None,
    baseline_spec: BaselineSpec | None = None,
    train_dir: Path | str | None = None,
) -> SlotEvaluation:
    """按开关评估槽位的模型与/或 PCA-SVM 基线。"""
    target_dir = Path(slot_dir).resolve()
    resolved_level, parent_id = resolve_slot_target(target_dir)
    target_level = resolved_level if level_name is None else str(level_name)
    model_result_dir = None
    model_metrics = None
    if include_model:
        model_result_dir = evaluate_model_run(
            target_dir,
            target_level,
            device,
            train_dir=train_dir,
        )
        model_metrics = _read_metrics(model_result_dir)
    baseline_result_dir = None
    baseline_metrics = None
    if include_baseline:
        baseline_result_dir = evaluate_baseline_run(
            target_dir,
            target_level,
            baseline_spec or BaselineSpec(),
        )
        baseline_metrics = _read_metrics(baseline_result_dir)
    return SlotEvaluation(
        slot_dir=target_dir,
        level_name=resolved_level,
        parent_id=parent_id,
        model_result_dir=model_result_dir,
        baseline_result_dir=baseline_result_dir,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
    )


def evaluate_slot_with_baseline(
    slot_dir: Path | str,
    device: str | None = None,
    baseline_spec: BaselineSpec | None = None,
) -> SlotEvaluation:
    """自动识别槽位层级并评估模型与 PCA-SVM 基线。"""
    return evaluate_slot_bundle(
        slot_dir,
        device=device,
        baseline_spec=baseline_spec,
    )


def _read_metrics(result_dir: Path) -> dict[str, Any]:
    return json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))


def format_test_report(report: dict[str, Any]) -> str:
    """把独立测试结果字典格式化为可读文本。"""
    result_dir = Path(report["result_dir"])
    metrics = _read_metrics(result_dir)
    header = (
        f"slot_dir = {report['run_dir']}  level_name = {report['level_name']}  "
        f"parent_id = {report['parent_id']}\n\n"
    )
    return header + build_result_summary(
        metrics,
        report["level_name"],
        "model",
        title="模型独立测试",
    ).rstrip()


__all__ = [
    "SlotEvaluation",
    "evaluate_slot_bundle",
    "evaluate_slot_with_baseline",
    "format_test_report",
]
