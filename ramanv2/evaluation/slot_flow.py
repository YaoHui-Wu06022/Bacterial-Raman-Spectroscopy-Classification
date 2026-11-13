"""单模型槽位的验证集模型与 PCA-SVM 基线评估。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramanv2.evaluation.context import resolve_slot_target
from ramanv2.evaluation.model_eval import evaluate_model_run
from ramanv2.evaluation.pca_svm import BaselineSpec, evaluate_baseline_run
from ramanv2.evaluation.summary_text import build_result_summary


@dataclass(frozen=True)
class SlotEvaluation:
    """一个模型槽位的模型、基线结果目录与指标。"""

    slot_dir: Path
    level_name: str
    parent_id: int | None
    model_result_dir: Path
    baseline_result_dir: Path
    model_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]


def evaluate_slot_with_baseline(
    slot_dir: Path | str,
    device: str | None = None,
    baseline_spec: BaselineSpec | None = None,
) -> SlotEvaluation:
    """自动识别槽位层级并评估模型与 PCA-SVM 基线。"""
    target_dir = Path(slot_dir).resolve()
    level_name, parent_id = resolve_slot_target(target_dir)
    model_result_dir = evaluate_model_run(target_dir, level_name, device)
    baseline_result_dir = evaluate_baseline_run(
        target_dir,
        level_name,
        baseline_spec or BaselineSpec(),
    )
    return SlotEvaluation(
        slot_dir=target_dir,
        level_name=level_name,
        parent_id=parent_id,
        model_result_dir=model_result_dir,
        baseline_result_dir=baseline_result_dir,
        model_metrics=_read_metrics(model_result_dir),
        baseline_metrics=_read_metrics(baseline_result_dir),
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
    "evaluate_slot_with_baseline",
    "format_test_report",
]
