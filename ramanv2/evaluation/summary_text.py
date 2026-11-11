"""测试/验证结果的人类可读 summary 统一格式。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SUMMARY_SEPARATOR = "=" * 37
FOLDER_TABLE_FIELDS = (
    "folder",
    "expected_label",
    "expected_in_model",
    "predicted_label",
    "majority_count",
    "total_count",
    "correct_count",
    "correct_ratio",
    "folder_correct",
)


def build_result_summary(
    metrics: Mapping[str, Any],
    level_name: str,
    mode: str,
    *,
    title: str = "独立测试集评估汇总",
    inference_payload: Mapping[str, Any] | None = None,
) -> str:
    """按模型指标和可选文件夹表生成一份 summary。"""
    summary = metrics.get("summary") or {}
    classes = metrics.get("classes") or []
    covered = sum(1 for item in classes if int(item.get("support", 0)) > 0)
    lines = [
        title,
        SUMMARY_SEPARATOR,
        f"target_level = {level_name}  mode = {mode}",
        f"Spectrum Accuracy = {float(summary.get('accuracy', 0.0)) * 100:.4f}%  "
        f"(support={int(summary.get('support', 0))})",
    ]
    folder = metrics.get("folder")
    if not isinstance(folder, Mapping) and inference_payload:
        folder = {
            "folder_accuracy": inference_payload.get("folder_accuracy", 0.0),
            "folder_correct": inference_payload.get("folder_correct_count", 0),
            "folder_count": inference_payload.get("evaluated_folder_count", 0),
        }
    if isinstance(folder, Mapping):
        lines.append(
            f"Folder Accuracy   = {float(folder.get('folder_accuracy', 0.0)) * 100:.4f}%  "
            f"({int(folder.get('folder_correct', 0))}/{int(folder.get('folder_count', 0))})"
        )
    lines.extend(
        [
            f"Class Coverage    = {covered}/{len(classes)}",
            f"Macro F1          = {float(summary.get('macro_f1', 0.0)) * 100:.4f}%",
            f"Macro Recall      = {float(summary.get('macro_recall', 0.0)) * 100:.4f}%",
            SUMMARY_SEPARATOR,
        ]
    )
    if inference_payload:
        table = build_folder_table(inference_payload)
        if table:
            lines.extend(["", table])
    return "\n".join(lines) + "\n"


def build_folder_table(payload: Mapping[str, Any]) -> str:
    """生成级联推理的逐文件夹表。"""
    rows = payload.get("rows") or []
    if not rows:
        return ""
    lines = ["\t".join(FOLDER_TABLE_FIELDS)]
    for row in rows:
        lines.append(
            "\t".join(
                (
                    str(row.get("folder", "")),
                    str(row.get("expected_label", "")),
                    str(bool(row.get("expected_in_model", False))),
                    str(row.get("predicted_label", "")),
                    str(row.get("majority_count", 0)),
                    str(row.get("total_count", 0)),
                    str(row.get("correct_count", 0)),
                    f"{float(row.get('correct_ratio', 0.0)):.6f}",
                    str(bool(row.get("folder_correct", False))),
                )
            )
        )
    return "\n".join(lines)


__all__ = ["build_result_summary", "build_folder_table"]
