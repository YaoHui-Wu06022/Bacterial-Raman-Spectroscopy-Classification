"""测试文件夹预期标签解析与文件夹级统计。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from raman_temp.common.naming import parse_test_folder_prefix
from raman_temp.core.hierarchy import normalize_level_name


def build_expected_label_lookup(
    meta: Mapping[str, Any],
    level_name: str,
) -> dict[str, str]:
    """从层级类别名构建测试菌前缀到目标层标签的唯一映射。"""
    prefixes: dict[str, set[str]] = {}
    target_depth = int(normalize_level_name(level_name).removeprefix("level_"))
    for labels in (meta.get("class_names_by_level") or {}).values():
        for source_label in labels:
            parts = [part for part in str(source_label).split("/") if part]
            if len(parts) < target_depth:
                continue
            prefix = parse_test_folder_prefix(parts[-1])
            target_label = "/".join(parts[:target_depth])
            prefixes.setdefault(prefix, set()).add(target_label)
    return {
        prefix: next(iter(labels))
        for prefix, labels in prefixes.items()
        if len(labels) == 1
    }


def build_folder_summary(
    folder_name: str,
    expected_label: str | None,
    class_names: list[str],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总文件夹多数预测、逐谱正确数和文件夹正确性。"""
    counter = Counter(item["top1_label"] for item in predictions)
    predicted_label, majority_count = counter.most_common(1)[0] if counter else ("unknown", 0)
    expected_in_model = expected_label in set(class_names)
    correct_count = (
        sum(item["top1_label"] == expected_label for item in predictions)
        if expected_in_model
        else 0
    )
    total_count = len(predictions)
    return {
        "folder": folder_name,
        "expected_label": expected_label or "unknown",
        "expected_in_model": bool(expected_in_model),
        "predicted_label": predicted_label,
        "majority_count": int(majority_count),
        "total_count": total_count,
        "correct_count": int(correct_count),
        "correct_ratio": correct_count / total_count if total_count else 0.0,
        "folder_correct": bool(expected_in_model and predicted_label == expected_label),
    }
