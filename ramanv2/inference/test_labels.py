"""测试文件夹预期标签解析与文件夹级统计。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ramanv2.common.naming import parse_folder_prefix
from ramanv2.data.hierarchy_spec import resolve_hierarchy_from_meta


def build_test_input_selection(
    folder_names: list[str],
    meta: Mapping[str, Any],
    level_name: str,
    class_names: list[str],
) -> tuple[set[str], list[dict[str, str | bool]]]:
    """按目标层标签空间筛选可推理的测试文件夹。"""
    hierarchy = resolve_hierarchy_from_meta(meta)
    model_labels = set(class_names)
    selected_names = set()
    rows = []
    for folder_name in folder_names:
        species_prefix = parse_folder_prefix(folder_name, uppercase_enable=True)
        try:
            expected_label = hierarchy.expected_label(folder_name, level_name)
        except ValueError:
            expected_label = None
        expected_in_model = expected_label in model_labels
        if expected_label is None:
            reason = "unmapped_species_prefix"
        elif not expected_in_model:
            reason = "outside_model_label_space"
        else:
            reason = "selected"
            selected_names.add(folder_name)
        rows.append(
            {
                "folder": folder_name,
                "species_prefix": species_prefix,
                "target_level": level_name,
                "expected_label": expected_label or "",
                "expected_in_model": expected_in_model,
                "selected": folder_name in selected_names,
                "reason": reason,
            }
        )
    return selected_names, rows


def build_test_folder_labels(
    folder_names: list[str],
    meta: Mapping[str, Any],
    level_name: str,
    class_names: list[str],
) -> tuple[dict[str, int], list[dict[str, str | bool]]]:
    """把扁平测试文件夹名映射到目标层全局类别索引，无法归属时记为 ``-1``。

    独立测试集目录不带属名层，标签只能从文件夹前缀反查。返回的索引与
    ``class_names`` 的位置一一对应，可直接作为评估真值。
    """
    position = {str(name): index for index, name in enumerate(class_names)}
    _, rows = build_test_input_selection(folder_names, meta, level_name, class_names)
    labels = {
        str(row["folder"]): position.get(str(row["expected_label"]), -1)
        for row in rows
    }
    return labels, rows


def build_folder_summary(
    folder_name: str,
    expected_label: str | None,
    class_names: list[str],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总文件夹投票，直接取票数最多的标签作为文件夹输出。"""
    counter = Counter(item["top1_label"] for item in predictions)
    majority_label, majority_count = counter.most_common(1)[0] if counter else ("unknown", 0)
    total_count = len(predictions)
    expected_in_model = expected_label in set(class_names)
    correct_count = (
        sum(item["top1_label"] == expected_label for item in predictions)
        if expected_in_model
        else 0
    )
    return {
        "folder": folder_name,
        "expected_label": expected_label or "unknown",
        "expected_in_model": bool(expected_in_model),
        "predicted_label": majority_label,
        "majority_count": int(majority_count),
        "total_count": total_count,
        "correct_count": int(correct_count),
        "correct_ratio": correct_count / total_count if total_count else 0.0,
        "folder_correct": bool(expected_in_model and majority_label == expected_label),
    }
