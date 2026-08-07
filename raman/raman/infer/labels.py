from collections import Counter

from raman.tool.hierarchy import label_from_parts
from raman.tool.naming import normalize_folder_prefix


def build_expected_lookup_from_meta(meta, level_name):
    """从 hierarchy_meta 的类别路径建立测试前缀到目标层标签的映射"""
    prefix_to_labels: dict[str, set[str]] = {}
    for labels in (meta.get("class_names_by_level") or {}).values():
        for label in labels:
            parts = [part for part in str(label).replace("\\", "/").split("/") if part]
            if not parts:
                continue
            expected_label = label_from_parts(parts, level_name)
            prefix = normalize_folder_prefix(parts[-1])
            if expected_label and prefix:
                prefix_to_labels.setdefault(prefix, set()).add(expected_label)

    expected_lookup = {}
    ambiguous = {}
    for prefix, labels in sorted(prefix_to_labels.items()):
        if len(labels) == 1:
            expected_lookup[prefix] = next(iter(labels))
        else:
            ambiguous[prefix] = sorted(labels)

    if ambiguous:
        preview = ", ".join(
            f"{prefix}={labels}" for prefix, labels in list(ambiguous.items())[:8]
        )
        suffix = "" if len(ambiguous) <= 8 else f", ... (+{len(ambiguous) - 8})"
        print(f"[Warn] ambiguous expected-label prefixes ignored: {preview}{suffix}")
    return expected_lookup


def folder_summary(folder_name, expected_label, class_names, predictions):
    """汇总单个测试文件夹的多数投票结果"""
    total = len(predictions)
    counter = Counter(item["top1_label"] for item in predictions)
    predicted, majority_count = counter.most_common(1)[0] if counter else ("unknown", 0)
    expected_in_model = expected_label in set(class_names)
    correct_count = (
        sum(1 for item in predictions if item["top1_label"] == expected_label)
        if expected_in_model
        else 0
    )
    correct_ratio = correct_count / total if total else 0.0
    return {
        "folder": folder_name,
        "expected_label": expected_label or "unknown",
        "expected_in_model": bool(expected_in_model),
        "predicted_label": predicted,
        "majority_count": int(majority_count),
        "total_count": int(total),
        "correct_count": int(correct_count),
        "correct_ratio": float(correct_ratio),
        "folder_correct": bool(expected_in_model and predicted == expected_label),
    }


def write_summary(path, rows, evaluate=True):
    """写出独立测试文件夹级汇总"""
    total = len(rows)
    correct = sum(1 for row in rows if row["folder_correct"]) if evaluate else 0
    lines = [
        "===== TEST SUMMARY =====",
        "",
        f"Folders        : {total}",
        (
            f"Folder correct : {correct}/{total} ({(correct / total * 100) if total else 0.0:.2f}%)"
            if evaluate
            else "Evaluation     : disabled"
        ),
        "",
        "\t".join(
            [
                "folder",
                "expected_label",
                "expected_in_model",
                "predicted_label",
                "majority_count",
                "total_count",
                "correct_count",
                "correct_ratio",
                "folder_correct",
            ]
            if evaluate
            else ["folder", "predicted_label", "majority_count", "total_count"]
        ),
    ]
    for row in rows:
        if evaluate:
            values = [
                row["folder"],
                row["expected_label"],
                str(row["expected_in_model"]),
                row["predicted_label"],
                str(row["majority_count"]),
                str(row["total_count"]),
                str(row["correct_count"]),
                f"{row['correct_ratio']:.6f}",
                str(row["folder_correct"]),
            ]
        else:
            values = [
                row["folder"],
                row["predicted_label"],
                str(row["majority_count"]),
                str(row["total_count"]),
            ]
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
