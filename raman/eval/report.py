import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def format_classification_report_text(
    report_dict,
    class_names,
    acc,
    macro_f1=None,
    macro_recall=None,
):
    """将 classification_report 的字典结果整理成统一文本"""

    def _fmt_line(name, p, r, f1, support):
        return (
            f"{name:<20}"
            f"{p * 100:>12.4f}%"
            f"{r * 100:>12.4f}%"
            f"{f1 * 100:>12.4f}%"
            f"{int(round(support)):>12d}"
        )

    lines = [
        f"{'':<20}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}",
        "",
    ]
    for cls_name in class_names:
        row = report_dict[cls_name]
        lines.append(
            _fmt_line(
                cls_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            )
        )

    total_support = int(sum(report_dict[name]["support"] for name in class_names))

    row = report_dict["macro avg"]
    if macro_f1 is None:
        macro_f1 = row["f1-score"]
    if macro_recall is None:
        macro_recall = row["recall"]
    lines.extend(
        [
            "",
            f"{'summary metric':<20}{'value':>12}{'support':>12}",
            f"{'Accuracy':<20}{acc * 100:>11.4f}%{total_support:>12d}",
            f"{'Macro F1-score':<20}{macro_f1 * 100:>11.4f}%{total_support:>12d}",
            f"{'Macro Recall':<20}{macro_recall * 100:>11.4f}%{total_support:>12d}",
        ]
    )

    return "\n".join(lines)


def build_confusion_annotation(cm):
    """计算归一化混淆矩阵和用于热图展示的标注文本"""
    denom = cm.sum(axis=1, keepdims=True).astype(np.float32)
    denom[denom == 0] = 1.0
    cm_norm = cm.astype(np.float32) / denom

    annot = np.empty_like(cm, dtype=object)
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            value = cm[row_idx, col_idx]
            annot[row_idx, col_idx] = (
                "0\n(0)"
                if value == 0
                else f"{cm_norm[row_idx, col_idx] * 100:.1f}%\n({value})"
            )

    return cm_norm, annot


def save_confusion_matrix_csv(cm, class_names, out_path):
    """保存原始混淆矩阵表格"""
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_path)


def save_confusion_matrix_figure(cm, class_names, out_path, show=False):
    """保存带百分比和计数标注的混淆矩阵热图"""
    cm_norm, annot = build_confusion_annotation(cm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=annot,
        fmt="",
        annot_kws={"size": 10},
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def write_text(out_path, content):
    """按 UTF-8 写文本文件"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
