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


def shorten_class_name(name):
    """混淆矩阵坐标轴只显示最后一级名称，避免上级路径过长"""
    text = str(name)
    text = text.replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    return parts[-1] if parts else text


def shorten_class_names(class_names):
    """批量压缩类别显示名"""
    return [shorten_class_name(name) for name in class_names]


def save_confusion_matrix_csv(cm, class_names, out_path):
    """保存原始混淆矩阵表格"""
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_path)


def auto_confusion_matrix_figsize(class_names):
    """按类别数量等比例放大图片，让每个格子的尺寸尽量一致"""
    num_classes = max(len(class_names), 1)
    max_label_len = max((len(str(name)) for name in class_names), default=0)

    cell_size = 0.58
    label_pad = min(max_label_len, 24) * 0.06
    width = 2.8 + num_classes * cell_size + label_pad
    height = 2.6 + num_classes * cell_size
    return (
        min(max(width, 6.0), 36.0),
        min(max(height, 5.6), 36.0),
    )


def auto_confusion_matrix_font_sizes(num_classes):
    """按格子物理尺寸估算字号，类别变多时只轻微压缩"""
    if num_classes <= 12:
        return 10, 11
    if num_classes <= 24:
        return 8, 10
    if num_classes <= 36:
        return 7, 9
    return 6, 8


def save_confusion_matrix_figure(
    cm,
    class_names,
    out_path,
    show=False,
    shorten_labels=True,
    figsize=None,
):
    """保存带百分比和计数标注的混淆矩阵热图"""
    cm_norm, annot = build_confusion_annotation(cm)
    display_names = shorten_class_names(class_names) if shorten_labels else class_names
    if figsize is None:
        figsize = auto_confusion_matrix_figsize(display_names)
    annot_size, tick_size = auto_confusion_matrix_font_sizes(len(display_names))

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        annot=annot,
        fmt="",
        annot_kws={"size": annot_size},
        square=True,
        cbar_kws={"fraction": 0.035, "pad": 0.025, "aspect": 35},
    )
    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.tick_params(labelsize=tick_size)
    # 只给混淆矩阵主体四边加黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    plt.xticks(rotation=45, ha="right", fontsize=tick_size)
    plt.yticks(rotation=0, fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def write_text(out_path, content):
    """按 UTF-8 写文本文件"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
