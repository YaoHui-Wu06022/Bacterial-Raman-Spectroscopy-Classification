"""分类任务共用的指标计算。"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_classification_metrics(y_true, y_pred, labels):
    """统一计算 accuracy、macro_f1 和 macro_recall。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_recall": 0.0,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
    }
