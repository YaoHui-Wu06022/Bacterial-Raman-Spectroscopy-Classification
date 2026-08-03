"""分类任务的通用指标计算。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: range | list[int],
) -> dict[str, float]:
    """计算 accuracy、macro-F1 和 macro recall。"""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    label_list = list(labels)
    if targets.size == 0 or predictions.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0}
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                labels=label_list,
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                average="macro",
                labels=label_list,
                zero_division=0,
            )
        ),
    }
