"""BID 评估入口：沿用通用评估并输出论文风格混淆矩阵。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ramanv2.bid.confusion_style import BID_CONFUSION_STYLE, BidConfusionStyle, write_bid_confusion_matrix
from ramanv2.evaluation.independent_test import evaluate_test_set
from ramanv2.evaluation.model_eval import evaluate_model_run


def evaluate_bid_model_run(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
    train_dir: Path | str | None = None,
    style: BidConfusionStyle = BID_CONFUSION_STYLE,
) -> Path:
    """评估 BID 验证集并重绘混淆矩阵。"""
    result_dir = evaluate_model_run(
        source_dir,
        level_name,
        device,
        train_dir=train_dir,
    )
    write_bid_confusion_matrix(result_dir, style)
    return result_dir


def evaluate_bid_test_set(
    source_dir: Path | str,
    device: str | None = None,
    test_dir: Path | str | None = None,
    sample_limit: int | None = None,
    train_dir: Path | str | None = None,
    style: BidConfusionStyle = BID_CONFUSION_STYLE,
) -> dict[str, Any]:
    """评估 BID 独立测试集并重绘混淆矩阵。"""
    report = evaluate_test_set(
        source_dir,
        device,
        test_dir,
        sample_limit,
        train_dir=train_dir,
    )
    write_bid_confusion_matrix(report["result_dir"], style)
    return report
