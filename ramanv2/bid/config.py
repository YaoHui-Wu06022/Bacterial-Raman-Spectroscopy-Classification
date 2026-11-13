"""BID 两阶段训练配置。"""

from __future__ import annotations

from typing import Any, Mapping

from ramanv2.bid.paths import BID_DATASET_ID
from ramanv2.common.hierarchy import FLAT_LABEL_SCHEMA
from ramanv2.core.config import Config, build_config


BID_INPUT_VALUES = {
    "cut_min": 381.98,
    "cut_max": 1792.4,
    "target_points": 1000,
    "bad_bands": (),
    "norm_method": "snv",
    "smooth_use": True,
    "d1_use": False,
    "win_smooth": 11,
}
BID_STAGE_TRAINING_VALUES = {
    "ref": {
        "train_split": 0.9,
        "patience": 40,
    },
    "finetune": {
        "train_split": 0.9,
        "patience": 40,
        "learning_rate": 1e-4,
        "epochs": 30,
        "use_supcon_loss": True,
    },
}


def build_bid_config(
    stage: str,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """构建 BID 指定阶段配置；输入为 1000 点原谱加 SG 平滑通道。"""
    if stage not in BID_STAGE_TRAINING_VALUES:
        raise ValueError(f"未知 BID 阶段：{stage}；可选值：{list(BID_STAGE_TRAINING_VALUES)}")
    values = {
        "dataset_id": BID_DATASET_ID,
        "profile_id": BID_DATASET_ID,
        "label_schema": FLAT_LABEL_SCHEMA,
        **BID_INPUT_VALUES,
        **BID_STAGE_TRAINING_VALUES[stage],
    }
    values.update(overrides or {})
    return build_config(values)
