"""BID 数据目录、两阶段配置与第二阶段权重初始化。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ramanv2.core.label_schema import FLAT_LABEL_SCHEMA
from ramanv2.common.paths import PROJECT_ROOT
from ramanv2.core.config import Config, build_config
from ramanv2.data.datasets import DatasetSpec
from ramanv2.data.hierarchy_spec import HierarchySpec
from ramanv2.data.profiles import DatasetProfile, DatasetSelection


BID_DATASET_ID = "BID"
BID_DATASET_DIR_NAME = "bacteriaID"
BID_TRAIN_REF_DIR_NAME = "train_ref"
BID_TRAIN_FINETUNE_DIR_NAME = "train_finetune"
BID_TEST_DIR_NAME = "test"


@dataclass(frozen=True)
class BidPaths:
    """BID 三份数据的固定目录。"""

    dataset_dir: Path
    train_ref_dir: Path
    train_finetune_dir: Path
    test_dir: Path


def resolve_bid_paths(project_root: Path | str | None = None) -> BidPaths:
    """解析 BID 数据目录；BID 不进入数据集注册表。"""
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    dataset_dir = (Path(root) / "dataset" / BID_DATASET_DIR_NAME).resolve()
    return BidPaths(
        dataset_dir=dataset_dir,
        train_ref_dir=dataset_dir / BID_TRAIN_REF_DIR_NAME,
        train_finetune_dir=dataset_dir / BID_TRAIN_FINETUNE_DIR_NAME,
        test_dir=dataset_dir / BID_TEST_DIR_NAME,
    )


def build_bid_selection(paths: BidPaths) -> DatasetSelection:
    """构造单层扁平选择，供 run_training 跳过注册表解析。"""
    hierarchy = HierarchySpec.flat()
    spec = DatasetSpec(
        dataset_id=BID_DATASET_ID,
        display_name="Bacteria-ID",
        base_dir=paths.dataset_dir.parent,
        directory=paths.dataset_dir,
        bad_bands=(),
        hierarchy=hierarchy,
    )
    profile = DatasetProfile(
        profile_id=BID_DATASET_ID,
        dataset_name=BID_DATASET_ID,
        dataset_id=BID_DATASET_ID,
        dataset_dir_name=paths.dataset_dir.name,
    )
    return DatasetSelection(
        spec=spec,
        profile=profile,
        dataset_dir=paths.dataset_dir,
        hierarchy=hierarchy,
    )


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
        "patience": 20,
    },
    "finetune": {
        "train_split": 0.9,
        "patience": 20,
        "learning_rate": 1e-4,
        "epochs": 50,
        # BID 为已预处理的干净数据，微调阶段关闭随机增强与同标签 mixup。
        "p_piecewise_gain": 0.0,
        "p_noise": 0.0,
        "p_axis": 0.0,
        "p_baseline_weak": 0.0,
        "p_baseline_strong": 0.0,
        "use_mixup": False,
        # BID 类平衡：focal 退化为普通 CE，关闭类别难度 EMA。
        "gamma": 0.0,
        "use_ema": False,
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


def build_checkpoint_initializer(
    model_path: Path | str,
) -> Callable[[nn.Module, Any], None]:
    """返回把 model.pt 权重严格载入新模型的训练回调。"""
    resolved = Path(model_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少第一阶段权重：{resolved}")

    def initialize_model(model: nn.Module, _train_task: Any) -> None:
        state = torch.load(resolved, map_location="cpu")
        model.load_state_dict(state, strict=True)

    return initialize_model


__all__ = [
    "BID_DATASET_DIR_NAME",
    "BID_DATASET_ID",
    "BID_INPUT_VALUES",
    "BID_STAGE_TRAINING_VALUES",
    "BidPaths",
    "build_bid_config",
    "build_bid_selection",
    "build_checkpoint_initializer",
    "resolve_bid_paths",
]