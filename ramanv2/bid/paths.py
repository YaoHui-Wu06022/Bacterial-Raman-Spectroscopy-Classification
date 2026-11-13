"""BID 目录解析与独立数据集选择。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.common.paths import PROJECT_ROOT
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
