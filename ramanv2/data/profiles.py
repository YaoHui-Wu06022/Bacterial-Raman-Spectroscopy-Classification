"""常规数据集 profile 映射。

Stanford 预训练和微调不属于本映射，后续由扩展包独立配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.core.paths import DATASET_ROOT


@dataclass(frozen=True)
class DatasetProfile:
    """描述常规数据集名称及其各阶段目录名称。"""

    profile_id: str
    dataset_name: str
    root_init: str = "init"
    root_init_test: str = "init_test"
    root_init_pack: str = "init.npz"
    root_train_clean: str = "train"
    root_test: str = "test"
    root_train_fig: str = "fig_train"
    pca_log_name: str = "pca_log.txt"
    cosmic_ray_log_name: str = "cosmic_ray_removal_log.txt"
    cosmic_ray_overrides: dict | None = None


PROFILES = {
    "MICRO": DatasetProfile("MICRO", "MICRO"),
    "GN": DatasetProfile("GN", "GN"),
    "GP": DatasetProfile("GP", "GP"),
    "FUNG": DatasetProfile("FUNG", "FUNG"),
    "resistance": DatasetProfile("resistance", "耐药菌"),
    "anaerobe": DatasetProfile("anaerobe", "厌氧菌"),
    "test": DatasetProfile("test", "测试菌"),
    "alldata": DatasetProfile("alldata", "alldata"),
}

PROFILE_LOOKUP = {
    key: profile
    for profile in PROFILES.values()
    for key in (profile.profile_id, profile.dataset_name)
}


def list_profiles() -> list[DatasetProfile]:
    """返回全部常规数据集 profile。"""
    return list(PROFILES.values())


def get_profile(profile_key: str) -> DatasetProfile:
    """按稳定 profile 名或显示数据集名解析常规 profile。"""
    try:
        return PROFILE_LOOKUP[profile_key]
    except KeyError as exc:
        raise KeyError(f"Unknown regular dataset profile: {profile_key}") from exc


def get_dataset_dir(profile: DatasetProfile, project_root: Path | str | None = None) -> Path:
    """返回 profile 在项目数据集根目录下的目录。"""
    root = DATASET_ROOT if project_root is None else Path(project_root) / "dataset"
    return (root / profile.dataset_name).resolve()


def resolve_training_dir(profile_key: str) -> Path:
    """优先解析构建后的 train 目录，缺失时使用可直接训练的 init 目录。"""
    profile = get_profile(profile_key)
    dataset_dir = get_dataset_dir(profile)
    train_dir = dataset_dir / profile.root_train_clean
    if train_dir.is_dir():
        return train_dir
    init_dir = dataset_dir / profile.root_init
    return init_dir if init_dir.is_dir() else dataset_dir
