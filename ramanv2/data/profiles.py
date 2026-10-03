"""常规数据集 profile 映射。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.core.paths import DATASET_ROOT


PIPELINE_PROFILE_IDS = ("MICRO", "GN", "GP", "FUNG")


@dataclass(frozen=True)
class DatasetProfile:
    """描述常规数据集名称及其各阶段目录名称。"""

    profile_id: str
    dataset_name: str
    root_init: str = "init"
    root_init_test: str = "init_test"
    root_train_clean: str = "train"
    root_test: str = "test"
    root_train_fig: str = "fig_train"


PROFILES = {
    "MICRO": DatasetProfile("MICRO", "MICRO"),
    "GN": DatasetProfile("GN", "GN"),
    "GP": DatasetProfile("GP", "GP"),
    "FUNG": DatasetProfile("FUNG", "FUNG"),
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


def resolve_training_dir(
    profile_key: str,
    project_root: Path | str | None = None,
    *,
    fallback_to_init_enable: bool = True,
) -> Path:
    """优先解析构建后的 train 目录，缺失时使用可直接训练的 init 目录。"""
    profile = get_profile(profile_key)
    dataset_dir = get_dataset_dir(profile, project_root)
    train_dir = dataset_dir / profile.root_train_clean
    if train_dir.is_dir():
        return train_dir
    if not fallback_to_init_enable:
        return train_dir
    init_dir = dataset_dir / profile.root_init
    return init_dir if init_dir.is_dir() else dataset_dir
