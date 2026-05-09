from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    """描述一个数据集的名称、别名和阶段目录"""
    profile_id: str
    dataset_name: str
    count_root: str
    root_init: str = "init"
    root_init_pack: str = "init.npz"
    root_process_raw: str = "train_raw"
    root_train_clean: str = "train"
    root_test_clean: str = "test"
    root_train_fig: str = "fig_train"
    root_test_fig: str = "fig_test"
    root_test_raw: str = "test_raw"
    root_init_fig: str = "fig_init"
    log_name: str = "log.txt"
    aliases: tuple[str, ...] = ()


PROFILES = {
    "bacteria": DatasetProfile(
        profile_id="bacteria",
        dataset_name="细菌",
        count_root="train",
        aliases=("细菌", "bacteria"),
    ),
    "Enterobacteriaceae": DatasetProfile(
        profile_id="Enterobacteriaceae",
        dataset_name="肠杆菌",
        count_root="train",
        aliases=("肠杆菌", "Enterobacteriaceae"),
    ),
    "resistance": DatasetProfile(
        profile_id="resistance",
        dataset_name="耐药菌",
        count_root="train",
        aliases=("耐药菌", "resistance"),
    ),
    "anaerobe": DatasetProfile(
        profile_id="anaerobe",
        dataset_name="厌氧菌",
        count_root="train",
        aliases=("厌氧菌", "anaerobe"),
    ),
    "delete": DatasetProfile(
        profile_id="delete",
        dataset_name="移除数据",
        count_root="train",
        aliases=("delete",),
    )
}

PROFILE_LOOKUP = {}
for profile in PROFILES.values():
    PROFILE_LOOKUP[profile.profile_id] = profile
    PROFILE_LOOKUP[profile.dataset_name] = profile
    for alias in profile.aliases:
        PROFILE_LOOKUP[alias] = profile


def list_profiles():
    return list(PROFILES.values())


def get_profile(profile_key):
    if profile_key not in PROFILE_LOOKUP:
        raise KeyError(f"Unknown dataset profile: {profile_key}")
    return PROFILE_LOOKUP[profile_key]


def get_dataset_dir(profile, project_root=None):
    base = Path(project_root) if project_root is not None else Path.cwd()
    return (base / "dataset" / profile.dataset_name).resolve()
