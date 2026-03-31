from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    """描述一个数据集在离线预处理阶段需要的固定目录和坏波段配置。"""
    profile_id: str
    dataset_name: str
    train_bad_bands: tuple[tuple[float, float], ...]
    test_bad_bands: tuple[tuple[float, float], ...]
    count_root: str
    root_init: str = "dataset_init"
    root_init_pack: str = "dataset_init.npz"
    root_process_raw: str = "dataset_raw"
    root_train_clean: str = "dataset_train"
    root_test_clean: str = "dataset_test"
    root_train_fig: str = "dataset_train_fig"
    root_test_fig: str = "dataset_test_fig"
    root_test_raw: str = "测试菌"
    root_init_fig: str = "dataset_init_fig"
    log_name: str = "log.txt"
    aliases: tuple[str, ...] = ()


PROFILES = {
    "bacteria": DatasetProfile(
        profile_id="bacteria",
        dataset_name="细菌",
        train_bad_bands=((900.0, 950.0),),
        test_bad_bands=((900.0, 950.0),),
        count_root="dataset_train",
        aliases=("细菌", "bacteria"),
    ),
    "resistance": DatasetProfile(
        profile_id="resistance",
        dataset_name="耐药菌",
        train_bad_bands=((900.0, 940.0),),
        test_bad_bands=((900.0, 940.0),),
        count_root="dataset_train",
        aliases=("耐药菌", "resistance"),
    ),
    "anaerobe": DatasetProfile(
        profile_id="anaerobe",
        dataset_name="厌氧菌",
        train_bad_bands=((890.0,940.0),),
        test_bad_bands=((890.0,940.0),),
        count_root="dataset_train",
        aliases=("厌氧菌", "anaerobe"),
    ),
    "ding": DatasetProfile(
        profile_id="ding",
        dataset_name="丁",
        train_bad_bands=(),
        test_bad_bands=(),
        count_root="dataset_train",
        aliases=("丁", "ding"),
    ),
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
