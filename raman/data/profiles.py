from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    """描述一个数据集的名称和阶段目录"""
    profile_id: str
    dataset_name: str
    root_init: str = "init"
    root_init_pack: str = "init.npz"
    root_process_raw: str = "train_raw"
    root_train_clean: str = "train"
    root_test_clean: str = "test"
    root_train_fig: str = "fig_train"
    root_test_fig: str = "fig_test"
    root_init_test: str = "init_test"
    root_init_fig: str = "fig_init"
    pca_log_name: str = "pca_log.txt"
    cosmic_ray_log_name: str = "cosmic_ray_removal_log.txt"


PROFILES = {
    "MICRO": DatasetProfile(
        profile_id="MICRO",
        dataset_name="微生物",
    ),
    "ENT": DatasetProfile(
        profile_id="ENT",
        dataset_name="肠杆菌",
    ),
    "NENT": DatasetProfile(
        profile_id="NENT",
        dataset_name="非肠杆菌",
    ),
    "GP": DatasetProfile(
        profile_id="GP",
        dataset_name="阳性菌",
    ),
    "FUNG": DatasetProfile(
        profile_id="FUNG",
        dataset_name="真菌"
    ),
    "MN_IgA": DatasetProfile(
        profile_id="MN_IgA",
        dataset_name="MN_IgA",
    ),
    "resistance": DatasetProfile(
        profile_id="resistance",
        dataset_name="耐药菌",
    ),
    "anaerobe": DatasetProfile(
        profile_id="anaerobe",
        dataset_name="厌氧菌",
    ),
    "test": DatasetProfile(
        profile_id="test",
        dataset_name="测试菌",
    )
}

PROFILE_LOOKUP = {}
for profile in PROFILES.values():
    PROFILE_LOOKUP[profile.profile_id] = profile
    PROFILE_LOOKUP[profile.dataset_name] = profile


def list_profiles():
    return list(PROFILES.values())


def get_profile(profile_key):
    if profile_key not in PROFILE_LOOKUP:
        raise KeyError(f"Unknown dataset profile: {profile_key}")
    return PROFILE_LOOKUP[profile_key]


def get_dataset_dir(profile, project_root=None):
    base = Path(project_root) if project_root is not None else Path.cwd()
    return (base / "dataset" / profile.dataset_name).resolve()
