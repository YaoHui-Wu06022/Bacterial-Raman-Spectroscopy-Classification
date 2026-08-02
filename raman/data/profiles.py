from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProfile:
    """描述一个数据集的名称和阶段目录"""
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
    train_build_mode: str = "standard"
    training_input_defaults: dict | None = None
    training_model_defaults: dict | None = None


PROFILES = {
    "MICRO": DatasetProfile(
        profile_id="MICRO",
        dataset_name="MICRO",
    ),
    "GN": DatasetProfile(
        profile_id="GN",
        dataset_name="GN",
    ),
    "GP": DatasetProfile(
        profile_id="GP",
        dataset_name="GP",
    ),
    "FUNG": DatasetProfile(
        profile_id="FUNG",
        dataset_name="FUNG",
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
    ),
    "original": DatasetProfile(
        profile_id="original",
        dataset_name="50种菌",
    ),
    "alldata": DatasetProfile(
        profile_id="alldata",
        dataset_name="alldata",
    ),
    "MN_IgA": DatasetProfile(
        profile_id="MN_IgA",
        dataset_name="MN_IgA",
    ),
    "Stanford": DatasetProfile(
        profile_id="Stanford",
        dataset_name="Stanforddataset",
        train_build_mode="grid_mask_only",
        training_input_defaults={
            "norm_method": "minmax",
            "p_piecewise_gain": 0.40,
            "p_noise": 0.70,
            "p_axis": 0.0,
            "p_baseline_weak": 0.55,
            "p_baseline_strong": 0.35,
            "p_shift": 0.0,
            "p_broadening": 0.0,
            "p_cut": 0.0,
            "max_pre_augs": 2,
            "max_post_augs": 0,
        },
        training_model_defaults={
            "batch_size": 256,
        },
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


def apply_training_profile_defaults(config):
    """Force profile-owned input and model defaults after user/global overrides."""
    profile = get_profile(config.dataset_name)
    for group, defaults in (
        (config.shared, profile.training_input_defaults or {}),
        (config.model, profile.training_model_defaults or {}),
    ):
        for name, value in defaults.items():
            if not hasattr(group, name):
                raise AttributeError(f"Unknown training setting in profile: {name}")
            setattr(group, name, deepcopy(value))
    return config
