"""常规数据集离线构建的配置模型。"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class DataBuildConfig:
    """描述常规数据集从 init 到 train/test 的固定清洗参数。"""

    baseline_method: str = "airPLS"
    baseline_lam: float = 1e5
    baseline_asls_p: float = 0.01
    baseline_max_iter: int = 15
    baseline_fit_min: float = 400.0
    baseline_fit_max: float = 2000.0
    cosmic_ray_profile_ids: tuple[str, ...] = ("original",)
    cosmic_ray_window_points: int = 7
    cosmic_ray_threshold: float = 7.0
    cosmic_ray_max_iter: int = 2
    min_samples_per_class: int = 8
    pca_enable: bool = False
    pca_components: float | int = 0.95
    pca_center_enable: bool = True
    pca_outlier_ratio: float = 0.01

DEFAULT_BUILD_CONFIG = DataBuildConfig()


def resolve_build_config(config: DataBuildConfig | None = None) -> DataBuildConfig:
    """返回调用方配置，缺失时使用默认常规数据构建配置。"""
    return DEFAULT_BUILD_CONFIG if config is None else config


def resolve_cosmic_options(profile, config: DataBuildConfig, label: str) -> dict:
    """合并 profile 路径覆盖，返回单谱宇宙射线处理参数。"""
    options = {
        "cosmic_ray_enable": profile.profile_id in set(config.cosmic_ray_profile_ids),
        "cosmic_ray_window_points": int(config.cosmic_ray_window_points),
        "cosmic_ray_threshold": float(config.cosmic_ray_threshold),
        "cosmic_ray_max_iter": int(config.cosmic_ray_max_iter),
    }
    normalized_label = str(label).replace("\\", "/").strip("/")
    key_map = {
        "enable": "cosmic_ray_enable",
        "window_points": "cosmic_ray_window_points",
        "threshold": "cosmic_ray_threshold",
        "max_iter": "cosmic_ray_max_iter",
    }
    matched = []
    for scope, values in (profile.cosmic_ray_overrides or {}).items():
        scope_key = str(scope).replace("\\", "/").strip("/")
        if scope_key in {"", "*"}:
            matched.append((0, values))
        elif normalized_label == scope_key or normalized_label.startswith(f"{scope_key}/"):
            matched.append((scope_key.count("/") + 1, values))
    for _depth, override in sorted(matched, key=lambda item: item[0]):
        for key, value in (override or {}).items():
            option_key = key_map.get(key, key)
            if option_key not in options:
                raise KeyError(f"Unknown cosmic ray override key: {key}")
            current = options[option_key]
            options[option_key] = bool(value) if isinstance(current, bool) else type(current)(value)
    return options
