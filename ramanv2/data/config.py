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

    def build_cosmic_ray_options(self, profile_id: str) -> dict:
        """按统一配置构建单谱宇宙射线处理参数。"""
        return {
            "cosmic_ray_enable": profile_id in set(self.cosmic_ray_profile_ids),
            "cosmic_ray_window_points": int(self.cosmic_ray_window_points),
            "cosmic_ray_threshold": float(self.cosmic_ray_threshold),
            "cosmic_ray_max_iter": int(self.cosmic_ray_max_iter),
        }

DEFAULT_BUILD_CONFIG = DataBuildConfig()


def resolve_build_config(config: DataBuildConfig | None = None) -> DataBuildConfig:
    """返回调用方配置，缺失时使用默认常规数据构建配置。"""
    return DEFAULT_BUILD_CONFIG if config is None else config


def build_cosmic_ray_options(profile_id: str, config: DataBuildConfig) -> dict:
    """按统一配置构建单谱宇宙射线处理参数。"""
    return config.build_cosmic_ray_options(profile_id)
