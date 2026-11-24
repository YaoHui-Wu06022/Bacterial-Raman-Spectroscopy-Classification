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
    baseline_fit_min: float = 200.0
    baseline_fit_max: float = 2000.0
    min_samples_per_class: int = 8

DEFAULT_BUILD_CONFIG = DataBuildConfig()


def resolve_build_config(config: DataBuildConfig | None = None) -> DataBuildConfig:
    """返回调用方配置，缺失时使用默认常规数据构建配置。"""
    return DEFAULT_BUILD_CONFIG if config is None else config
