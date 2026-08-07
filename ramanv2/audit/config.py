"""审核流程的阈值与光谱范围配置。"""

from __future__ import annotations

from dataclasses import dataclass

from ramanv2.core.config import InputConfig
from ramanv2.data.config import DEFAULT_BUILD_CONFIG, DataBuildConfig
from ramanv2.audit.raw_quality import RawQualityConfig
from ramanv2.audit.similarity import NeighborConfig


@dataclass(frozen=True)
class AuditConfig:
    """定义审核阶段共用的原始质量、近邻与平移参数。"""

    input: InputConfig = InputConfig()
    cleaning: DataBuildConfig = DEFAULT_BUILD_CONFIG
    raw_quality: RawQualityConfig = RawQualityConfig()
    neighbor: NeighborConfig = NeighborConfig()


DEFAULT_AUDIT_CONFIG = AuditConfig()


def resolve_audit_config(config: AuditConfig | None = None) -> AuditConfig:
    """返回调用方配置；未提供时使用固定审核配置。"""
    return DEFAULT_AUDIT_CONFIG if config is None else config
