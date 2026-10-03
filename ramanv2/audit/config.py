"""文件夹级审核的光谱范围和近邻阈值配置。"""

from __future__ import annotations

from dataclasses import dataclass

from ramanv2.core.config import InputConfig
from ramanv2.data.config import DEFAULT_BUILD_CONFIG, DataBuildConfig
from ramanv2.audit.similarity import NeighborConfig


@dataclass(frozen=True)
class AuditConfig:
    """定义文件夹审核和总览图共用的输入、预处理与近邻参数。"""

    input: InputConfig = InputConfig()
    cleaning: DataBuildConfig = DEFAULT_BUILD_CONFIG
    neighbor: NeighborConfig = NeighborConfig()


DEFAULT_AUDIT_CONFIG = AuditConfig()


def resolve_audit_config(config: AuditConfig | None = None) -> AuditConfig:
    """返回调用方配置；未提供时使用固定审核配置。"""
    return DEFAULT_AUDIT_CONFIG if config is None else config
