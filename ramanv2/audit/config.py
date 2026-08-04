"""审核流程的阈值与光谱范围配置。"""

from __future__ import annotations

from dataclasses import dataclass

from ramanv2.core.config import InputConfig
from ramanv2.data.config import DEFAULT_BUILD_CONFIG, DataBuildConfig


@dataclass(frozen=True)
class AuditConfig:
    """定义审核阶段共用的原始质量、近邻与平移参数。"""

    input: InputConfig = InputConfig()
    cleaning: DataBuildConfig = DEFAULT_BUILD_CONFIG
    raw_min_points: int = 20
    raw_coverage_min: float = 0.98
    raw_flat_window: int = 40
    raw_long_flat_points: int = 100
    raw_saturation_points: int = 25
    similarity_min_references: int = 8
    shift_limit: float = 30.0
    shift_anchor_min: float = 980.0
    shift_anchor_max: float = 1020.0
    shift_target_cm: float = 1002.0
    shift_default_residual_min: float = 1000.0
    shift_default_residual_max: float = 1004.0
    shift_fungal_residual_min: float = 1001.0
    shift_fungal_residual_max: float = 1003.0
    shift_smooth_window: int = 5
    shift_total_limit: float = 10.0
    shift_large_move_folders: frozenset[str] = frozenset(
        {"BCC01", "ECL04", "EC03", "KAE03", "KAE04"}
    )
    shift_fixed_totals: tuple[tuple[str, str, float], ...] = (
        ("Burkholderia", "BCC01", -26.0),
        ("Enterobacter", "ECL04", -27.0),
        ("Escherichia", "EC03", -14.3),
        ("Klebsiella", "KAE03", -26.0),
        ("Klebsiella", "KAE04", -26.0),
        ("Proteus", "PVU03", 3.0),
    )
    fungal_genera: frozenset[str] = frozenset({"Candida"})
    shift_fit_min: float = 992.0
    shift_fit_max: float = 1012.0
    shift_fit_min_r2: float = 0.35
    shift_double_bic_gain: float = 10.0
    shift_double_min_separation: float = 2.5
    shift_double_max_separation: float = 9.0
    shift_double_min_amplitude_ratio: float = 0.20

    def get_fixed_shift(self, genus: str, folder: str) -> float | None:
        """返回指定属和文件夹的固定累计平移；未配置时返回空值。"""
        for fixed_genus, fixed_folder, delta in self.shift_fixed_totals:
            if (fixed_genus, fixed_folder) == (genus, folder):
                return delta
        return None


DEFAULT_AUDIT_CONFIG = AuditConfig()


def resolve_audit_config(config: AuditConfig | None = None) -> AuditConfig:
    """返回调用方配置；未提供时使用固定审核配置。"""
    return DEFAULT_AUDIT_CONFIG if config is None else config
