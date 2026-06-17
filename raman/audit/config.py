"""audit 分阶段清洗参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditConfig:
    """所有 audit 阶段共用的阈值"""

    min_prefix_samples: int = 5

    # 第一阶段：无效谱
    invalid_flat_window_points: int = 40  # 平坦段检测的滑动窗口点数
    invalid_flat_range_max: float = 0.08  # 窗口内标准化值极差低于此值时，认为该窗口近似无信息
    invalid_long_flat_points: int = 100  # 连续平坦点数达到此值，直接进入删除候选
    invalid_flat_fraction: float = 0.30  # 全谱贴近中位数的比例过高，认为整体有效峰结构弱
    invalid_flat_near_median: float = 0.03  # 判断“贴近中位数”的标准化距离阈值
    invalid_noise_detail_min: float = 0.42  # 高频细节波动下限，超过说明谱线锯齿波动过大

    # 第二阶段：同属同前缀类内相似性
    class_min_ref_samples: int = 8  # 参考池最少谱数，低于此值不自动判定
    class_corr_ref_remove_max: float = 0.78  # 与同前缀中位谱相关性低于此值，作为删除强证据
    class_rmse_ref_remove_min: float = 0.75  # 与同前缀中位谱 RMSE 高于此值，作为删除强证据
    class_local_residual_min: float = 0.25  # 局部正残差异常连续区域的残差下限
    class_local_residual_width_min_points: int = 10  # 局部残差异常最小连续点数
    class_local_residual_remove_area_min: float = 60.0  # 局部正残差超出阈值的面积达到此值，作为删除强证据

    delete_categories: tuple[str, ...] = (
        "Invalid Spectrum",
        "Similar Outliers",
    )


DEFAULT_AUDIT_CONFIG = AuditConfig()
