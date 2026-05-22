"""audit 分阶段清洗参数。"""

from __future__ import annotations

from dataclasses import dataclass

from raman.data.build import COSMIC_RAY_PEAK_WIDTH_MAX_POINTS


@dataclass(frozen=True)
class AuditConfig:
    """所有 audit 阶段共用的阈值"""

    min_prefix_samples: int = 5

    # 第一阶段：无效谱
    invalid_raw_coverage_min: float = 0.9  # 原始波数覆盖比例下限，低于此值认为有长段缺失
    invalid_flat_window_points: int = 40  # 平坦段检测的滑动窗口点数
    invalid_flat_range_max: float = 0.08  # 窗口内 SNV 极差低于此值时，认为该窗口近似无信息
    invalid_long_flat_points: int = 100  # 连续平坦点数达到此值，直接进入删除候选
    invalid_review_flat_points: int = 80  # 连续平坦点数达到此值但未到删除线，进入复核候选
    invalid_flat_fraction: float = 0.30  # 全谱贴近中位数的比例过高，认为整体有效峰结构弱
    invalid_flat_near_median: float = 0.03  # 判断“贴近中位数”的 SNV 距离阈值
    invalid_noise_roughness_min: float = 0.80  # 谱自身一阶差分粗糙度下限，越高说明高频噪声越强
    invalid_noise_smooth_points: int = 31  # 平滑窗口点数，用于分离有效慢变结构和高频细节
    invalid_noise_structure_ratio_max: float = 1.50  # 平滑结构幅度 / 粗糙度过低，说明强噪声里缺少有效峰结构
    invalid_noise_review_roughness_min: float = 0.6  # 强噪声复核线，比删除线稍宽松
    invalid_noise_review_structure_ratio_max: float = 2.00  # 有效结构偏弱复核线，比删除线稍宽松

    # 第二阶段：宇宙射线清理后仍残留的宽上升平台 / 阶梯异常
    anomalous_wide_min_points: int = COSMIC_RAY_PEAK_WIDTH_MAX_POINTS # 宽异常至少要超过 peak 可修复宽度
    anomalous_wide_smooth_points: int = 5  # 检测前的短窗口平滑点数
    anomalous_wide_floor_window_points: int = 71  # 局部下包络窗口点数，用于估计正常底部
    anomalous_wide_z_min: float = 3.0  # 宽异常连续区域的 z 下限
    anomalous_wide_area_z_min: float = 50.0  # 删除候选的宽异常面积下限
    anomalous_wide_max_z_min: float = 5.0  # 删除候选的宽异常峰值 z 下限
    anomalous_wide_review_edge_z_min: float = 3.75  # 宽异常进入复核候选的边缘突跳 z 下限
    anomalous_wide_delete_edge_z_min: float = 5.0  # 删除候选的边缘突跳 z 下限

    # 第三阶段：同属同前缀类内相似性和局部正残差异常
    class_min_ref_samples: int = 8  # 参考池最少谱数，低于此值不自动判定
    class_corr_ref_review_max: float = 0.86  # 与同前缀中位谱相关性低于此值，进入复核证据
    class_corr_ref_remove_max: float = 0.78  # 与同前缀中位谱相关性低于此值，作为删除强证据
    class_nearest_ref_review_max: float = 0.90  # 与其它小文件夹最近邻相关性低于此值，进入复核证据
    class_nearest_ref_remove_max: float = 0.82  # 与其它小文件夹最近邻相关性低于此值，作为删除强证据
    class_rmse_ref_review_min: float = 0.55  # 与同前缀中位谱 RMSE 高于此值，进入复核证据
    class_rmse_ref_remove_min: float = 0.75  # 与同前缀中位谱 RMSE 高于此值，作为删除强证据
    class_local_z_min: float = 3.0  # 局部正残差连续区域的 z 下限
    class_local_width_min_points: int = COSMIC_RAY_PEAK_WIDTH_MAX_POINTS  # 局部异常最小连续点数
    class_local_review_z_min: float = 6.0  # 局部正残差峰值复核线
    class_local_review_area_min: float = 100.0  # 局部正残差面积复核线
    class_local_remove_z_min: float = 8.0  # 局部正残差峰值删除线
    class_local_remove_area_min: float = 150.0  # 局部正残差面积删除线
    class_folder_candidate_max_count: int = 8  # 同小文件夹候选数超过此值，只做复核
    class_folder_candidate_max_fraction: float = 0.30  # 同小文件夹候选比例超过此值，只做复核

    delete_categories: tuple[str, ...] = (
        "Invalid Spectrum",
        "Anomalous_Cosmic_Rays",
        "Class_Similarity_Outliers",
    )
    delete_reason_labels: tuple[str, ...] = (
        "Invalid Spectrum",
        "Anomalous_Cosmic_Rays",
        "Class_Similarity_Outliers",
    )


DEFAULT_AUDIT_CONFIG = AuditConfig()
