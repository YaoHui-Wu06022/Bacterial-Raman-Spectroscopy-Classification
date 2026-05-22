"""审核阈值统一配置

这些参数集中放在这里，避免单谱审核、参考组审核和全库扫描各自维护一套阈值
CLI 只负责输入输出和报告数量，不再暴露重复评分参数
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditConfig:
    """所有审计入口共用的参数配置"""

    # 前缀池离群评分：将同属同前缀的小文件夹先合并，再与前缀均值谱比较
    min_group_samples: int = 5
    group_score_threshold: float = 3.5
    group_corr_threshold: float = 0.92
    group_point_z_threshold: float = 8.0
    group_bad_ratio_threshold: float = 0.03
    prefix_strong_score_threshold: float = 4.5
    prefix_strong_corr_threshold: float = 0.88
    prefix_strong_bad_ratio_threshold: float = 0.04
    prefix_extreme_bad_ratio_threshold: float = 0.08
    prefix_variance_remove_score_threshold: float = 8.0
    prefix_variance_remove_corr_threshold: float = 0.82
    prefix_other_folder_corr_threshold: float = 0.88
    prefix_other_folder_invalid_corr_threshold: float = 0.82

    # 粗糙噪声评分：在同一文件夹内，基于一阶差分的鲁棒尺度进行判断
    roughness_z_threshold: float = 3.5
    roughness_min: float = 0.12

    # 参考组评分：当文件数量足够时，优先使用同属同前缀的文件夹作为参考
    min_ref_files: int = 20
    min_ref_samples: int = 5
    ref_corr_floor: float = 0.80
    ref_corr_margin: float = 0.05
    nearest_ref_corr_floor: float = 0.86
    nearest_ref_corr_margin: float = 0.03
    ref_bad_ratio_floor: float = 0.04
    ref_bad_ratio_margin: float = 0.02
    ref_rmse_floor: float = 0.75
    ref_rmse_multiplier: float = 1.35
    ref_cosmic_floor: int = 80
    ref_cosmic_margin: int = 30
    folder_corr_ref_warning: float = 0.75

    # 阶梯状光谱检测：检测预处理后仍存在的长平台或突跳
    step_smooth_points: int = 21
    step_side_points: int = 28
    step_gap_points: int = 4
    step_jump_z_threshold: float = 8.0
    step_level_z_threshold: float = 12.0
    step_min_delta: float = 0.9
    step_opposite_window: int = 32
    step_edge_cm: float = 12.0

    # 残差宇宙射线样异常复核：检测清理后残差中正向、短到中等宽度的异常峰
    residual_pos_z_threshold: float = 8.0
    residual_min_max_z: float = 12.0
    residual_review_max_pos_z: float = 16.0
    residual_max_width_cm: float = 30.0

    # 前缀均值谱预筛：先筛明显额外凸起、阶梯样异常和完全不贴合的重噪声无效谱
    local_bump_z_threshold: float = 3.5
    local_bump_remove_z: float = 5.0
    local_bump_remove_area: float = 60.0
    local_bump_max_width_cm: float = 90.0
    species_bump_min_max_z: float = 12.0
    species_invalid_corr_threshold: float = 0.75
    species_invalid_nearest_corr_threshold: float = 0.82
    species_invalid_bad_ratio_threshold: float = 0.06

    # 文件夹级汇总规则：这些规则只用于触发文件夹层面的复核提示
    folder_candidate_fraction_review: float = 0.20
    folder_ref_outlier_fraction_review: float = 0.15
    folder_step_fraction_review: float = 0.15
    folder_candidate_fraction_remove: float = 0.45
    folder_ref_outlier_fraction_remove: float = 0.30
    folder_step_fraction_remove: float = 0.25

    # 手动删除记录中的原因标签
    delete_reason_labels: tuple[str, ...] = (
        "残留宇宙射线",
        "阶梯谱",
        "无效噪声谱",
        "前缀离群",
        "粗糙噪声",
        "参考组离群",
        "组内离群",
    )


DEFAULT_AUDIT_CONFIG = AuditConfig()
