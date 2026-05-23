"""第二阶段：清理宇宙射线去除后仍残留的宽上升平台 / 阶梯异常"""

from __future__ import annotations

from raman.audit.config import AuditConfig
from raman.audit.scoring import STAGE_DELETE_CATEGORY


def classify_anomalous_cosmic(records, audit_cfg: AuditConfig):
    """只判断单谱自身是否存在宽于 peak 修复范围的残留异常"""
    for record in records:
        record.stage = "anomalous-cosmic"
        if record.z is None:
            record.decision = "skip"
            record.reasons = (record.skip_reason or "preprocess_failed",)
            continue

        has_wide_region = record.wide_bump_count > 0
        strong_wide_region = (
            has_wide_region
            and record.wide_bump_max_z >= audit_cfg.anomalous_wide_max_z_min
            and record.wide_bump_area >= audit_cfg.anomalous_wide_area_z_min
            and record.wide_edge_jump_z >= audit_cfg.anomalous_wide_delete_edge_z_min
        )

        if strong_wide_region:
            record.decision = "remove_candidate"
            record.delete_category = STAGE_DELETE_CATEGORY["anomalous-cosmic"]
            record.reasons = ("anomalous_cosmic_wide_edge_jump",)
        elif has_wide_region:
            record.decision = "review_candidate"
            record.reasons = ("anomalous_cosmic_wide_edge_review",)
        else:
            record.decision = "keep"
            record.reasons = ()

        record.risk_score = 0.0
        record.risk_score += max(record.wide_bump_max_z, 0.0)
        record.risk_score += max(record.wide_edge_jump_z, 0.0)
        record.risk_score += max(record.wide_bump_area, 0.0) / 20.0
        record.risk_score += max(record.wide_bump_width_points - audit_cfg.anomalous_wide_min_points, 0) / 4.0
        record.risk_score += 5.0 if record.decision == "remove_candidate" else 0.0
