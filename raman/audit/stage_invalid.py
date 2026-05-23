"""第一阶段：无效谱清洗"""

from __future__ import annotations

import numpy as np

from raman.audit.config import AuditConfig
from raman.audit.scoring import STAGE_DELETE_CATEGORY


def classify_invalid(records, audit_cfg: AuditConfig):
    """只判断谱本身是否无效，不做类内离群删除"""
    for record in records:
        record.stage = "invalid"
        if record.z is None:
            record.decision = "skip"
            record.reasons = (record.skip_reason or "preprocess_failed",)
            continue

        reasons = []
        if np.isfinite(record.coverage_ratio) and record.coverage_ratio < audit_cfg.invalid_raw_coverage_min:
            reasons.append("invalid_missing_region")
        if record.long_flat_points >= audit_cfg.invalid_long_flat_points or record.flat_fraction >= audit_cfg.invalid_flat_fraction:
            reasons.append("invalid_flat_region")
        if (
            record.roughness >= audit_cfg.invalid_noise_roughness_min
            and np.isfinite(record.structure_ratio)
            and record.structure_ratio <= audit_cfg.invalid_noise_structure_ratio_max
        ):
            reasons.append("invalid_noise")

        review_reasons = []
        if not reasons and record.long_flat_points >= audit_cfg.invalid_review_flat_points:
            review_reasons.append("invalid_flat_region_review")
        if (
            not reasons
            and record.roughness >= audit_cfg.invalid_noise_review_roughness_min
            and np.isfinite(record.structure_ratio)
            and record.structure_ratio <= audit_cfg.invalid_noise_review_structure_ratio_max
        ):
            review_reasons.append("invalid_noise_review")

        if reasons:
            record.decision = "remove_candidate"
            record.delete_category = STAGE_DELETE_CATEGORY["invalid"]
            record.reasons = tuple(reasons)
        elif review_reasons:
            record.decision = "review_candidate"
            record.reasons = tuple(review_reasons)
        else:
            record.decision = "keep"
            record.reasons = ()

        record.risk_score = 0.0
        if np.isfinite(record.coverage_ratio):
            record.risk_score += max((audit_cfg.invalid_raw_coverage_min - record.coverage_ratio) * 20.0, 0.0)
        record.risk_score += record.long_flat_points / 45.0
        record.risk_score += record.flat_fraction * 8.0
        record.risk_score += max(record.roughness, 0.0) * 3.0
        if np.isfinite(record.structure_ratio):
            record.risk_score += max(audit_cfg.invalid_noise_review_structure_ratio_max - record.structure_ratio, 0.0) * 3.0
        record.risk_score += 5.0 if record.decision == "remove_candidate" else 0.0
