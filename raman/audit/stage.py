"""审核阶段判定规则"""

from __future__ import annotations

import numpy as np

from raman.audit.config import AuditConfig
from raman.audit.scoring import STAGE_DELETE_CATEGORY


def classify_invalid(records, audit_cfg: AuditConfig):
    """第一阶段：只判断光谱本身是否无效，不做类内离群删除"""
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


def _finite_le(value, threshold):
    """判断有限值是否小于等于阈值"""
    return np.isfinite(value) and value <= threshold


def _finite_ge(value, threshold):
    """判断有限值是否大于等于阈值"""
    return np.isfinite(value) and value >= threshold


def _candidate_evidence(record, audit_cfg: AuditConfig):
    """汇总单条谱的类内相似性候选证据"""
    reasons = []
    strong_count = 0
    review_count = 0

    if _finite_le(record.corr_ref, audit_cfg.class_corr_ref_remove_max):
        reasons.append("class_low_corr_ref")
        strong_count += 1
    elif _finite_le(record.corr_ref, audit_cfg.class_corr_ref_review_max):
        reasons.append("class_low_corr_ref_review")
        review_count += 1

    if _finite_le(record.nearest_ref_corr, audit_cfg.class_nearest_ref_remove_max):
        reasons.append("class_low_nearest_ref_corr")
        strong_count += 1
    elif _finite_le(record.nearest_ref_corr, audit_cfg.class_nearest_ref_review_max):
        reasons.append("class_low_nearest_ref_corr_review")
        review_count += 1

    if _finite_ge(record.rmse_to_ref, audit_cfg.class_rmse_ref_remove_min):
        reasons.append("class_high_rmse_to_ref")
        strong_count += 1
    elif _finite_ge(record.rmse_to_ref, audit_cfg.class_rmse_ref_review_min):
        reasons.append("class_high_rmse_to_ref_review")
        review_count += 1

    local_strong = (
        record.local_pos_width_points >= audit_cfg.class_local_width_min_points
        and record.local_pos_max_z >= audit_cfg.class_local_remove_z_min
        and record.local_pos_area >= audit_cfg.class_local_remove_area_min
    )
    local_review = (
        record.local_pos_width_points >= audit_cfg.class_local_width_min_points
        and record.local_pos_max_z >= audit_cfg.class_local_review_z_min
        and record.local_pos_area >= audit_cfg.class_local_review_area_min
    )
    if local_strong:
        reasons.append("class_local_positive_outlier")
    elif local_review:
        reasons.append("class_local_positive_review")
        review_count += 1

    overall_strong = strong_count >= 2
    overall_review = strong_count >= 1 or review_count >= 1
    is_candidate = overall_strong or overall_review or local_strong or local_review
    wants_remove = overall_strong or local_strong
    return is_candidate, wants_remove, reasons


def _risk_score(record, audit_cfg: AuditConfig):
    """计算类内相似性风险排序分"""
    score = 0.0
    if np.isfinite(record.corr_ref):
        score += max(audit_cfg.class_corr_ref_review_max - record.corr_ref, 0.0) * 20.0
    if np.isfinite(record.nearest_ref_corr):
        score += max(audit_cfg.class_nearest_ref_review_max - record.nearest_ref_corr, 0.0) * 16.0
    if np.isfinite(record.rmse_to_ref):
        score += max(record.rmse_to_ref - audit_cfg.class_rmse_ref_review_min, 0.0) * 8.0
    score += max(record.local_pos_max_z - audit_cfg.class_local_review_z_min, 0.0)
    score += max(record.local_pos_area - audit_cfg.class_local_review_area_min, 0.0) / 30.0
    score += max(record.local_pos_width_points - audit_cfg.class_local_width_min_points, 0) / 8.0
    return score


def classify_class_similarity(records, audit_cfg: AuditConfig):
    """第二阶段：用同属同前缀参考池判断类内相似性"""
    preliminary: dict[int, tuple[bool, bool, tuple[str, ...]]] = {}
    group_totals = {}
    group_candidate_counts = {}

    for idx, record in enumerate(records):
        record.stage = "class-similarity"
        record.delete_category = ""
        if record.z is None:
            record.decision = "skip"
            record.reasons = (record.skip_reason or "preprocess_failed",)
            continue

        group_totals[record.group] = group_totals.get(record.group, 0) + 1
        if record.ref_pool_size < audit_cfg.class_min_ref_samples:
            preliminary[idx] = (False, False, ())
            continue

        is_candidate, wants_remove, reasons = _candidate_evidence(record, audit_cfg)
        preliminary[idx] = (is_candidate, wants_remove, tuple(reasons))
        if is_candidate:
            group_candidate_counts[record.group] = group_candidate_counts.get(record.group, 0) + 1

    for idx, record in enumerate(records):
        if record.z is None:
            continue

        group_count = group_candidate_counts.get(record.group, 0)
        group_total = max(group_totals.get(record.group, 0), 1)
        record.folder_candidate_count = int(group_count)
        record.folder_candidate_fraction = float(group_count / group_total)

        is_candidate, wants_remove, reasons = preliminary.get(idx, (False, False, ()))
        if record.ref_pool_size < audit_cfg.class_min_ref_samples:
            record.decision = "keep"
            record.reasons = ()
            record.risk_score = 0.0
            continue
        if not is_candidate:
            record.decision = "keep"
            record.reasons = ()
            record.risk_score = _risk_score(record, audit_cfg)
            continue

        high_concentration = (
            group_count > audit_cfg.class_folder_candidate_max_count
            or record.folder_candidate_fraction > audit_cfg.class_folder_candidate_max_fraction
        )
        no_other_folder_ref = record.other_ref_pool_size <= 0
        final_reasons = list(reasons)
        if high_concentration:
            final_reasons.append("class_folder_concentrated")
        if no_other_folder_ref:
            final_reasons.append("class_no_other_folder_reference")

        if wants_remove and not high_concentration and not no_other_folder_ref:
            record.decision = "remove_candidate"
            record.delete_category = STAGE_DELETE_CATEGORY["class-similarity"]
        else:
            record.decision = "review_candidate"
            record.delete_category = ""

        record.reasons = tuple(dict.fromkeys(final_reasons))
        record.risk_score = _risk_score(record, audit_cfg)
        if record.decision == "remove_candidate":
            record.risk_score += 5.0
