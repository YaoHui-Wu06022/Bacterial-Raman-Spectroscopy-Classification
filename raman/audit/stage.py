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
        if record.long_flat_points >= audit_cfg.invalid_long_flat_points or record.flat_fraction >= audit_cfg.invalid_flat_fraction:
            reasons.append("invalid_flat_region")
        if record.detail_noise >= audit_cfg.invalid_noise_detail_min:
            reasons.append("invalid_noise")

        if reasons:
            record.decision = "remove_candidate"
            record.delete_category = STAGE_DELETE_CATEGORY["invalid"]
            record.reasons = tuple(reasons)
        else:
            record.decision = "keep"
            record.reasons = ()

        record.risk_score = 0.0
        record.risk_score += record.long_flat_points / 45.0
        record.risk_score += record.flat_fraction * 8.0
        record.risk_score += max(record.detail_noise - audit_cfg.invalid_noise_detail_min, 0.0) * 10.0
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

    if _finite_le(record.corr_ref, audit_cfg.class_corr_ref_remove_max):
        reasons.append("class_low_corr_ref")
        strong_count += 1

    if _finite_ge(record.rmse_to_ref, audit_cfg.class_rmse_ref_remove_min):
        reasons.append("class_high_rmse_to_ref")
        strong_count += 1

    local_remove = _finite_ge(record.local_residual_area, audit_cfg.class_local_residual_remove_area_min)
    if local_remove:
        reasons.append("class_local_residual_anomaly")

    overall_strong = strong_count >= 2
    is_candidate = overall_strong or local_remove
    return is_candidate, reasons


def _risk_score(record, audit_cfg: AuditConfig):
    """计算类内相似性风险排序分"""
    score = 0.0
    if np.isfinite(record.corr_ref):
        score += max(audit_cfg.class_corr_ref_remove_max - record.corr_ref, 0.0) * 20.0
    if np.isfinite(record.rmse_to_ref):
        score += max(record.rmse_to_ref - audit_cfg.class_rmse_ref_remove_min, 0.0) * 8.0
    if np.isfinite(record.local_residual_area):
        score += max(record.local_residual_area - audit_cfg.class_local_residual_remove_area_min, 0.0) / 20.0
    if np.isfinite(record.local_residual_max):
        score += max(record.local_residual_max - audit_cfg.class_local_residual_min, 0.0)
    return score


def classify_class_similarity(records, audit_cfg: AuditConfig):
    """第二阶段：用同属同前缀参考池判断类内相似性"""
    preliminary: dict[int, tuple[bool, tuple[str, ...]]] = {}
    group_totals = {}
    group_candidate_counts = {}

    for idx, record in enumerate(records):
        record.stage = "similar"
        record.delete_category = ""
        if record.z is None:
            record.decision = "skip"
            record.reasons = (record.skip_reason or "preprocess_failed",)
            continue

        group_totals[record.group] = group_totals.get(record.group, 0) + 1
        if record.ref_pool_size < audit_cfg.class_min_ref_samples:
            preliminary[idx] = (False, ())
            continue

        is_candidate, reasons = _candidate_evidence(record, audit_cfg)
        preliminary[idx] = (is_candidate, tuple(reasons))
        if is_candidate:
            group_candidate_counts[record.group] = group_candidate_counts.get(record.group, 0) + 1

    for idx, record in enumerate(records):
        if record.z is None:
            continue

        group_count = group_candidate_counts.get(record.group, 0)
        group_total = max(group_totals.get(record.group, 0), 1)
        record.folder_candidate_count = int(group_count)
        record.folder_candidate_fraction = float(group_count / group_total)

        is_candidate, reasons = preliminary.get(idx, (False, ()))
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

        record.decision = "remove_candidate"
        record.delete_category = STAGE_DELETE_CATEGORY["similar"]
        record.reasons = tuple(dict.fromkeys(reasons))
        record.risk_score = _risk_score(record, audit_cfg)
        record.risk_score += 5.0
