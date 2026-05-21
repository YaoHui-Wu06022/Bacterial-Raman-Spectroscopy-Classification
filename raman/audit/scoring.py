"""数据审核共享评分逻辑"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman.audit.common import (
    contiguous_regions,
    corr_many_to_one,
    moving_average,
    output_wn,
    prefix_of,
    region_width_cm,
    robust_mad_scale,
    robust_scale,
    robust_wave_stats,
    spectral_corr,
)
from raman.audit.config import AuditConfig, DEFAULT_AUDIT_CONFIG


@dataclass
class SpectrumRecord:
    path: Path
    rel_path: str
    group: str
    genus: str
    folder: str
    file: str
    z: np.ndarray | None = None
    sp: np.ndarray | None = None
    skip_reason: str = ""
    cosmic_total: int = 0
    cosmic_narrow: int = 0
    cosmic_peak: int = 0
    cosmic_residual: int = 0

    group_score: float = np.nan
    corr_group: float = np.nan
    rmse_group: float = np.nan
    max_abs_z_group: float = np.nan
    p95_abs_z_group: float = np.nan
    bad_ratio_group: float = np.nan
    max_pos_z_group: float = np.nan
    positive_bad_ratio_z8: float = np.nan
    residual_cosmic_regions: int = 0

    corr_ref: float = np.nan
    nearest_ref_corr: float = np.nan
    bad_ratio_z6: float = np.nan
    bad_ratio_z8: float = np.nan
    max_abs_z_ref: float = np.nan
    rmse_to_ref: float = np.nan

    roughness: float = np.nan
    roughness_z: float = np.nan
    step_count: int = 0
    bad_band_edge_step_count: int = 0
    max_step_delta: float = 0.0
    max_step_z: float = 0.0
    step_positions: tuple[float, ...] = ()

    decision: str = "keep"
    reasons: tuple[str, ...] = ()
    risk_score: float = 0.0


@dataclass
class FolderRecord:
    group: str
    genus: str
    folder: str
    files: int
    valid: int
    skipped: int
    remove_candidates: int = 0
    review_candidates: int = 0
    candidate_fraction: float = 0.0
    group_flagged_fraction: float = 0.0
    ref_remove_fraction: float = 0.0
    folder_corr_ref: float = np.nan
    median_cosmic_total: float = np.nan
    p95_cosmic_total: float = np.nan
    max_cosmic_total: int = 0
    step_spectra: int = 0
    residual_cosmic_spectra: int = 0
    decision: str = "keep"
    reasons: tuple[str, ...] = ()


def is_bad_band_edge(pos, bad_bands, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    for band_min, band_max in bad_bands:
        if abs(pos - band_min) <= audit_cfg.step_edge_cm or abs(pos - band_max) <= audit_cfg.step_edge_cm:
            return True
    return False


def detect_steps(wn, z, bad_bands, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    """检测预处理后仍存在的持续平台或突变台阶"""
    smooth = moving_average(z, audit_cfg.step_smooth_points)
    diff = np.diff(smooth)
    diff_scale = robust_mad_scale(diff)
    jump_z = np.abs(diff) / diff_scale
    candidates = np.where(jump_z >= audit_cfg.step_jump_z_threshold)[0]

    positions = []
    edge_positions = []
    max_delta = 0.0
    max_step_z = 0.0
    for idx in candidates:
        left_start = max(0, idx - audit_cfg.step_gap_points - audit_cfg.step_side_points)
        left_end = max(0, idx - audit_cfg.step_gap_points)
        right_start = min(smooth.size, idx + audit_cfg.step_gap_points + 1)
        right_end = min(smooth.size, idx + audit_cfg.step_gap_points + 1 + audit_cfg.step_side_points)
        if left_end - left_start < audit_cfg.step_side_points // 2:
            continue
        if right_end - right_start < audit_cfg.step_side_points // 2:
            continue

        left_level = float(np.median(smooth[left_start:left_end]))
        right_level = float(np.median(smooth[right_start:right_end]))
        delta = right_level - left_level
        level_z = abs(delta) / diff_scale
        if abs(delta) < audit_cfg.step_min_delta or level_z < audit_cfg.step_level_z_threshold:
            continue

        sign = np.sign(diff[idx])
        local_start = max(0, idx - audit_cfg.step_opposite_window)
        local_end = min(diff.size, idx + audit_cfg.step_opposite_window + 1)
        local_diff = diff[local_start:local_end]
        opposite = np.any((np.sign(local_diff) == -sign) & (np.abs(local_diff) >= abs(diff[idx]) * 0.6))
        if opposite:
            continue

        pos = float(wn[min(idx + 1, len(wn) - 1)])
        max_delta = max(max_delta, abs(delta))
        max_step_z = max(max_step_z, level_z)
        if is_bad_band_edge(pos, bad_bands, audit_cfg):
            edge_positions.append(pos)
        else:
            positions.append(pos)

    return {
        "step_count": len(positions),
        "bad_band_edge_step_count": len(edge_positions),
        "max_step_delta": max_delta,
        "max_step_z": max_step_z,
        "step_positions": tuple(positions + edge_positions),
    }


def residual_cosmic_regions(wn, signed_z, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    """统计清理后残留的短宽度正向高残差片段"""
    mask = signed_z >= audit_cfg.residual_pos_z_threshold
    count = 0
    for start, end in contiguous_regions(mask):
        if region_width_cm(wn, start, end) <= audit_cfg.residual_max_width_cm:
            count += 1
    return count


def score_group_payloads(payloads, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    """为单文件夹审核载荷计算组内离群评分"""
    valid_payloads = [item for item in payloads if not item["skip_reason"]]
    if len(valid_payloads) < audit_cfg.min_group_samples:
        return valid_payloads, None

    spectra = np.vstack([item["z"] for item in valid_payloads])
    center, wave_scale = robust_wave_stats(spectra, min_scale=1e-8, floor_fraction=1.0)
    q10 = np.quantile(spectra, 0.10, axis=0)
    q90 = np.quantile(spectra, 0.90, axis=0)
    residual = spectra - center
    rmse_values = np.sqrt(np.mean(residual * residual, axis=1))
    rmse_center, rmse_scale = robust_scale(rmse_values)

    for item, rmse, diff in zip(valid_payloads, rmse_values, residual):
        robust_z = diff / wave_scale
        abs_robust_z = np.abs(robust_z)
        item["rmse"] = float(rmse)
        item["score"] = float((rmse - rmse_center) / rmse_scale)
        item["corr"] = spectral_corr(item["z"], center)
        item["max_abs_z"] = float(np.max(abs_robust_z))
        item["p95_abs_z"] = float(np.quantile(abs_robust_z, 0.95))
        item["bad_point_ratio"] = float(np.mean(abs_robust_z > audit_cfg.group_point_z_threshold))
        item["robust_z"] = robust_z
        item["flagged"] = (
            item["score"] >= audit_cfg.group_score_threshold
            or item["corr"] <= audit_cfg.group_corr_threshold
            or item["bad_point_ratio"] >= audit_cfg.group_bad_ratio_threshold
        )

    return valid_payloads, {"center": center, "q10": q10, "q90": q90, "wave_scale": wave_scale}


def reference_thresholds(ref_arr, ref_median, ref_scale, ref_cosmic, audit_cfg: AuditConfig):
    """根据参考谱自身分布生成自适应阈值"""
    ref_corrs = corr_many_to_one(ref_arr, ref_median)
    ref_bad6 = np.mean(np.abs((ref_arr - ref_median) / ref_scale) > 6.0, axis=1)
    ref_rmse = np.sqrt(np.mean((ref_arr - ref_median) ** 2, axis=1))
    return {
        "corr_threshold": max(audit_cfg.ref_corr_floor, float(np.percentile(ref_corrs, 1) - audit_cfg.ref_corr_margin)),
        "nearest_threshold": max(
            audit_cfg.nearest_ref_corr_floor,
            float(np.percentile(ref_corrs, 5) - audit_cfg.nearest_ref_corr_margin),
        ),
        "bad_threshold": max(
            audit_cfg.ref_bad_ratio_floor,
            float(np.percentile(ref_bad6, 99) + audit_cfg.ref_bad_ratio_margin),
        ),
        "rmse_threshold": max(audit_cfg.ref_rmse_floor, float(np.percentile(ref_rmse, 99) * audit_cfg.ref_rmse_multiplier)),
        "cosmic_threshold": max(
            audit_cfg.ref_cosmic_floor,
            int(np.percentile(ref_cosmic, 99) + audit_cfg.ref_cosmic_margin),
        )
        if len(ref_cosmic)
        else 120,
    }


def build_reference_index(records):
    by_genus = {}
    by_genus_prefix = {}
    for record in records:
        if record.z is None:
            continue
        by_genus.setdefault(record.genus, []).append(record)
        by_genus_prefix.setdefault((record.genus, prefix_of(record.folder)), []).append(record)
    return by_genus, by_genus_prefix


def reference_for_group(group_records, by_genus, by_genus_prefix, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    first = group_records[0]
    same_prefix = [
        record
        for record in by_genus_prefix.get((first.genus, prefix_of(first.folder)), [])
        if record.group != first.group
    ]
    if len(same_prefix) >= audit_cfg.min_ref_files:
        return same_prefix
    return [record for record in by_genus.get(first.genus, []) if record.group != first.group]


def score_reference_rows(target_files, target_arr, target_stats, ref_arr, ref_stats, audit_cfg: AuditConfig):
    ref_median, ref_scale = robust_wave_stats(ref_arr, min_scale=0.05, floor_fraction=0.25)
    folder_median = np.median(target_arr, axis=0)
    thresholds = reference_thresholds(
        ref_arr,
        ref_median,
        ref_scale,
        np.asarray([int(getattr(stats, "total", 0)) for stats in ref_stats], dtype=np.float32),
        audit_cfg,
    )
    folder_corrs = np.array([spectral_corr(spec, folder_median) for spec in target_arr])
    thresholds["corr_folder_min"] = max(0.85, float(np.percentile(folder_corrs, 10) - 0.05))
    thresholds["folder_corr_ref"] = spectral_corr(folder_median, ref_median)
    thresholds["folder_warning"] = bool(thresholds["folder_corr_ref"] < audit_cfg.folder_corr_ref_warning)

    rows = []
    for path, spec, stats in zip(target_files, target_arr, target_stats):
        abs_dz = np.abs((spec - ref_median) / ref_scale)
        row = {
            "file": path.name,
            "path": str(path),
            "corr_ref": spectral_corr(spec, ref_median),
            "nearest_ref_corr": float(np.max(corr_many_to_one(ref_arr, spec))),
            "corr_folder": spectral_corr(spec, folder_median),
            "bad_ratio_z6": float(np.mean(abs_dz > 6.0)),
            "bad_ratio_z8": float(np.mean(abs_dz > 8.0)),
            "max_abs_z": float(np.max(abs_dz)),
            "rmse_to_ref": float(np.sqrt(np.mean((spec - ref_median) ** 2))),
            "cosmic_total": int(getattr(stats, "total", 0)),
            "cosmic_narrow": int(getattr(stats, "narrow", 0)),
            "cosmic_peak": int(getattr(stats, "peak", 0)),
            "cosmic_residual": int(getattr(stats, "residual", 0)),
        }

        reasons = []
        if row["corr_ref"] < thresholds["corr_threshold"] and row["nearest_ref_corr"] < thresholds["nearest_threshold"]:
            reasons.append("low_ref_similarity")
        if row["bad_ratio_z6"] > thresholds["bad_threshold"]:
            reasons.append("many_ref_point_outliers")
        if row["rmse_to_ref"] > thresholds["rmse_threshold"]:
            reasons.append("high_rmse_to_ref")
        if row["cosmic_total"] > thresholds["cosmic_threshold"]:
            reasons.append("excessive_cosmic_cleanup")
        if row["corr_folder"] < thresholds["corr_folder_min"]:
            reasons.append("low_folder_similarity")

        row["decision"] = "remove" if len(reasons) >= 2 else "keep"
        row["reasons"] = ",".join(reasons)
        rows.append(row)

    return rows, thresholds, ref_median, ref_scale, folder_median


def score_groups(records, cfg, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    group_stats = {}
    wn = output_wn(cfg)
    for group in sorted({record.group for record in records}):
        valid = [record for record in records if record.group == group and record.z is not None]
        if len(valid) < audit_cfg.min_group_samples:
            group_stats[group] = None
            continue

        arr = np.vstack([record.z for record in valid])
        center, wave_scale = robust_wave_stats(arr, min_scale=1e-8, floor_fraction=1.0)
        q10 = np.quantile(arr, 0.10, axis=0)
        q90 = np.quantile(arr, 0.90, axis=0)
        residual = arr - center
        rmse_values = np.sqrt(np.mean(residual * residual, axis=1))
        rmse_center, rmse_scale = robust_scale(rmse_values)
        roughness_values = np.array([robust_mad_scale(np.diff(record.z)) for record in valid], dtype=np.float32)
        rough_center, rough_scale = robust_scale(roughness_values)

        for record, rmse, diff, roughness in zip(valid, rmse_values, residual, roughness_values):
            signed_z = diff / wave_scale
            abs_z = np.abs(signed_z)
            record.rmse_group = float(rmse)
            record.group_score = float((rmse - rmse_center) / rmse_scale)
            record.corr_group = spectral_corr(record.z, center)
            record.max_abs_z_group = float(np.max(abs_z))
            record.p95_abs_z_group = float(np.quantile(abs_z, 0.95))
            record.bad_ratio_group = float(np.mean(abs_z > audit_cfg.group_point_z_threshold))
            record.max_pos_z_group = float(np.max(signed_z))
            record.positive_bad_ratio_z8 = float(np.mean(signed_z > audit_cfg.group_point_z_threshold))
            record.residual_cosmic_regions = residual_cosmic_regions(wn, signed_z, audit_cfg)
            record.roughness = float(roughness)
            record.roughness_z = float((roughness - rough_center) / rough_scale)
            step = detect_steps(wn, record.z, cfg.bad_bands, audit_cfg)
            record.step_count = step["step_count"]
            record.bad_band_edge_step_count = step["bad_band_edge_step_count"]
            record.max_step_delta = step["max_step_delta"]
            record.max_step_z = step["max_step_z"]
            record.step_positions = step["step_positions"]

        group_stats[group] = {"center": center, "q10": q10, "q90": q90, "wave_scale": wave_scale, "valid": len(valid)}
    return group_stats


def score_references(records, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    by_genus, by_genus_prefix = build_reference_index(records)
    ref_stats_by_group = {}
    for group in sorted({record.group for record in records}):
        valid = [record for record in records if record.group == group and record.z is not None]
        if not valid:
            ref_stats_by_group[group] = None
            continue
        refs = reference_for_group(valid, by_genus, by_genus_prefix, audit_cfg)
        if len(refs) < audit_cfg.min_ref_samples:
            ref_stats_by_group[group] = None
            continue

        ref_arr = np.vstack([record.z for record in refs])
        ref_median, ref_scale = robust_wave_stats(ref_arr, min_scale=0.05, floor_fraction=0.25)
        thresholds = reference_thresholds(
            ref_arr,
            ref_median,
            ref_scale,
            np.asarray([record.cosmic_total for record in refs], dtype=np.float32),
            audit_cfg,
        )

        for record in valid:
            abs_dz = np.abs((record.z - ref_median) / ref_scale)
            record.corr_ref = spectral_corr(record.z, ref_median)
            record.nearest_ref_corr = float(np.max(corr_many_to_one(ref_arr, record.z)))
            record.bad_ratio_z6 = float(np.mean(abs_dz > 6.0))
            record.bad_ratio_z8 = float(np.mean(abs_dz > 8.0))
            record.max_abs_z_ref = float(np.max(abs_dz))
            record.rmse_to_ref = float(np.sqrt(np.mean((record.z - ref_median) ** 2)))

        group_arr = np.vstack([record.z for record in valid])
        folder_median = np.median(group_arr, axis=0)
        ref_stats_by_group[group] = {
            "ref_median": ref_median,
            "ref_scale": ref_scale,
            "ref_count": len(refs),
            "ref_dirs": sorted({record.folder for record in refs}),
            "folder_corr_ref": spectral_corr(folder_median, ref_median),
            **thresholds,
        }
    return ref_stats_by_group


def classify_records(records, ref_stats_by_group, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    for record in records:
        if record.z is None:
            record.decision = "skip"
            record.reasons = (record.skip_reason or "preprocess_failed",)
            continue

        reasons = []
        ref_stats = ref_stats_by_group.get(record.group)
        if record.group_score >= audit_cfg.group_score_threshold:
            reasons.append("group_shape_score")
        if record.corr_group <= audit_cfg.group_corr_threshold:
            reasons.append("low_group_corr")
        if record.bad_ratio_group >= audit_cfg.group_bad_ratio_threshold:
            reasons.append("group_point_outlier")
        if record.roughness_z >= audit_cfg.roughness_z_threshold and record.roughness >= audit_cfg.roughness_min:
            reasons.append("rough_noise_outlier")

        if ref_stats is not None:
            if record.corr_ref < ref_stats["corr_threshold"] and record.nearest_ref_corr < ref_stats["nearest_threshold"]:
                reasons.append("low_ref_similarity")
            if record.bad_ratio_z6 > ref_stats["bad_threshold"]:
                reasons.append("many_ref_point_outliers")
            if record.rmse_to_ref > ref_stats["rmse_threshold"]:
                reasons.append("high_rmse_to_ref")
            if record.cosmic_total > ref_stats["cosmic_threshold"]:
                reasons.append("excessive_cosmic_cleanup")

        if record.step_count > 0:
            reasons.append("step_like_spectrum")
        elif record.bad_band_edge_step_count > 0:
            reasons.append("bad_band_edge_step")
        if (
            record.residual_cosmic_regions > 0
            and record.max_pos_z_group >= audit_cfg.residual_min_max_z
            and record.positive_bad_ratio_z8 > 0
        ):
            reasons.append("residual_cosmic_like")

        ref_evidence = sum(reason in reasons for reason in ("low_ref_similarity", "many_ref_point_outliers", "high_rmse_to_ref"))
        group_evidence = sum(reason in reasons for reason in ("group_shape_score", "low_group_corr", "group_point_outlier"))
        strong_noise = "rough_noise_outlier" in reasons and (
            "group_point_outlier" in reasons
            or "many_ref_point_outliers" in reasons
            or ("low_ref_similarity" in reasons and record.corr_group <= 0.65)
        )
        strong_residual = "residual_cosmic_like" in reasons and (
            "group_point_outlier" in reasons or "many_ref_point_outliers" in reasons
        ) and ("low_group_corr" in reasons or "low_ref_similarity" in reasons)
        strong_ref = ref_evidence >= 3 and group_evidence >= 1
        strong_group = group_evidence >= 3 and ref_evidence >= 1
        review_residual = "residual_cosmic_like" in reasons and (
            record.max_pos_z_group >= audit_cfg.residual_review_max_pos_z
            or "group_point_outlier" in reasons
            or "many_ref_point_outliers" in reasons
        )
        review_group = (
            "group_point_outlier" in reasons
            or ("group_shape_score" in reasons and "low_group_corr" in reasons and record.corr_group <= 0.70)
        )
        review_ref = ref_evidence >= 2 and ("many_ref_point_outliers" in reasons or "high_rmse_to_ref" in reasons)
        review_noise = "rough_noise_outlier" in reasons and (
            "group_point_outlier" in reasons
            or "many_ref_point_outliers" in reasons
            or ("low_ref_similarity" in reasons and record.corr_group <= 0.75)
        )

        if "step_like_spectrum" in reasons or strong_noise or strong_residual or strong_ref or strong_group:
            record.decision = "remove_candidate"
        elif "bad_band_edge_step" in reasons or review_residual or review_group or review_ref or review_noise:
            record.decision = "review_candidate"
        else:
            record.decision = "keep"

        record.reasons = tuple(reasons)
        record.risk_score = float(
            max(record.group_score if np.isfinite(record.group_score) else 0.0, 0.0)
            + max((1.0 - record.corr_group) * 10.0 if np.isfinite(record.corr_group) else 0.0, 0.0)
            + max(record.bad_ratio_group * 80.0 if np.isfinite(record.bad_ratio_group) else 0.0, 0.0)
            + max(record.bad_ratio_z6 * 50.0 if np.isfinite(record.bad_ratio_z6) else 0.0, 0.0)
            + max(record.max_step_delta * 2.0, 0.0)
            + min(record.cosmic_total / 40.0, 6.0)
        )


def build_folder_records(records, group_stats, ref_stats_by_group, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    folders = []
    for group in sorted({record.group for record in records}):
        group_records = [record for record in records if record.group == group]
        valid = [record for record in group_records if record.z is not None]
        genus = group_records[0].genus if group_records else ""
        folder = group_records[0].folder if group_records else ""
        candidate_records = [record for record in valid if record.decision in {"remove_candidate", "review_candidate"}]
        remove_records = [record for record in valid if record.decision == "remove_candidate"]
        group_flagged = [
            record
            for record in valid
            if (
                record.group_score >= audit_cfg.group_score_threshold
                or record.corr_group <= audit_cfg.group_corr_threshold
                or record.bad_ratio_group >= audit_cfg.group_bad_ratio_threshold
            )
        ]
        ref_remove = [
            record
            for record in valid
            if any(reason in record.reasons for reason in ("low_ref_similarity", "many_ref_point_outliers", "high_rmse_to_ref"))
        ]
        cosmic_values = np.asarray([record.cosmic_total for record in valid], dtype=np.float32)
        ref_stats = ref_stats_by_group.get(group) or {}
        folder_record = FolderRecord(
            group=group,
            genus=genus,
            folder=folder,
            files=len(group_records),
            valid=len(valid),
            skipped=len(group_records) - len(valid),
            remove_candidates=len(remove_records),
            review_candidates=len(candidate_records) - len(remove_records),
            candidate_fraction=len(candidate_records) / max(len(valid), 1),
            group_flagged_fraction=len(group_flagged) / max(len(valid), 1),
            ref_remove_fraction=len(ref_remove) / max(len(valid), 1),
            folder_corr_ref=float(ref_stats.get("folder_corr_ref", np.nan)),
            median_cosmic_total=float(np.median(cosmic_values)) if cosmic_values.size else np.nan,
            p95_cosmic_total=float(np.quantile(cosmic_values, 0.95)) if cosmic_values.size else np.nan,
            max_cosmic_total=int(np.max(cosmic_values)) if cosmic_values.size else 0,
            step_spectra=sum(record.step_count > 0 or record.bad_band_edge_step_count > 0 for record in valid),
            residual_cosmic_spectra=sum("residual_cosmic_like" in record.reasons for record in valid),
        )

        reasons = []
        if folder_record.valid < audit_cfg.min_group_samples:
            reasons.append("too_few_valid_spectra")
        if np.isfinite(folder_record.folder_corr_ref) and folder_record.folder_corr_ref < audit_cfg.folder_corr_ref_warning:
            reasons.append("folder_far_from_references")
        if folder_record.candidate_fraction >= audit_cfg.folder_candidate_fraction_review:
            reasons.append("many_candidate_spectra")
        if folder_record.ref_remove_fraction >= audit_cfg.folder_ref_outlier_fraction_review:
            reasons.append("many_ref_outliers")
        if folder_record.step_spectra / max(folder_record.valid, 1) >= audit_cfg.folder_step_fraction_review:
            reasons.append("many_step_like_spectra")

        if (
            folder_record.ref_remove_fraction >= audit_cfg.folder_ref_outlier_fraction_remove
            or folder_record.candidate_fraction >= audit_cfg.folder_candidate_fraction_remove
            or folder_record.step_spectra / max(folder_record.valid, 1) >= audit_cfg.folder_step_fraction_remove
        ):
            folder_record.decision = "remove_candidate"
        elif reasons:
            folder_record.decision = "review_candidate"
        folder_record.reasons = tuple(reasons)
        folders.append(folder_record)
    return folders


def cap_remove_candidates(records, max_remove_candidates):
    if max_remove_candidates <= 0:
        return
    remove_records = [record for record in records if record.decision == "remove_candidate"]
    if len(remove_records) <= max_remove_candidates:
        return
    selected_ids = {
        id(record)
        for record in sorted(remove_records, key=lambda item: (-item.risk_score, item.rel_path))[:max_remove_candidates]
    }
    for record in remove_records:
        if id(record) not in selected_ids:
            record.decision = "review_candidate"


def top_folder_candidates(folders, max_folder_candidates):
    candidates = [folder for folder in folders if folder.decision in {"remove_candidate", "review_candidate"}]
    candidates = sorted(candidates, key=lambda item: (item.decision != "remove_candidate", -item.candidate_fraction, item.group))
    if max_folder_candidates <= 0:
        return []
    return candidates[:max_folder_candidates]


def reason_labels(reasons, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    labels = []
    if "step_like_spectrum" in reasons or "bad_band_edge_step" in reasons or "residual_cosmic_like" in reasons:
        labels.append("阶梯谱")
    if "rough_noise_outlier" in reasons:
        labels.append("粗糙噪声")
    if any(reason in reasons for reason in ("low_ref_similarity", "many_ref_point_outliers", "high_rmse_to_ref", "excessive_cosmic_cleanup")):
        labels.append("参考组离群")
    if any(reason in reasons for reason in ("group_shape_score", "low_group_corr", "group_point_outlier")):
        labels.append("组内离群")
    return tuple(label for label in audit_cfg.delete_reason_labels if label in labels)


def record_to_row(record):
    labels = reason_labels(record.reasons)
    return {
        "decision": record.decision,
        "reasons": ";".join(record.reasons),
        "reason_labels": ";".join(labels),
        "rel_path": record.rel_path,
        "group": record.group,
        "file": record.file,
        "corr_group": f"{record.corr_group:.3f}" if np.isfinite(record.corr_group) else "",
        "bad_ratio_group": f"{record.bad_ratio_group:.3f}" if np.isfinite(record.bad_ratio_group) else "",
        "corr_ref": f"{record.corr_ref:.3f}" if np.isfinite(record.corr_ref) else "",
        "ref_bad_ratio": f"{record.bad_ratio_z6:.3f}" if np.isfinite(record.bad_ratio_z6) else "",
        "step_flag": int(record.step_count > 0 or record.bad_band_edge_step_count > 0),
        "residual_cosmic_regions": record.residual_cosmic_regions,
        "cosmic_total": record.cosmic_total,
    }


def folder_to_row(record):
    return {
        "decision": record.decision,
        "reasons": ";".join(record.reasons),
        "group": record.group,
        "files": record.files,
        "remove_candidates": record.remove_candidates,
        "review_candidates": record.review_candidates,
        "candidate_fraction": f"{record.candidate_fraction:.6f}",
        "folder_corr_ref": f"{record.folder_corr_ref:.6f}" if np.isfinite(record.folder_corr_ref) else "",
        "step_spectra": record.step_spectra,
    }
