"""audit 分阶段清洗的公共评分工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter

from raman.audit.common import (
    contiguous_regions,
    median_step_cm,
    moving_average,
    output_wn,
    prefix_of,
    region_width_cm,
    robust_mad_scale,
    robust_wave_stats,
)
from raman.audit.config import AuditConfig, DEFAULT_AUDIT_CONFIG
from raman.data.spectrum import read_arc_data


VALID_STAGES = ("invalid", "anomalous-cosmic", "class-similarity")
STAGE_DELETE_CATEGORY = {
    "invalid": "Invalid Spectrum",
    "anomalous-cosmic": "Anomalous_Cosmic_Rays",
    "class-similarity": "Class_Similarity_Outliers",
}


@dataclass
class SpectrumRecord:
    """单条光谱的审核评分记录"""

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

    stage: str = ""
    decision: str = "keep"
    reasons: tuple[str, ...] = ()
    risk_score: float = 0.0
    delete_category: str = ""

    prefix_scope: str = ""

    raw_wn_min: float = np.nan
    raw_wn_max: float = np.nan
    raw_points: int = 0
    coverage_ratio: float = np.nan
    long_flat_points: int = 0
    flat_fraction: float = 0.0
    roughness: float = np.nan
    smooth_range: float = np.nan
    detail_noise: float = np.nan
    structure_ratio: float = np.nan

    wide_bump_count: int = 0
    wide_bump_max_z: float = 0.0
    wide_bump_area: float = 0.0
    wide_bump_width_points: int = 0
    wide_bump_width_cm: float = 0.0
    wide_bump_center_cm: float = 0.0
    wide_region_kind: str = ""
    wide_edge_jump_z: float = 0.0
    wide_left_edge_jump_z: float = 0.0
    wide_right_edge_jump_z: float = 0.0
    wide_bump_positions: tuple[float, ...] = ()
    wide_bump_ranges: tuple[tuple[int, int], ...] = ()
    wide_z: np.ndarray | None = None
    wide_smooth: np.ndarray | None = None
    wide_floor: np.ndarray | None = None
    rising_region_count: int = 0
    rising_region_area: float = 0.0
    rising_region_width_cm: float = 0.0
    rising_region_center_cm: float = 0.0

    ref_pool_scope: str = ""
    ref_pool_size: int = 0
    other_ref_pool_size: int = 0
    corr_ref: float = np.nan
    nearest_ref_corr: float = np.nan
    rmse_to_ref: float = np.nan
    local_pos_count: int = 0
    local_pos_max_z: float = 0.0
    local_pos_area: float = 0.0
    local_pos_width_points: int = 0
    local_pos_width_cm: float = 0.0
    local_pos_center_cm: float = 0.0
    local_pos_positions: tuple[float, ...] = ()
    local_pos_ranges: tuple[tuple[int, int], ...] = ()
    local_pos_z: np.ndarray | None = None
    folder_candidate_count: int = 0
    folder_candidate_fraction: float = 0.0


def validate_stage(stage: str) -> str:
    """校验并规范化 audit 阶段名。"""
    stage = str(stage or "").strip().lower()
    if stage not in VALID_STAGES:
        raise ValueError(f"Unsupported audit stage: {stage}. Choose one of: {', '.join(VALID_STAGES)}")
    return stage


def prefix_scope(record: SpectrumRecord) -> str:
    """返回属/前缀形式的参考池键。"""
    return f"{record.genus}/{prefix_of(record.folder)}"


def fmt_value(value, digits=3):
    """把有限数值格式化成 CSV 文本。"""
    try:
        value = float(value)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def raw_coverage(wn, cut_min, cut_max):
    """计算原始波数轴覆盖目标裁剪区间的比例。"""
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size == 0:
        return np.nan
    low = max(float(np.min(wn)), float(cut_min))
    high = min(float(np.max(wn)), float(cut_max))
    return max(0.0, high - low) / max(float(cut_max) - float(cut_min), 1e-8)


def contiguous_index_ranges(wn):
    """按波数轴大间隔拆成连续索引段。"""
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size == 0:
        return []
    step = median_step_cm(wn)
    cuts = np.where(np.abs(np.diff(wn)) > step * 1.8)[0] + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [wn.size]])
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def longest_flat_points(wn, z, audit_cfg: AuditConfig):
    """计算最长低变化平坦段点数。"""
    wn = np.asarray(wn, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    window = max(3, int(audit_cfg.invalid_flat_window_points))
    best_points = 0
    for start, end in contiguous_index_ranges(wn):
        segment = z[start:end]
        if segment.size < window:
            continue
        current = 0
        for idx in range(0, segment.size - window + 1):
            is_flat = float(np.ptp(segment[idx : idx + window])) <= audit_cfg.invalid_flat_range_max
            if is_flat:
                current += 1
                best_points = max(best_points, current + window - 1)
            else:
                current = 0
    return int(best_points)


def rolling_quantile(values, window_points, q=0.20):
    """计算滚动分位数曲线。"""
    values = np.asarray(values, dtype=np.float32)
    window = max(3, int(window_points))
    if window % 2 == 0:
        window += 1
    if values.size == 0:
        return values.copy()
    return percentile_filter(values, percentile=float(q) * 100.0, size=window, mode="nearest").astype(np.float32, copy=False)


def _diff_scale_by_segments(values, segments):
    """按连续段计算一阶差分鲁棒尺度。"""
    diffs = []
    for start, end in segments:
        if end - start >= 2:
            diffs.append(np.diff(values[start:end]))
    if not diffs:
        return 1e-8
    return robust_mad_scale(np.concatenate(diffs))


def _region_edge_jump_z(smooth_values, start, end, seg_start, seg_end, scale, window_points):
    """估计宽异常段两侧边缘跳变强度。"""
    diffs = np.diff(np.asarray(smooth_values, dtype=np.float32))
    if diffs.size == 0:
        return 0.0, 0.0, 0.0

    window = max(2, int(window_points))
    left_start = max(int(seg_start), int(start) - window)
    left_end = min(int(seg_end) - 1, int(start) + window)
    right_start = max(int(seg_start), int(end) - window)
    right_end = min(int(seg_end) - 1, int(end) + window)

    left_jump = 0.0
    if left_end > left_start:
        left_jump = max(float(np.max(diffs[left_start:left_end])), 0.0)
    right_jump = 0.0
    if right_end > right_start:
        right_jump = max(float(np.max(-diffs[right_start:right_end])), 0.0)

    denom = max(float(scale), 1e-8)
    left_z = left_jump / denom
    right_z = right_jump / denom
    return max(left_z, right_z), left_z, right_z


def _wide_regions_by_points(wn, z_values, smooth_values, threshold, min_points, segments, edge_scale, edge_window_points):
    """按点数阈值提取宽平台/阶梯候选段。"""
    regions = []
    for seg_start, seg_end in segments:
        mask = z_values[seg_start:seg_end] >= float(threshold)
        for rel_start, rel_end in contiguous_regions(mask):
            start = seg_start + int(rel_start)
            end = seg_start + int(rel_end)
            width_points = end - start
            if width_points < int(min_points):
                continue
            segment = z_values[start:end]
            if segment.size == 0:
                continue
            peak_idx = int(start + np.argmax(segment))
            edge_jump_z, left_edge_jump_z, right_edge_jump_z = _region_edge_jump_z(
                smooth_values,
                start,
                end,
                seg_start,
                seg_end,
                edge_scale,
                edge_window_points,
            )
            regions.append(
                {
                    "start": start,
                    "end": end,
                    "center_cm": float(wn[peak_idx]),
                    "width_points": int(width_points),
                    "width_cm": region_width_cm(wn, start, end),
                    "max_z": float(np.max(segment)),
                    "area": float(np.sum(np.maximum(segment, 0.0))),
                    "kind": "edge",
                    "edge_jump_z": float(edge_jump_z),
                    "left_edge_jump_z": float(left_edge_jump_z),
                    "right_edge_jump_z": float(right_edge_jump_z),
                }
            )
    return regions


def _long_rising_regions(wn, z_values, audit_cfg: AuditConfig, segments):
    """检测清理后仍存在的长上升尾段"""
    regions = []
    scale = _diff_scale_by_segments(z_values, segments)
    for seg_start, seg_end in segments:
        if seg_end - seg_start < int(audit_cfg.anomalous_rising_min_points):
            continue
        segment = moving_average(z_values[seg_start:seg_end], audit_cfg.anomalous_wide_smooth_points)
        edge_points = max(25, int(segment.size // 5))
        left = float(np.median(segment[:edge_points]))
        right = float(np.median(segment[-edge_points:]))
        center = float(np.median(segment))
        rise_z = (right - left) / max(scale, 1e-8)
        end_lift = right - center
        if rise_z < float(audit_cfg.anomalous_rising_z_min) or end_lift < float(audit_cfg.anomalous_rising_snv_min):
            continue

        start_threshold = left + 0.30 * (right - left)
        start_candidates = np.flatnonzero(segment >= start_threshold)
        start = seg_start
        if start_candidates.size:
            start = seg_start + int(max(0, start_candidates[0] - audit_cfg.anomalous_wide_smooth_points))
        end = seg_end
        width_points = int(end - start)
        if width_points < int(audit_cfg.anomalous_rising_min_points):
            continue

        center_idx = min(end - 1, max(start, start + width_points // 2))
        regions.append(
            {
                "start": int(start),
                "end": int(end),
                "center_cm": float(wn[center_idx]),
                "width_points": width_points,
                "width_cm": region_width_cm(wn, start, end),
                "max_z": float(rise_z),
                "area": float(rise_z * width_points),
                "kind": "rising",
                "edge_jump_z": 0.0,
                "left_edge_jump_z": 0.0,
                "right_edge_jump_z": 0.0,
            }
        )
    return regions


def score_anomalous_wide_regions(records, cfg, audit_cfg: AuditConfig):
    """为第二阶段宽上升平台/阶梯异常打分。"""
    wn_full = output_wn(cfg)
    segments_by_size = {}
    for record in records:
        if record.sp is None:
            continue

        values = np.asarray(record.sp, dtype=np.float32)
        wn = wn_full[: values.size]
        segments = segments_by_size.get(values.size)
        if segments is None:
            segments = contiguous_index_ranges(wn)
            segments_by_size[values.size] = segments
        smooth = np.empty_like(values, dtype=np.float32)
        floor = np.empty_like(values, dtype=np.float32)
        for start, end in segments:
            segment = values[start:end]
            smooth_segment = moving_average(segment, audit_cfg.anomalous_wide_smooth_points)
            smooth[start:end] = smooth_segment
            floor[start:end] = rolling_quantile(
                smooth_segment,
                audit_cfg.anomalous_wide_floor_window_points,
                q=0.20,
            )

        residual = smooth - floor
        scale = _diff_scale_by_segments(values, segments)
        wide_z = residual / max(scale, 1e-8)
        edge_regions = _wide_regions_by_points(
            wn,
            wide_z,
            smooth,
            threshold=audit_cfg.anomalous_wide_z_min,
            min_points=audit_cfg.anomalous_wide_min_points,
            segments=segments,
            edge_scale=scale,
            edge_window_points=audit_cfg.anomalous_wide_smooth_points,
        )
        edge_threshold = audit_cfg.anomalous_wide_review_edge_z_min
        edge_regions = [region for region in edge_regions if region["edge_jump_z"] >= edge_threshold]
        rising_regions = []
        if record.z is not None:
            rising_regions = _long_rising_regions(wn, np.asarray(record.z, dtype=np.float32), audit_cfg, segments)
        regions = edge_regions + rising_regions
        best = max(
            regions,
            key=lambda item: (item["kind"] == "rising", item.get("edge_jump_z", 0.0), item["max_z"], item["area"], item["width_points"]),
            default=None,
        )

        record.wide_z = wide_z.astype(np.float32, copy=False)
        record.wide_smooth = smooth
        record.wide_floor = floor
        record.wide_bump_count = len(regions)
        record.wide_bump_positions = tuple(region["center_cm"] for region in regions)
        record.wide_bump_ranges = tuple((int(region["start"]), int(region["end"])) for region in regions)
        if best is not None:
            record.wide_bump_max_z = best["max_z"]
            record.wide_bump_area = best["area"]
            record.wide_bump_width_points = best["width_points"]
            record.wide_bump_width_cm = best["width_cm"]
            record.wide_bump_center_cm = best["center_cm"]
            record.wide_region_kind = best.get("kind", "")
            record.wide_edge_jump_z = best["edge_jump_z"]
            record.wide_left_edge_jump_z = best["left_edge_jump_z"]
            record.wide_right_edge_jump_z = best["right_edge_jump_z"]
            record.rising_region_count = len(rising_regions)
            if best.get("kind") == "rising":
                record.rising_region_area = best["area"]
                record.rising_region_width_cm = best["width_cm"]
                record.rising_region_center_cm = best["center_cm"]


def _local_positive_regions(wn, z_values, audit_cfg: AuditConfig):
    """提取类内参考残差里的局部正异常段。"""
    regions = []
    for start, end in contiguous_regions(z_values >= float(audit_cfg.class_local_z_min)):
        width_points = int(end - start)
        if width_points < int(audit_cfg.class_local_width_min_points):
            continue
        segment = z_values[start:end]
        if segment.size == 0:
            continue
        peak_idx = int(start + np.argmax(segment))
        regions.append(
            {
                "start": int(start),
                "end": int(end),
                "center_cm": float(wn[peak_idx]),
                "width_points": width_points,
                "width_cm": region_width_cm(wn, start, end),
                "max_z": float(np.max(segment)),
                "area": float(np.sum(np.maximum(segment, 0.0))),
            }
        )
    return regions


def _row_corr_to_one(spectra, reference):
    """计算多条谱到一条参考谱的相关系数"""
    spectra = np.asarray(spectra, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if spectra.size == 0 or reference.std() <= 1e-8:
        return np.zeros(spectra.shape[0], dtype=np.float32)
    x = spectra - spectra.mean(axis=1, keepdims=True)
    y = reference - float(reference.mean())
    denom = np.linalg.norm(x, axis=1) * float(np.linalg.norm(y))
    corr = np.zeros(spectra.shape[0], dtype=np.float32)
    ok = denom > 1e-8
    if np.any(ok):
        corr[ok] = (x[ok] @ y / denom[ok]).astype(np.float32, copy=False)
    return corr


def _nearest_other_folder_corr(spectra, folders, block_size=256):
    """分块计算非同小文件夹最近邻相关性"""
    spectra = np.asarray(spectra, dtype=np.float32)
    folders = np.asarray(folders, dtype=object)
    count = int(spectra.shape[0])
    nearest = np.full(count, np.nan, dtype=np.float32)
    other_counts = np.zeros(count, dtype=np.int32)
    if count <= 1:
        return nearest, other_counts

    centered = spectra - spectra.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 1e-8
    normed = np.zeros_like(centered, dtype=np.float32)
    normed[valid] = centered[valid] / norms[valid, None]

    folder_indices = {folder: np.flatnonzero(folders == folder) for folder in np.unique(folders)}
    for indices in folder_indices.values():
        other_counts[indices] = count - int(indices.size)

    block_size = max(16, int(block_size))
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        sim = normed[start:end] @ normed.T
        sim[:, ~valid] = -np.inf
        for row, idx in enumerate(range(start, end)):
            sim[row, folder_indices[folders[idx]]] = -np.inf
        best = np.max(sim, axis=1)
        ok = np.isfinite(best)
        nearest[start:end][ok] = best[ok].astype(np.float32, copy=False)
    return nearest, other_counts


def score_class_similarity(records, cfg, audit_cfg: AuditConfig):
    """为第三阶段同前缀类内相似性打分。"""
    wn_full = output_wn(cfg)
    by_scope: dict[str, list[SpectrumRecord]] = {}
    for record in records:
        record.prefix_scope = prefix_scope(record)
        record.ref_pool_scope = record.prefix_scope
        if record.z is not None:
            by_scope.setdefault(record.prefix_scope, []).append(record)

    prefix_stats = {}
    for scope, scope_records in by_scope.items():
        spectra = np.stack([record.z for record in scope_records]).astype(np.float32, copy=False)
        folders = np.asarray([record.folder for record in scope_records], dtype=object)
        center_all, scale_all = robust_wave_stats(spectra)
        prefix_stats[scope] = {
            "center": center_all,
            "wave_scale": scale_all,
            "q10": np.quantile(spectra, 0.10, axis=0).astype(np.float32),
            "q90": np.quantile(spectra, 0.90, axis=0).astype(np.float32),
            "records": scope_records,
        }

        if len(scope_records) < int(audit_cfg.class_min_ref_samples):
            for record in scope_records:
                record.ref_pool_size = max(len(scope_records) - 1, 0)
            continue

        corr_ref = _row_corr_to_one(spectra, center_all)
        rmse_to_ref = np.sqrt(np.mean((spectra - center_all) ** 2, axis=1))
        nearest_ref, other_counts = _nearest_other_folder_corr(spectra, folders)
        for idx, record in enumerate(scope_records):
            sample = spectra[idx]
            record.ref_pool_size = int(len(scope_records) - 1)
            record.other_ref_pool_size = int(other_counts[idx])
            record.corr_ref = float(corr_ref[idx])
            record.rmse_to_ref = float(rmse_to_ref[idx])
            if np.isfinite(nearest_ref[idx]):
                record.nearest_ref_corr = float(nearest_ref[idx])

            local_z = (sample - center_all) / np.maximum(scale_all, 1e-8)
            record.local_pos_z = local_z.astype(np.float32, copy=False)
            wn = wn_full[: sample.size]
            regions = _local_positive_regions(wn, local_z, audit_cfg)
            best = max(regions, key=lambda item: (item["max_z"], item["area"], item["width_points"]), default=None)
            record.local_pos_count = len(regions)
            record.local_pos_positions = tuple(region["center_cm"] for region in regions)
            record.local_pos_ranges = tuple((int(region["start"]), int(region["end"])) for region in regions)
            if best is not None:
                record.local_pos_max_z = best["max_z"]
                record.local_pos_area = best["area"]
                record.local_pos_width_points = best["width_points"]
                record.local_pos_width_cm = best["width_cm"]
                record.local_pos_center_cm = best["center_cm"]

    return prefix_stats


def score_raw_and_basic(records, cfg, audit_cfg: AuditConfig):
    """为第一阶段无效谱计算自身质量特征。"""
    wn = output_wn(cfg)
    for record in records:
        record.prefix_scope = prefix_scope(record)
        if record.z is None:
            continue

        if record.raw_points <= 0:
            try:
                raw_wn, _ = read_arc_data(record.path)
            except Exception:
                raw_wn = np.asarray([], dtype=np.float32)
            record.raw_points = int(raw_wn.size)
            if raw_wn.size:
                record.raw_wn_min = float(np.min(raw_wn))
                record.raw_wn_max = float(np.max(raw_wn))
            record.coverage_ratio = raw_coverage(raw_wn, cfg.cut_min, cfg.cut_max)
        record.long_flat_points = longest_flat_points(wn[: record.z.size], record.z, audit_cfg)
        median_level = float(np.median(record.z))
        record.flat_fraction = float(np.mean(np.abs(record.z - median_level) <= audit_cfg.invalid_flat_near_median))
        record.roughness = robust_mad_scale(np.diff(record.z))
        smooth = moving_average(record.z, audit_cfg.invalid_noise_smooth_points)
        record.smooth_range = float(np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05))
        record.detail_noise = robust_mad_scale(record.z - smooth)
        record.structure_ratio = record.smooth_range / max(record.roughness, 1e-8)


def score_stage(records, cfg, stage: str, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    """按阶段调度评分和分类规则。"""
    stage = validate_stage(stage)
    if stage == "invalid":
        from raman.audit.stage_invalid import classify_invalid

        score_raw_and_basic(records, cfg, audit_cfg)
        classify_invalid(records, audit_cfg)
    elif stage == "anomalous-cosmic":
        from raman.audit.stage_anomalous_cosmic import classify_anomalous_cosmic

        score_anomalous_wide_regions(records, cfg, audit_cfg)
        classify_anomalous_cosmic(records, audit_cfg)
    elif stage == "class-similarity":
        from raman.audit.stage_class_similarity import classify_class_similarity

        prefix_stats = score_class_similarity(records, cfg, audit_cfg)
        classify_class_similarity(records, audit_cfg)
        return prefix_stats
    return {}


def reason_labels(reasons, audit_cfg: AuditConfig = DEFAULT_AUDIT_CONFIG):
    """把内部原因映射成 delete 分类标签。"""
    reasons = set(reasons or ())
    labels = []
    if any(reason.startswith("invalid_") for reason in reasons):
        labels.append("Invalid Spectrum")
    if any(reason.startswith("anomalous_cosmic") for reason in reasons):
        labels.append("Anomalous_Cosmic_Rays")
    if any(reason.startswith("class_") for reason in reasons):
        labels.append("Class_Similarity_Outliers")
    return tuple(label for label in audit_cfg.delete_reason_labels if label in labels)


def record_to_row(record: SpectrumRecord):
    """把谱记录转换成 CSV 行。"""
    labels = reason_labels(record.reasons)
    return {
        "stage": record.stage,
        "decision": record.decision,
        "reasons": ";".join(record.reasons),
        "reason_labels": ";".join(labels),
        "delete_category": record.delete_category,
        "rel_path": record.rel_path,
        "genus": record.genus,
        "folder": record.folder,
        "group": record.group,
        "file": record.file,
        "prefix_scope": record.prefix_scope,
        "raw_wn_min": fmt_value(record.raw_wn_min, 3),
        "raw_wn_max": fmt_value(record.raw_wn_max, 3),
        "raw_points": record.raw_points,
        "coverage_ratio": fmt_value(record.coverage_ratio, 6),
        "long_flat_points": record.long_flat_points,
        "flat_fraction": fmt_value(record.flat_fraction, 6),
        "roughness": fmt_value(record.roughness, 6),
        "smooth_range": fmt_value(record.smooth_range, 6),
        "detail_noise": fmt_value(record.detail_noise, 6),
        "structure_ratio": fmt_value(record.structure_ratio, 6),
        "wide_bump_count": record.wide_bump_count,
        "wide_bump_max_z": fmt_value(record.wide_bump_max_z, 3),
        "wide_bump_area": fmt_value(record.wide_bump_area, 3),
        "wide_bump_width_points": record.wide_bump_width_points,
        "wide_bump_width_cm": fmt_value(record.wide_bump_width_cm, 3),
        "wide_bump_center_cm": fmt_value(record.wide_bump_center_cm, 3),
        "wide_region_kind": record.wide_region_kind,
        "wide_edge_jump_z": fmt_value(record.wide_edge_jump_z, 3),
        "wide_left_edge_jump_z": fmt_value(record.wide_left_edge_jump_z, 3),
        "wide_right_edge_jump_z": fmt_value(record.wide_right_edge_jump_z, 3),
        "wide_bump_positions": ";".join(f"{pos:.1f}" for pos in record.wide_bump_positions),
        "rising_region_count": record.rising_region_count,
        "rising_region_area": fmt_value(record.rising_region_area, 3),
        "rising_region_width_cm": fmt_value(record.rising_region_width_cm, 3),
        "rising_region_center_cm": fmt_value(record.rising_region_center_cm, 3),
        "ref_pool_scope": record.ref_pool_scope,
        "ref_pool_size": record.ref_pool_size,
        "other_ref_pool_size": record.other_ref_pool_size,
        "corr_ref": fmt_value(record.corr_ref, 6),
        "nearest_ref_corr": fmt_value(record.nearest_ref_corr, 6),
        "rmse_to_ref": fmt_value(record.rmse_to_ref, 6),
        "local_pos_count": record.local_pos_count,
        "local_pos_max_z": fmt_value(record.local_pos_max_z, 3),
        "local_pos_area": fmt_value(record.local_pos_area, 3),
        "local_pos_width_points": record.local_pos_width_points,
        "local_pos_width_cm": fmt_value(record.local_pos_width_cm, 3),
        "local_pos_center_cm": fmt_value(record.local_pos_center_cm, 3),
        "local_pos_positions": ";".join(f"{pos:.1f}" for pos in record.local_pos_positions),
        "folder_candidate_count": record.folder_candidate_count,
        "folder_candidate_fraction": fmt_value(record.folder_candidate_fraction, 6),
        "cosmic_total": record.cosmic_total,
        "cosmic_narrow": record.cosmic_narrow,
        "cosmic_peak": record.cosmic_peak,
        "risk_score": fmt_value(record.risk_score, 3),
    }


def stage_title(stage: str) -> str:
    """返回阶段展示名称。"""
    stage = validate_stage(stage)
    if stage == "anomalous-cosmic":
        return "Anomalous Cosmic Rays"
    if stage == "class-similarity":
        return "Class Similarity"
    return "Invalid Spectrum"
