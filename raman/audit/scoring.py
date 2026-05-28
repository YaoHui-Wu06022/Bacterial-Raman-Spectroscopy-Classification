"""audit 分阶段清洗的公共评分工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman.audit.config import AuditConfig, DEFAULT_AUDIT_CONFIG
from raman.data.io import read_arc_data
from raman.tool.array import (
    contiguous_regions,
    moving_average,
    robust_mad_scale,
    robust_wave_stats,
)
from raman.tool.naming import prefix_of
from raman.tool.spectrum import (
    contiguous_index_ranges,
    output_wavenumbers as output_wn,
    region_width_cm,
)


VALID_STAGES = ("invalid", "class-similarity")
STAGE_DELETE_CATEGORY = {
    "invalid": "Invalid Spectrum",
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
    cosmic_ray_replaced: int = 0

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
    """为第二阶段同前缀类内相似性打分。"""
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
        from raman.audit.stage import classify_invalid

        score_raw_and_basic(records, cfg, audit_cfg)
        classify_invalid(records, audit_cfg)
    elif stage == "class-similarity":
        from raman.audit.stage import classify_class_similarity

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
        "cosmic_ray_replaced": record.cosmic_ray_replaced,
        "risk_score": fmt_value(record.risk_score, 3),
    }


def stage_title(stage: str) -> str:
    """返回阶段展示名称。"""
    stage = validate_stage(stage)
    if stage == "class-similarity":
        return "Class Similarity"
    return "Invalid Spectrum"
