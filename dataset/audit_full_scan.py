"""Full read-only audit for spectra under one dataset init folder.

Usage:
    python dataset/audit_full_scan.py 细菌
    python dataset/audit_full_scan.py 细菌 --max-remove-candidates 100

The script only writes reports and review figures. It never moves or deletes
raw .arc_data files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman.data.archive import iter_arc_dirs  # noqa: E402
from audit_single_spectra import preprocess_spectrum_for_audit, robust_scale, robust_wave_stats, spectral_corr  # noqa: E402
from raman.data.build import DEFAULT_PIPELINE_CONFIG, _cosmic_ray_enabled  # noqa: E402
from raman.data.offline import remove_cosmic_rays  # noqa: E402
from raman.data.profiles import get_dataset_dir, get_profile  # noqa: E402
from raman.data.spectrum import build_valid_mask, read_arc_data  # noqa: E402


GROUP_SCORE_THRESHOLD = 3.5
GROUP_CORR_THRESHOLD = 0.92
GROUP_POINT_Z_THRESHOLD = 8.0
GROUP_BAD_RATIO_THRESHOLD = 0.03

STEP_SMOOTH_POINTS = 21
STEP_SIDE_POINTS = 28
STEP_GAP_POINTS = 4
STEP_JUMP_Z_THRESHOLD = 8.0
STEP_LEVEL_Z_THRESHOLD = 12.0
STEP_MIN_DELTA = 0.9
STEP_OPPOSITE_WINDOW = 32
STEP_EDGE_CM = 12.0

RESIDUAL_POS_Z_THRESHOLD = 8.0
RESIDUAL_MIN_MAX_Z = 12.0
RESIDUAL_MAX_WIDTH_CM = 30.0


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="细菌", help="Dataset name or profile id.")
    parser.add_argument("--output-root", default=None, help="Override output root. Default: dataset/<name>/audit_full_scan")
    parser.add_argument("--timestamp", default=None, help="Override timestamp folder name.")
    parser.add_argument("--min-ref-files", type=int, default=20, help="Minimum same-prefix reference files.")
    parser.add_argument("--max-remove-candidates", type=int, default=100, help="Maximum high-confidence spectra to mark as remove_candidate.")
    parser.add_argument("--max-folder-candidates", type=int, default=5, help="Maximum folder-level candidates to report.")
    parser.add_argument("--max-spectrum-figures", type=int, default=0, help="Maximum spectrum review figures. 0 means all candidates, negative means none.")
    parser.add_argument("--max-folder-figures", type=int, default=0, help="Maximum folder overview figures.")
    return parser.parse_args()


def prefix_of(name: str) -> str:
    letters = []
    for char in name:
        if char.isalpha():
            letters.append(char)
        else:
            break
    return "".join(letters) or name


def robust_mad_scale(values, floor=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(floor)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= floor:
        scale = float(np.std(values))
    return max(scale, float(floor))


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float32)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def median_step_cm(wn):
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size < 2:
        return 1.0
    diffs = np.abs(np.diff(wn))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-8)]
    return float(np.median(diffs)) if diffs.size else 1.0


def region_width_cm(wn, start, end):
    if end <= start:
        return 0.0
    step = median_step_cm(wn)
    return float(abs(wn[end - 1] - wn[start]) + step)


def output_wn(cfg):
    wn = cfg.build_wn_ref()
    keep = build_valid_mask(wn, cfg.bad_bands)
    return wn[keep] if keep is not None else wn


def is_bad_band_edge(pos, bad_bands):
    for band_min, band_max in bad_bands:
        if abs(pos - band_min) <= STEP_EDGE_CM or abs(pos - band_max) <= STEP_EDGE_CM:
            return True
    return False


def detect_steps(wn, z, bad_bands):
    smooth = moving_average(z, STEP_SMOOTH_POINTS)
    diff = np.diff(smooth)
    diff_scale = robust_mad_scale(diff)
    jump_z = np.abs(diff) / diff_scale
    candidates = np.where(jump_z >= STEP_JUMP_Z_THRESHOLD)[0]

    positions = []
    edge_positions = []
    max_delta = 0.0
    max_step_z = 0.0
    for idx in candidates:
        left_start = max(0, idx - STEP_GAP_POINTS - STEP_SIDE_POINTS)
        left_end = max(0, idx - STEP_GAP_POINTS)
        right_start = min(smooth.size, idx + STEP_GAP_POINTS + 1)
        right_end = min(smooth.size, idx + STEP_GAP_POINTS + 1 + STEP_SIDE_POINTS)
        if left_end - left_start < STEP_SIDE_POINTS // 2 or right_end - right_start < STEP_SIDE_POINTS // 2:
            continue

        left_level = float(np.median(smooth[left_start:left_end]))
        right_level = float(np.median(smooth[right_start:right_end]))
        delta = right_level - left_level
        level_z = abs(delta) / diff_scale
        if abs(delta) < STEP_MIN_DELTA or level_z < STEP_LEVEL_Z_THRESHOLD:
            continue

        sign = np.sign(diff[idx])
        local_start = max(0, idx - STEP_OPPOSITE_WINDOW)
        local_end = min(diff.size, idx + STEP_OPPOSITE_WINDOW + 1)
        local_diff = diff[local_start:local_end]
        opposite = np.any((np.sign(local_diff) == -sign) & (np.abs(local_diff) >= abs(diff[idx]) * 0.6))
        if opposite:
            continue

        pos = float(wn[min(idx + 1, len(wn) - 1)])
        max_delta = max(max_delta, abs(delta))
        max_step_z = max(max_step_z, level_z)
        if is_bad_band_edge(pos, bad_bands):
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


def residual_cosmic_regions(wn, signed_z):
    mask = signed_z >= RESIDUAL_POS_Z_THRESHOLD
    regions = contiguous_regions(mask)
    count = 0
    for start, end in regions:
        if region_width_cm(wn, start, end) <= RESIDUAL_MAX_WIDTH_CM:
            count += 1
    return count


def corr_many_to_one(arr, spec):
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    return (arr @ spec / max(spec.size, 1)).astype(np.float32, copy=False)


def plot_segments_without_bad_bands(ax, wn, values, bad_bands, **kwargs):
    wn = np.asarray(wn, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    keep = np.ones_like(wn, dtype=bool)
    for band_min, band_max in bad_bands:
        keep &= ~((wn >= band_min) & (wn <= band_max))

    if wn.size >= 2:
        step = median_step_cm(wn)
        gap_breaks = np.where(np.abs(np.diff(wn)) > step * 1.8)[0]
        keep[gap_breaks + 1] = False

    label = kwargs.pop("label", None)
    labeled = False
    for start, end in contiguous_regions(keep):
        if end - start >= 2:
            line_label = label if label and not labeled else None
            ax.plot(wn[start:end], values[start:end], label=line_label, **kwargs)
            labeled = True
    for band_min, band_max in bad_bands:
        ax.axvspan(band_min, band_max, color="gray", alpha=0.14)


def fill_between_segments_without_bad_bands(ax, wn, lower, upper, bad_bands, **kwargs):
    wn = np.asarray(wn, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    keep = np.ones_like(wn, dtype=bool)
    for band_min, band_max in bad_bands:
        keep &= ~((wn >= band_min) & (wn <= band_max))
    if wn.size >= 2:
        step = median_step_cm(wn)
        gap_breaks = np.where(np.abs(np.diff(wn)) > step * 1.8)[0]
        keep[gap_breaks + 1] = False

    label = kwargs.pop("label", None)
    labeled = False
    for start, end in contiguous_regions(keep):
        if end - start >= 2:
            band_label = label if label and not labeled else None
            ax.fill_between(wn[start:end], lower[start:end], upper[start:end], label=band_label, **kwargs)
            labeled = True
    for band_min, band_max in bad_bands:
        ax.axvspan(band_min, band_max, color="gray", alpha=0.14)


def load_records(profile, cfg, dataset_dir, init_root):
    records = []
    folder_order = []
    wn_ref = cfg.build_wn_ref()
    for root, arc_files in iter_arc_dirs(init_root):
        rel_group = root.relative_to(init_root)
        if len(rel_group.parts) < 2:
            genus = rel_group.parts[0] if rel_group.parts else "."
            folder = root.name
        else:
            genus, folder = rel_group.parts[0], rel_group.parts[1]
        group = rel_group.as_posix()
        folder_order.append(group)
        print(f"[Preprocess] {group}: {len(arc_files)} files")

        for filename in arc_files:
            path = root / filename
            payload = preprocess_spectrum_for_audit(path, profile, cfg, wn_ref=wn_ref, include_raw=False)
            record = SpectrumRecord(
                path=path,
                rel_path=path.relative_to(dataset_dir).as_posix(),
                group=group,
                genus=genus,
                folder=folder,
                file=filename,
                skip_reason=payload.get("skip_reason", ""),
            )
            if not record.skip_reason:
                stats = payload["cosmic_stats"]
                record.z = np.asarray(payload["z"], dtype=np.float32)
                record.sp = np.asarray(payload["sp"], dtype=np.float32)
                record.cosmic_total = int(stats)
                record.cosmic_narrow = int(getattr(stats, "narrow", 0))
                record.cosmic_peak = int(getattr(stats, "peak", 0))
                record.cosmic_residual = int(getattr(stats, "residual", 0))
            records.append(record)
    return records, folder_order


def score_groups(records, cfg):
    group_stats = {}
    for group in sorted({record.group for record in records}):
        group_records = [record for record in records if record.group == group]
        valid = [record for record in group_records if record.z is not None]
        if len(valid) < 5:
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
            record.bad_ratio_group = float(np.mean(abs_z > GROUP_POINT_Z_THRESHOLD))
            record.max_pos_z_group = float(np.max(signed_z))
            record.positive_bad_ratio_z8 = float(np.mean(signed_z > GROUP_POINT_Z_THRESHOLD))
            record.residual_cosmic_regions = residual_cosmic_regions(output_wn(cfg), signed_z)
            record.roughness = float(roughness)
            record.roughness_z = float((roughness - rough_center) / rough_scale)
            step = detect_steps(output_wn(cfg), record.z, cfg.bad_bands)
            record.step_count = step["step_count"]
            record.bad_band_edge_step_count = step["bad_band_edge_step_count"]
            record.max_step_delta = step["max_step_delta"]
            record.max_step_z = step["max_step_z"]
            record.step_positions = step["step_positions"]

        group_stats[group] = {
            "center": center,
            "q10": q10,
            "q90": q90,
            "wave_scale": wave_scale,
            "valid": len(valid),
        }
    return group_stats


def build_reference_index(records):
    by_genus = {}
    by_genus_prefix = {}
    for record in records:
        if record.z is None:
            continue
        by_genus.setdefault(record.genus, []).append(record)
        by_genus_prefix.setdefault((record.genus, prefix_of(record.folder)), []).append(record)
    return by_genus, by_genus_prefix


def reference_for_group(group_records, by_genus, by_genus_prefix, min_ref_files):
    first = group_records[0]
    same_prefix = [record for record in by_genus_prefix.get((first.genus, prefix_of(first.folder)), []) if record.group != first.group]
    if len(same_prefix) >= min_ref_files:
        return same_prefix
    return [record for record in by_genus.get(first.genus, []) if record.group != first.group]


def score_references(records, min_ref_files):
    by_genus, by_genus_prefix = build_reference_index(records)
    ref_stats_by_group = {}
    for group in sorted({record.group for record in records}):
        group_records = [record for record in records if record.group == group]
        valid = [record for record in group_records if record.z is not None]
        if not valid:
            ref_stats_by_group[group] = None
            continue
        refs = reference_for_group(valid, by_genus, by_genus_prefix, min_ref_files)
        if len(refs) < 5:
            ref_stats_by_group[group] = None
            continue

        ref_arr = np.vstack([record.z for record in refs])
        ref_median, ref_scale = robust_wave_stats(ref_arr, min_scale=0.05, floor_fraction=0.25)
        ref_corrs = corr_many_to_one(ref_arr, ref_median)
        ref_bad6 = np.mean(np.abs((ref_arr - ref_median) / ref_scale) > 6.0, axis=1)
        ref_rmse = np.sqrt(np.mean((ref_arr - ref_median) ** 2, axis=1))
        ref_cosmic = np.asarray([record.cosmic_total for record in refs], dtype=np.float32)

        corr_threshold = max(0.80, float(np.percentile(ref_corrs, 1) - 0.05))
        nearest_threshold = max(0.86, float(np.percentile(ref_corrs, 5) - 0.03))
        bad_threshold = max(0.04, float(np.percentile(ref_bad6, 99) + 0.02))
        rmse_threshold = max(0.75, float(np.percentile(ref_rmse, 99) * 1.35))
        cosmic_threshold = max(80, int(np.percentile(ref_cosmic, 99) + 30)) if ref_cosmic.size else 120

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
            "corr_threshold": corr_threshold,
            "nearest_threshold": nearest_threshold,
            "bad_threshold": bad_threshold,
            "rmse_threshold": rmse_threshold,
            "cosmic_threshold": cosmic_threshold,
            "folder_corr_ref": spectral_corr(folder_median, ref_median),
        }
    return ref_stats_by_group


def classify_records(records, ref_stats_by_group):
    for record in records:
        if record.z is None:
            record.decision = "skip"
            record.reasons = (record.skip_reason or "preprocess_failed",)
            continue

        reasons = []
        ref_stats = ref_stats_by_group.get(record.group)

        if record.group_score >= GROUP_SCORE_THRESHOLD:
            reasons.append("group_shape_score")
        if record.corr_group <= GROUP_CORR_THRESHOLD:
            reasons.append("low_group_corr")
        if record.bad_ratio_group >= GROUP_BAD_RATIO_THRESHOLD:
            reasons.append("group_point_outlier")
        if record.roughness_z >= GROUP_SCORE_THRESHOLD and record.roughness >= 0.12:
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
            and record.max_pos_z_group >= RESIDUAL_MIN_MAX_Z
            and record.positive_bad_ratio_z8 > 0
        ):
            reasons.append("residual_cosmic_like")

        ref_evidence = sum(
            reason in reasons
            for reason in ("low_ref_similarity", "many_ref_point_outliers", "high_rmse_to_ref")
        )
        group_evidence = sum(
            reason in reasons
            for reason in ("group_shape_score", "low_group_corr", "group_point_outlier")
        )
        strong_noise = "rough_noise_outlier" in reasons and (
            "group_point_outlier" in reasons
            or "many_ref_point_outliers" in reasons
            or ("low_ref_similarity" in reasons and record.corr_group <= 0.65)
        )
        strong_residual_cosmic = "residual_cosmic_like" in reasons and (
            "group_point_outlier" in reasons or "many_ref_point_outliers" in reasons
        ) and (
            "low_group_corr" in reasons or "low_ref_similarity" in reasons
        )
        strong_ref_outlier = ref_evidence >= 3 and group_evidence >= 1
        strong_group_outlier = group_evidence >= 3 and ref_evidence >= 1
        review_residual_cosmic = "residual_cosmic_like" in reasons and (
            record.max_pos_z_group >= 16.0
            or "group_point_outlier" in reasons
            or "many_ref_point_outliers" in reasons
        )
        review_group_outlier = (
            "group_point_outlier" in reasons
            or ("group_shape_score" in reasons and "low_group_corr" in reasons and record.corr_group <= 0.70)
        )
        review_ref_outlier = ref_evidence >= 2 and (
            "many_ref_point_outliers" in reasons or "high_rmse_to_ref" in reasons
        )
        review_noise = "rough_noise_outlier" in reasons and (
            "group_point_outlier" in reasons
            or "many_ref_point_outliers" in reasons
            or ("low_ref_similarity" in reasons and record.corr_group <= 0.75)
        )

        if "step_like_spectrum" in reasons or strong_noise or strong_residual_cosmic or strong_ref_outlier or strong_group_outlier:
            record.decision = "remove_candidate"
        elif "bad_band_edge_step" in reasons or review_residual_cosmic or review_group_outlier or review_ref_outlier or review_noise:
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


def build_folder_records(records, group_stats, ref_stats_by_group):
    folders = []
    for group in sorted({record.group for record in records}):
        group_records = [record for record in records if record.group == group]
        valid = [record for record in group_records if record.z is not None]
        if group_records:
            genus = group_records[0].genus
            folder = group_records[0].folder
        else:
            genus = folder = ""

        candidate_records = [record for record in valid if record.decision in {"remove_candidate", "review_candidate"}]
        remove_records = [record for record in valid if record.decision == "remove_candidate"]
        group_flagged = [
            record
            for record in valid
            if (
                record.group_score >= GROUP_SCORE_THRESHOLD
                or record.corr_group <= GROUP_CORR_THRESHOLD
                or record.bad_ratio_group >= GROUP_BAD_RATIO_THRESHOLD
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
        if folder_record.valid < 5:
            reasons.append("too_few_valid_spectra")
        if np.isfinite(folder_record.folder_corr_ref) and folder_record.folder_corr_ref < 0.75:
            reasons.append("folder_far_from_references")
        if folder_record.candidate_fraction >= 0.20:
            reasons.append("many_candidate_spectra")
        if folder_record.ref_remove_fraction >= 0.15:
            reasons.append("many_ref_outliers")
        if folder_record.step_spectra / max(folder_record.valid, 1) >= 0.15:
            reasons.append("many_step_like_spectra")

        if (
            folder_record.ref_remove_fraction >= 0.30
            or folder_record.candidate_fraction >= 0.45
            or folder_record.step_spectra / max(folder_record.valid, 1) >= 0.25
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


def record_to_row(record):
    return {
        "decision": record.decision,
        "reasons": ";".join(record.reasons),
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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cosmic_clean_for_plot(wn, sp, profile, cfg):
    if not _cosmic_ray_enabled(profile, cfg):
        return np.asarray(sp, dtype=np.float32)
    cleaned, _ = remove_cosmic_rays(
        sp,
        window_cm=cfg.cosmic_ray_narrow_window_cm,
        threshold=cfg.cosmic_ray_threshold,
        max_iter=cfg.cosmic_ray_max_iter,
        valid_mask=None,
        wn=wn,
        peak_prominence_z=cfg.cosmic_ray_peak_prominence_z,
        peak_width_max_cm=cfg.cosmic_ray_peak_width_max_cm,
        peak_ratio_z_per_cm=cfg.cosmic_ray_peak_ratio_z_per_cm,
        peak_pad_cm=cfg.cosmic_ray_peak_pad_cm,
        peak_rel_height=cfg.cosmic_ray_peak_rel_height,
        residual_threshold_z=cfg.cosmic_ray_residual_threshold_z,
        residual_max_points_fraction=cfg.cosmic_ray_residual_max_points_fraction,
    )
    return cleaned


def plot_spectrum_candidate(record, out_path, profile, cfg, group_stats, ref_stats):
    raw_wn, raw_sp = read_arc_data(record.path)
    raw_cosmic = cosmic_clean_for_plot(raw_wn, raw_sp, profile, cfg)
    wn = output_wn(cfg)
    gstats = group_stats.get(record.group)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
    plot_segments_without_bad_bands(axes[0], raw_wn, raw_sp, cfg.bad_bands, color="0.60", linewidth=0.9, label="raw")
    plot_segments_without_bad_bands(axes[0], raw_wn, raw_cosmic, cfg.bad_bands, color="C0", linewidth=0.9, label="cosmic cleaned")
    axes[0].set_title("Raw / cosmic cleaned")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="best")

    fill_between_segments_without_bad_bands(axes[1], wn, gstats["q10"], gstats["q90"], cfg.bad_bands, color="C0", alpha=0.16, label="group q10-q90")
    plot_segments_without_bad_bands(axes[1], wn, gstats["center"], cfg.bad_bands, color="C0", linewidth=1.4, label="group median")
    if ref_stats is not None:
        plot_segments_without_bad_bands(axes[1], wn, ref_stats["ref_median"], cfg.bad_bands, color="C2", linewidth=1.1, label="ref median")
    plot_segments_without_bad_bands(axes[1], wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample SNV")
    axes[1].set_title("After full preprocessing")
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    group_z = (record.z - gstats["center"]) / gstats["wave_scale"]
    plot_segments_without_bad_bands(axes[2], wn, group_z, cfg.bad_bands, color="C4", linewidth=0.9, label="group robust z")
    if ref_stats is not None:
        ref_z = (record.z - ref_stats["ref_median"]) / ref_stats["ref_scale"]
        plot_segments_without_bad_bands(axes[2], wn, ref_z, cfg.bad_bands, color="C1", linewidth=0.8, alpha=0.75, label="ref robust z")
    axes[2].axhline(GROUP_POINT_Z_THRESHOLD, color="C3", linestyle="--", linewidth=0.8)
    axes[2].axhline(-GROUP_POINT_Z_THRESHOLD, color="C3", linestyle="--", linewidth=0.8)
    for pos in record.step_positions:
        axes[2].axvline(pos, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_title("Robust z-score and detected steps")
    axes[2].set_ylabel("z")
    axes[2].legend(loc="best")

    axes[3].axis("off")
    text = (
        f"Decision: {record.decision}\n"
        f"Reasons: {'; '.join(record.reasons)}\n"
        f"corr_group={record.corr_group:.3f}, bad_ratio_group={record.bad_ratio_group:.3f}\n"
        f"corr_ref={record.corr_ref:.3f}, nearest_ref_corr={record.nearest_ref_corr:.3f}, bad_ratio_z6={record.bad_ratio_z6:.3f}\n"
        f"step_count={record.step_count}, bad_band_edge_step={record.bad_band_edge_step_count}, residual_regions={record.residual_cosmic_regions}\n"
        f"cosmic_total={record.cosmic_total}"
    )
    axes[3].text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(record.rel_path)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_folder_candidate(folder_record, records, out_path, cfg, group_stats, ref_stats):
    valid = [record for record in records if record.group == folder_record.group and record.z is not None]
    if not valid:
        return
    wn = output_wn(cfg)
    gstats = group_stats.get(folder_record.group)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for record in valid:
        color = "C3" if record.decision == "remove_candidate" else ("C1" if record.decision == "review_candidate" else "0.75")
        alpha = 0.75 if record.decision != "keep" else 0.25
        plot_segments_without_bad_bands(axes[0], wn, record.z, cfg.bad_bands, color=color, alpha=alpha, linewidth=0.75)
    plot_segments_without_bad_bands(axes[0], wn, gstats["center"], cfg.bad_bands, color="C0", linewidth=1.8, label="group median")
    if ref_stats is not None:
        plot_segments_without_bad_bands(axes[0], wn, ref_stats["ref_median"], cfg.bad_bands, color="C2", linewidth=1.4, label="ref median")
    axes[0].set_ylabel("SNV intensity")
    axes[0].set_title("Folder spectra overview")
    axes[0].legend(loc="best")

    axes[1].axis("off")
    text = (
        f"Decision: {folder_record.decision}\n"
        f"Reasons: {'; '.join(folder_record.reasons)}\n"
        f"Files/valid/skipped: {folder_record.files}/{folder_record.valid}/{folder_record.skipped}\n"
        f"Remove/review candidates: {folder_record.remove_candidates}/{folder_record.review_candidates}\n"
        f"candidate_fraction={folder_record.candidate_fraction:.3f}, ref_remove_fraction={folder_record.ref_remove_fraction:.3f}\n"
        f"folder_corr_ref={folder_record.folder_corr_ref:.3f}, p95_cosmic={folder_record.p95_cosmic_total:.1f}, max_cosmic={folder_record.max_cosmic_total}"
    )
    axes[1].text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle(folder_record.group)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_figures(out_dir, records, folders, profile, cfg, group_stats, ref_stats_by_group, max_spectrum_figures, max_folder_figures):
    candidates = [record for record in records if record.decision in {"remove_candidate", "review_candidate"} and record.z is not None]
    candidates = sorted(candidates, key=lambda item: (item.decision != "remove_candidate", -item.risk_score, item.rel_path))
    if max_spectrum_figures < 0:
        spectrum_to_plot = []
    elif max_spectrum_figures == 0:
        spectrum_to_plot = candidates
    else:
        spectrum_to_plot = candidates[:max_spectrum_figures]
    for record in spectrum_to_plot:
        out_path = out_dir / "figures" / "spectra" / record.genus / record.folder / f"{Path(record.file).stem}.png"
        plot_spectrum_candidate(record, out_path, profile, cfg, group_stats, ref_stats_by_group.get(record.group))

    folder_candidates = top_folder_candidates(folders, max_folder_figures)
    for folder in folder_candidates:
        out_path = out_dir / "figures" / "folders" / folder.genus / f"{folder.folder}.png"
        plot_folder_candidate(folder, records, out_path, cfg, group_stats, ref_stats_by_group.get(folder.group))
    return len(spectrum_to_plot), len(folder_candidates)


def parameter_advice(records, folders, cfg):
    valid = [record for record in records if record.z is not None]
    residual_records = [record for record in valid if "residual_cosmic_like" in record.reasons]
    residual_groups = len({record.group for record in residual_records})
    step_records = [record for record in valid if "step_like_spectrum" in record.reasons or "bad_band_edge_step" in record.reasons]
    high_cosmic = [record for record in valid if "excessive_cosmic_cleanup" in record.reasons]

    if not valid:
        return "没有有效光谱，无法评估宇宙射线参数。"

    residual_ratio = len(residual_records) / len(valid)
    high_cosmic_ratio = len(high_cosmic) / len(valid)
    if residual_ratio > 0.03 and residual_groups > 20:
        return (
            "残留正向异常在多个文件夹中系统性出现，后续可以考虑小幅提高 "
            "`COSMIC_RAY_PEAK_WIDTH_MAX_CM` 或降低 residual 阈值；本轮仍不自动改参数。"
        )
    if high_cosmic_ratio > 0.05:
        return (
            "过量宇宙射线替换样本比例偏高，但需要先看候选图确认是否是噪声态/阶梯谱；"
            "当前更建议先剔除异常谱，不直接改参数。"
        )
    if step_records and len(step_records) >= len(residual_records):
        return "异常主要表现为阶梯状或噪声态光谱，优先作为数据质量问题复核移除；不建议调整宇宙射线参数。"
    return "未看到必须调参的系统性证据，当前宇宙射线参数建议保持不变。"


def write_summary(out_dir, records, folders, reported_folders, cfg, dataset_dir, init_root, fig_counts):
    valid = [record for record in records if record.z is not None]
    skipped = [record for record in records if record.z is None]
    remove_records = [record for record in valid if record.decision == "remove_candidate"]
    review_records = [record for record in valid if record.decision == "review_candidate"]
    remove_folders = [folder for folder in reported_folders if folder.decision == "remove_candidate"]
    review_folders = [folder for folder in reported_folders if folder.decision == "review_candidate"]
    cosmic_totals = np.asarray([record.cosmic_total for record in valid], dtype=np.float32)

    lines = [
        "# dataset/细菌 全库异常谱复查报告",
        "",
        "## 总体",
        "",
        f"- 数据目录：`{init_root}`",
        f"- 输出目录：`{out_dir}`",
        f"- 小文件夹数：{len(folders)}",
        f"- 总光谱数：{len(records)}",
        f"- 有效光谱数：{len(valid)}",
        f"- 跳过光谱数：{len(skipped)}",
        f"- 建议移除候选谱：{len(remove_records)}",
        f"- 仅复核候选谱：{len(review_records)}",
        f"- 输出文件夹候选：{len(remove_folders) + len(review_folders)}",
        f"- 已输出候选谱图：{fig_counts[0]}",
        f"- 已输出文件夹图：{fig_counts[1]}",
        "- 本报告只读生成，没有移动或删除任何 `.arc_data`。",
        "",
        "## 宇宙射线统计",
        "",
        f"- 当前参数：`peak_width_max_cm={cfg.cosmic_ray_peak_width_max_cm}`, `peak_prominence_z={cfg.cosmic_ray_peak_prominence_z}`, `residual_threshold_z={cfg.cosmic_ray_residual_threshold_z}`",
    ]
    if cosmic_totals.size:
        lines.extend(
            [
                f"- cosmic_total：中位数 {np.median(cosmic_totals):.1f}，p90 {np.quantile(cosmic_totals, 0.90):.1f}，p95 {np.quantile(cosmic_totals, 0.95):.1f}，最大 {np.max(cosmic_totals):.0f}",
                f"- residual_cosmic_like 候选：{sum('residual_cosmic_like' in record.reasons for record in valid)}",
                f"- step_like / bad_band_edge_step 候选：{sum(('step_like_spectrum' in record.reasons or 'bad_band_edge_step' in record.reasons) for record in valid)}",
            ]
        )
    lines.extend(["", "## 参数建议", "", f"- {parameter_advice(records, folders, cfg)}", ""])

    lines.extend(["## 建议移除候选谱", ""])
    if remove_records:
        for record in sorted(remove_records, key=lambda item: (-item.risk_score, item.rel_path))[:80]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(record.reasons)}")
    else:
        lines.append("- 暂无。")

    lines.extend(["", "## 仅复核候选谱", ""])
    if review_records:
        for record in sorted(review_records, key=lambda item: (-item.risk_score, item.rel_path))[:80]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(record.reasons)}")
    else:
        lines.append("- 暂无。")

    lines.extend(["", "## 文件夹候选", ""])
    folder_candidates = remove_folders + review_folders
    if folder_candidates:
        for folder in sorted(folder_candidates, key=lambda item: (item.decision != "remove_candidate", -item.candidate_fraction, item.group)):
            lines.append(
                f"- `{folder.group}`：{folder.decision}，{'; '.join(folder.reasons)}，"
                f"候选比例 {folder.candidate_fraction:.2%}，folder_corr_ref={folder.folder_corr_ref:.3f}"
            )
    else:
        lines.append("- 暂无。")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "dataset_dir": str(dataset_dir),
        "init_root": str(init_root),
        "output_dir": str(out_dir),
        "config": asdict(cfg),
        "total_folders": len(folders),
        "total_spectra": len(records),
        "valid_spectra": len(valid),
        "skipped_spectra": len(skipped),
        "remove_candidates": len(remove_records),
        "review_candidates": len(review_records),
        "folder_candidates": len(folder_candidates),
        "figures": {"spectra": fig_counts[0], "folders": fig_counts[1]},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    profile = get_profile(args.dataset)
    cfg = DEFAULT_PIPELINE_CONFIG
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    init_root = dataset_dir / profile.root_init
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init folder: {init_root}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else dataset_dir / "audit_full_scan"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    out_dir = output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_dir}")
    print(f"Input: {init_root}")
    print(f"Output: {out_dir}")

    records, _ = load_records(profile, cfg, dataset_dir, init_root)
    group_stats = score_groups(records, cfg)
    ref_stats_by_group = score_references(records, args.min_ref_files)
    classify_records(records, ref_stats_by_group)
    cap_remove_candidates(records, args.max_remove_candidates)
    folder_records = build_folder_records(records, group_stats, ref_stats_by_group)
    reported_folder_records = top_folder_candidates(folder_records, args.max_folder_candidates)

    candidate_rows = [
        record_to_row(record)
        for record in records
        if record.decision in {"remove_candidate", "review_candidate", "skip"}
    ]
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["decision"] != "remove_candidate", row["group"], row["file"]))
    folder_rows = [folder_to_row(record) for record in reported_folder_records]
    all_rows = [record_to_row(record) for record in records]

    write_csv(out_dir / "spectrum_candidates.csv", candidate_rows)
    write_csv(out_dir / "folder_candidates.csv", folder_rows)
    write_csv(out_dir / "all_spectra_scores.csv", all_rows)

    fig_counts = write_figures(
        out_dir,
        records,
        reported_folder_records,
        profile,
        cfg,
        group_stats,
        ref_stats_by_group,
        args.max_spectrum_figures,
        args.max_folder_figures,
    )
    write_summary(out_dir, records, folder_records, reported_folder_records, cfg, dataset_dir, init_root, fig_counts)

    print("\nFull scan finished:")
    print(f"- Summary: {out_dir / 'summary.md'}")
    print(f"- Spectrum candidates: {out_dir / 'spectrum_candidates.csv'}")
    print(f"- Folder candidates: {out_dir / 'folder_candidates.csv'}")
    print(f"- All scores: {out_dir / 'all_spectra_scores.csv'}")
    print(f"- Figures: {out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
