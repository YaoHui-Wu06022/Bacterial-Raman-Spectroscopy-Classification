"""共享 Stage 1 原始质量审核。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ramanv2.audit.records import CleanRecord
from ramanv2.common.arc_data import read_raw_arc_data


@dataclass(frozen=True)
class RawQualityConfig:
    """定义原始质量审核的固定阈值。"""

    min_points: int = 20
    coverage_min: float = 0.98
    flat_window: int = 40
    long_flat_points: int = 100
    saturation_points: int = 25


def build_clean_records(input_dir: Path, folder_depth: int) -> list[CleanRecord]:
    """按固定目录深度收集待清洗的原始谱，不推断任何类别标签。"""
    records = []
    for path in sorted(input_dir.rglob("*.arc_data")):
        relative = path.relative_to(input_dir)
        if len(relative.parts) != folder_depth + 1:
            continue
        group = "/".join(relative.parts[: folder_depth - 1])
        records.append(
            CleanRecord(path, relative.as_posix(), group, relative.parts[folder_depth - 1])
        )
    return records


def find_longest_true_run(values: np.ndarray) -> int:
    """返回布尔序列中最长连续真值段的长度。"""
    best = 0
    length = 0
    for value in values:
        if value:
            length += 1
            best = max(best, length)
        else:
            length = 0
    return best


def compute_raw_quality(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    cut_min: float,
    cut_max: float,
    config: RawQualityConfig,
) -> tuple[float, int, int, float]:
    """计算已通过结构校验的原始谱覆盖、平坦、重复和相对噪声。"""
    low = max(float(wavenumbers.min()), cut_min)
    high = min(float(wavenumbers.max()), cut_max)
    coverage = max(high - low, 0.0) / max(cut_max - cut_min, 1e-8)
    spread = float(np.quantile(intensities, 0.95) - np.quantile(intensities, 0.05))
    if spread <= 1e-10:
        return coverage, int(intensities.size), int(intensities.size), 0.0
    longest_flat_points = 0
    if intensities.size >= config.flat_window:
        flat = np.asarray(
            [
                np.ptp(intensities[index : index + config.flat_window]) <= spread * 0.01
                for index in range(intensities.size - config.flat_window + 1)
            ],
            dtype=bool,
        )
        if flat.any():
            longest_flat_points = find_longest_true_run(flat) + config.flat_window - 1
    rounded = np.round(intensities, 6)
    saturation_points = find_longest_true_run(np.r_[False, np.diff(rounded) == 0])
    difference = np.diff(intensities)
    noise_ratio = float(np.median(np.abs(difference - np.median(difference))) / spread)
    return coverage, longest_flat_points, saturation_points, noise_ratio


def score_raw_record(record: CleanRecord, input_config, config: RawQualityConfig) -> None:
    """校验并计算一条原始谱的 Stage 1 指标。"""
    try:
        wavenumbers, intensities, malformed_lines = read_raw_arc_data(record.path)
    except OSError:
        record.state = "unscorable"
        record.reasons = ("read_failed",)
        return
    record.points = int(wavenumbers.size)
    record.malformed_lines = malformed_lines
    if malformed_lines:
        record.state = "unscorable"
        record.reasons = ("malformed_rows",)
        return
    if wavenumbers.size < config.min_points:
        record.state = "unscorable"
        record.reasons = ("too_few_points",)
        return
    if not (np.isfinite(wavenumbers).all() and np.isfinite(intensities).all()):
        record.state = "unscorable"
        record.reasons = ("non_finite_values",)
        return
    if np.any(np.diff(wavenumbers) <= 0):
        record.state = "unscorable"
        record.reasons = ("wavenumber_not_strictly_increasing",)
        return
    (
        record.coverage,
        record.longest_flat_points,
        record.saturation_points,
        record.noise_ratio,
    ) = compute_raw_quality(
        wavenumbers,
        intensities,
        input_config.cut_min,
        input_config.cut_max,
        config,
    )


def compute_noise_limit(records: list[CleanRecord]) -> float:
    """根据可评分谱的 MAD 计算一批原始谱的相对噪声上限。"""
    values = np.asarray(
        [record.noise_ratio for record in records if record.state == "keep"],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if not values.size:
        return math.inf
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + max(8.0 * 1.4826 * mad, median * 2.0, 1e-5)


def mark_raw_candidates(records: list[CleanRecord], config: RawQualityConfig) -> float:
    """按固定质量阈值和批内噪声上限标记 Stage 1 候选。"""
    noise_limit = compute_noise_limit(records)
    for record in records:
        if record.state != "keep":
            continue
        reasons = []
        if record.coverage < config.coverage_min:
            reasons.append("insufficient_wavenumber_coverage")
        if record.longest_flat_points >= config.long_flat_points:
            reasons.append("long_flat_raw_region")
        if record.saturation_points >= config.saturation_points:
            reasons.append("repeated_raw_values")
        if record.noise_ratio > noise_limit:
            reasons.append("extreme_raw_noise")
        if reasons:
            record.state = "candidate"
            record.reasons = tuple(reasons)
    return noise_limit
