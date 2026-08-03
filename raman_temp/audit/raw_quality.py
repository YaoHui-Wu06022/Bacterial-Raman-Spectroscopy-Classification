"""Stage1 原始质量审核与运行池内候选归档。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from raman_temp.audit.config import AuditConfig, resolve_audit_config
from raman_temp.audit.io import read_raw_spectrum
from raman_temp.audit.records import RawRecord, StageResult
from raman_temp.audit.reports import build_stage_dir, write_csv, write_json
from raman_temp.audit.workspace import move_stage_candidate
from raman_temp.common.naming import parse_folder_prefix


RAW_FIELDS = (
    "state",
    "reasons",
    "rel_path",
    "source_rel_path",
    "genus",
    "folder",
    "prefix",
    "points",
    "coverage",
    "malformed_lines",
    "longest_flat_points",
    "saturation_points",
    "noise_ratio",
)


def build_raw_records(input_dir: Path) -> list[RawRecord]:
    """遍历运行池 `init/` 并建立带属、文件夹和相对路径信息的记录。"""
    records = []
    for path in sorted(input_dir.rglob("*.arc_data")):
        relative = path.relative_to(input_dir)
        if len(relative.parts) < 3:
            continue
        genus, folder = relative.parts[:2]
        records.append(
            RawRecord(
                path=path,
                rel_path=relative.as_posix(),
                genus=genus,
                folder=folder,
                prefix=parse_folder_prefix(folder),
            )
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


def score_raw_record(record: RawRecord, config: AuditConfig) -> None:
    """计算原始谱完整性、平坦段、重复值和噪声指标并更新记录状态。"""
    try:
        wavenumbers, intensities, malformed_lines = read_raw_spectrum(record.path)
    except OSError:
        record.state = "unscorable"
        record.reasons = ("read_failed",)
        return

    record.wavenumbers = wavenumbers
    record.intensities = intensities
    record.points = int(wavenumbers.size)
    record.malformed_lines = malformed_lines
    if malformed_lines:
        record.state = "unscorable"
        record.reasons = ("malformed_rows",)
        return
    if wavenumbers.size < config.raw_min_points:
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

    spectrum = config.input
    low = max(float(wavenumbers.min()), spectrum.cut_min)
    high = min(float(wavenumbers.max()), spectrum.cut_max)
    record.coverage = max(high - low, 0.0) / max(spectrum.cut_max - spectrum.cut_min, 1e-8)
    spread = float(np.quantile(intensities, 0.95) - np.quantile(intensities, 0.05))
    if spread <= 1e-10:
        record.longest_flat_points = int(intensities.size)
        record.saturation_points = int(intensities.size)
        record.noise_ratio = 0.0
        return

    if intensities.size >= config.raw_flat_window:
        flat = np.asarray(
            [
                np.ptp(intensities[index : index + config.raw_flat_window]) <= spread * 0.01
                for index in range(intensities.size - config.raw_flat_window + 1)
            ],
            dtype=bool,
        )
        record.longest_flat_points = (
            find_longest_true_run(flat) + config.raw_flat_window - 1 if flat.any() else 0
        )
    rounded = np.round(intensities, 6)
    record.saturation_points = find_longest_true_run(np.r_[False, np.diff(rounded) == 0])
    difference = np.diff(intensities)
    record.noise_ratio = float(
        np.median(np.abs(difference - np.median(difference))) / spread
    )


def build_noise_limit(records: list[RawRecord]) -> float:
    """从可评分谱的噪声分布计算 MAD 自适应上限。"""
    values = np.asarray(
        [record.noise_ratio for record in records if record.state == "keep" and np.isfinite(record.noise_ratio)],
        dtype=float,
    )
    if not values.size:
        return math.inf
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + max(8.0 * 1.4826 * mad, median * 2.0, 1e-5)


def mark_raw_candidates(records: list[RawRecord], config: AuditConfig) -> float:
    """依据固定质量规则和群体噪声上限标记 Stage1 candidate。"""
    noise_limit = build_noise_limit(records)
    for record in records:
        if record.state != "keep":
            continue
        reasons = []
        if record.coverage < config.raw_coverage_min:
            reasons.append("insufficient_wavenumber_coverage")
        if record.longest_flat_points >= config.raw_long_flat_points:
            reasons.append("long_flat_raw_region")
        if record.saturation_points >= config.raw_saturation_points:
            reasons.append("repeated_raw_values")
        if record.noise_ratio > noise_limit:
            reasons.append("extreme_raw_noise")
        if reasons:
            record.state = "candidate"
            record.reasons = tuple(reasons)
    return noise_limit


def build_raw_rows(records: list[RawRecord]) -> list[dict[str, object]]:
    """将 Stage1 记录转换为字段稳定的 CSV 行。"""
    return [
        {
            "state": record.state,
            "reasons": ";".join(record.reasons),
            "rel_path": record.rel_path,
            "source_rel_path": record.source_rel_path,
            "genus": record.genus,
            "folder": record.folder,
            "prefix": record.prefix,
            "points": record.points,
            "coverage": f"{record.coverage:.6f}" if np.isfinite(record.coverage) else "",
            "malformed_lines": record.malformed_lines,
            "longest_flat_points": record.longest_flat_points,
            "saturation_points": record.saturation_points,
            "noise_ratio": f"{record.noise_ratio:.8f}" if np.isfinite(record.noise_ratio) else "",
        }
        for record in records
    ]


def run_stage1(
    input_dir: Path,
    work_dir: Path,
    run_dir: Path,
    config: AuditConfig | None = None,
    move_enable: bool = True,
) -> StageResult:
    """执行 Stage1，并仅在运行池内归档候选与写出审核记录。"""
    audit_config = resolve_audit_config(config)
    records = build_raw_records(input_dir)
    for record in records:
        score_raw_record(record, audit_config)
    noise_limit = mark_raw_candidates(records, audit_config)

    stage_dir = build_stage_dir(run_dir, "stage1")
    rows = build_raw_rows(records)
    candidates = [record for record in records if record.state in {"candidate", "unscorable"}]
    write_csv(stage_dir / "stage1_raw_scores.csv", rows, RAW_FIELDS)
    write_csv(stage_dir / "stage1_candidates.csv", build_raw_rows(candidates), RAW_FIELDS)

    moved_count = 0
    if move_enable:
        for record in candidates:
            move_stage_candidate(record, work_dir, "stage1")
            moved_count += 1
    write_json(
        stage_dir / "run.json",
        {
            "stage": "stage1",
            "records": len(records),
            "candidates": len(candidates),
            "moved": moved_count,
            "noise_limit": noise_limit,
        },
    )
    return StageResult(stage_dir, len(records), len(candidates), moved_count)
