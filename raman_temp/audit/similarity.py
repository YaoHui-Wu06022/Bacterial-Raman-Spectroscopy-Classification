"""Stage2 与 Stage3 共用的近邻相似性评分。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from raman_temp.audit.config import AuditConfig, resolve_audit_config
from raman_temp.audit.records import SimilarityRecord, StageResult
from raman_temp.audit.preprocess import preprocess_audit_spectrum
from raman_temp.audit.raw_quality import build_raw_records
from raman_temp.audit.reports import build_stage_dir, write_csv, write_json
from raman_temp.audit.test_sync import parse_audit_source_folder
from raman_temp.audit.workspace import move_stage_candidate
from raman_temp.spectra.axis import build_wn_ref


SIMILARITY_FIELDS = (
    "state",
    "reasons",
    "rel_path",
    "source_rel_path",
    "genus",
    "folder",
    "prefix",
    "reference_count",
    "neighbor_count",
    "neighbor_corr",
    "rmse",
)


def calculate_mad_limit(values: np.ndarray, direction: str) -> float:
    """按缩放 MAD 为一组相似性指标计算单侧阈值。"""
    median = float(np.median(values))
    scale = max(float(np.median(np.abs(values - median))) * 1.4826, 1e-6)
    return median - 3.5 * scale if direction == "low" else median + 3.5 * scale


def score_similarity_group(records: list[SimilarityRecord], config: AuditConfig | None = None) -> None:
    """在一个近邻组内计算相关性、RMSE、MAD 阈值和 candidate 状态。"""
    audit_config = resolve_audit_config(config)
    count = len(records)
    if count - 1 < audit_config.similarity_min_references:
        for record in records:
            record.state = "insufficient_reference"
        return
    spectra = np.stack([record.spectrum for record in records])
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    correlation = (centered @ centered.T) / np.maximum(np.outer(norms, norms), 1e-8)
    neighbor_count = max(3, int(math.sqrt(count - 1)))
    for index, record in enumerate(records):
        order = np.argsort(-correlation[index])
        neighbors = [value for value in order if value != index][:neighbor_count]
        reference = np.median(spectra[neighbors], axis=0)
        record.reference_count = count - 1
        record.neighbor_count = len(neighbors)
        record.neighbor_corr = float(np.median(correlation[index, neighbors]))
        record.rmse = float(np.sqrt(np.mean((spectra[index] - reference) ** 2)))
    corr_limit = calculate_mad_limit(
        np.asarray([record.neighbor_corr for record in records]), "low"
    )
    rmse_limit = calculate_mad_limit(
        np.asarray([record.rmse for record in records]), "high"
    )
    for record in records:
        if record.neighbor_corr < corr_limit and record.rmse > rmse_limit:
            record.state = "candidate"
            record.reasons = ("low_neighbor_agreement", "high_neighbor_rmse")


def build_similarity_rows(records: list[SimilarityRecord]) -> list[dict[str, object]]:
    """将近邻评分记录转换为 Stage2 和 Stage3 共用的 CSV 行。"""
    return [
        {
            "state": record.state,
            "reasons": ";".join(record.reasons),
            "rel_path": record.raw.rel_path,
            "source_rel_path": record.raw.source_rel_path,
            "genus": record.raw.genus,
            "folder": record.raw.folder,
            "prefix": record.raw.prefix,
            "reference_count": record.reference_count,
            "neighbor_count": record.neighbor_count,
            "neighbor_corr": f"{record.neighbor_corr:.6f}" if np.isfinite(record.neighbor_corr) else "",
            "rmse": f"{record.rmse:.6f}" if np.isfinite(record.rmse) else "",
        }
        for record in records
    ]


def build_pool_records(input_dir: Path):
    """读取运行池记录；全量测试菌存在时排除重复的 `*t` 副本。"""
    records = build_raw_records(input_dir)
    has_test_pool = any(parse_audit_source_folder(record.folder) is not None for record in records)
    if not has_test_pool:
        return records
    return [
        record
        for record in records
        if parse_audit_source_folder(record.folder) is not None or not record.folder.lower().endswith("t")
    ]


def build_similarity_records(
    input_dir: Path,
    profile,
    unresolved_folders: set[tuple[str, str]],
    config: AuditConfig | None = None,
) -> list[SimilarityRecord]:
    """预处理运行池光谱，并为近邻审核建立可评分记录。"""
    audit_config = resolve_audit_config(config)
    reference_axis = build_wn_ref(
        audit_config.input.cut_min,
        audit_config.input.cut_max,
        audit_config.input.target_points,
    )
    records = []
    for raw in build_pool_records(input_dir):
        if (raw.genus, raw.folder) in unresolved_folders:
            records.append(
                SimilarityRecord(raw=raw, state="shift_unresolved", reasons=("shift_anchor_unresolved",))
            )
            continue
        spectrum = preprocess_audit_spectrum(
            raw.path,
            profile,
            audit_config,
            reference_wavenumbers=reference_axis,
            label=raw.rel_path,
        )
        if spectrum.normalized is None:
            records.append(
                SimilarityRecord(raw=raw, state="unscorable", reasons=(spectrum.skip_reason,))
            )
            continue
        records.append(SimilarityRecord(raw=raw, spectrum=spectrum.normalized))
    return records


def move_similarity_candidates(
    records: list[SimilarityRecord],
    work_dir: Path,
    stage: str,
) -> int:
    """将近邻审核 candidate 移入运行池对应阶段目录。"""
    candidates = [record for record in records if record.state == "candidate"]
    for record in candidates:
        move_stage_candidate(record.raw, work_dir, stage)
    return len(candidates)


def run_stage2(
    input_dir: Path,
    work_dir: Path,
    run_dir: Path,
    profile,
    unresolved_folders: set[tuple[str, str]] | None = None,
    config: AuditConfig | None = None,
    move_enable: bool = True,
) -> StageResult:
    """按属和类别前缀执行全种范围的近邻相似性审核。"""
    audit_config = resolve_audit_config(config)
    records = build_similarity_records(input_dir, profile, unresolved_folders or set(), audit_config)
    groups: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for record in records:
        if record.spectrum is not None:
            groups.setdefault((record.raw.genus, record.raw.prefix), []).append(record)
    for group in groups.values():
        score_similarity_group(group, audit_config)
    stage_dir = build_stage_dir(run_dir, "stage2")
    candidates = [record for record in records if record.state == "candidate"]
    write_csv(stage_dir / "stage2_similarity_scores.csv", build_similarity_rows(records), SIMILARITY_FIELDS)
    write_csv(stage_dir / "stage2_candidates.csv", build_similarity_rows(candidates), SIMILARITY_FIELDS)
    moved_count = move_similarity_candidates(records, work_dir, "stage2") if move_enable else 0
    write_json(
        stage_dir / "run.json",
        {"stage": "stage2", "records": len(records), "candidates": len(candidates), "moved": moved_count},
    )
    return StageResult(stage_dir, len(records), len(candidates), moved_count)


def run_stage3(
    input_dir: Path,
    work_dir: Path,
    run_dir: Path,
    profile,
    unresolved_folders: set[tuple[str, str]] | None = None,
    config: AuditConfig | None = None,
    move_enable: bool = True,
) -> StageResult:
    """在多批次类别的单个文件夹内执行近邻相似性审核。"""
    audit_config = resolve_audit_config(config)
    records = build_similarity_records(input_dir, profile, unresolved_folders or set(), audit_config)
    classes: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for record in records:
        if record.spectrum is not None:
            classes.setdefault((record.raw.genus, record.raw.prefix), []).append(record)
    selected = {
        key
        for key, group in classes.items()
        if len({record.raw.folder for record in group}) > 1
    }
    selected_records = [
        record
        for record in records
        if (record.raw.genus, record.raw.prefix) in selected
    ]
    folders: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for record in selected_records:
        if record.spectrum is not None:
            folders.setdefault((record.raw.genus, record.raw.folder), []).append(record)
    for group in folders.values():
        score_similarity_group(group, audit_config)
    stage_dir = build_stage_dir(run_dir, "stage3")
    candidates = [record for record in selected_records if record.state == "candidate"]
    write_csv(stage_dir / "stage3_scores.csv", build_similarity_rows(selected_records), SIMILARITY_FIELDS)
    moved_count = move_similarity_candidates(selected_records, work_dir, "stage3") if move_enable else 0
    write_json(
        stage_dir / "run.json",
        {
            "stage": "stage3",
            "records": len(selected_records),
            "groups": len(selected),
            "candidates": len(candidates),
            "moved": moved_count,
        },
    )
    return StageResult(stage_dir, len(selected_records), len(candidates), moved_count)
