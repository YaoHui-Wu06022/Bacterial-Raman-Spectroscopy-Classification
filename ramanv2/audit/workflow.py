"""层级数据集与 CS 测试集的审核编排。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ramanv2.audit.config import AuditConfig, resolve_audit_config
from ramanv2.audit.raw_quality import build_clean_records, mark_raw_candidates, score_raw_record
from ramanv2.audit.reports import build_raw_rows, build_similarity_rows, build_stage_dir, write_csv, write_json
from ramanv2.audit.similarity import preprocess_similarity_records, score_folder_neighbor_groups, score_neighbor_group
from ramanv2.audit.workspace import (
    build_candidate_rows,
    build_work_dir,
    commit_candidate_rows,
    move_stage_candidate,
)
from ramanv2.common.naming import parse_folder_prefix
from ramanv2.data.config import DataBuildConfig, resolve_build_config
from ramanv2.data.profiles import get_dataset_dir, get_profile


HIERARCHY_RAW_FIELDS = (
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
HIERARCHY_SIMILARITY_FIELDS = (
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
COMMIT_FIELDS = ("stage", "rel_path")
CS_RAW_FIELDS = (
    "state",
    "reasons",
    "rel_path",
    "folder",
    "points",
    "coverage",
    "malformed_lines",
    "longest_flat_points",
    "saturation_points",
    "noise_ratio",
)
CS_SIMILARITY_FIELDS = (
    "state",
    "reasons",
    "rel_path",
    "folder",
    "reference_count",
    "neighbor_count",
    "neighbor_corr",
    "rmse",
)


def build_audit_run_dir(dataset_dir: Path) -> Path:
    """创建保存本次层级数据集审核报告和运行池的目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dataset_dir / "audit_runs" / f"{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_hierarchy_rows(rows: list[dict[str, object]], records) -> list[dict[str, object]]:
    """为层级数据集审核行补充属与类别前缀字段。"""
    return [
        {
            **row,
            "source_rel_path": "",
            "genus": record.group,
            "prefix": parse_folder_prefix(record.folder),
        }
        for row, record in zip(rows, records)
    ]


def build_hierarchy_raw_rows(records) -> list[dict[str, object]]:
    """构建含属名和类别前缀的 Stage 1 报告行。"""
    return build_hierarchy_rows(build_raw_rows(records), records)


def build_hierarchy_similarity_rows(records) -> list[dict[str, object]]:
    """构建含属名和类别前缀的近邻相似性报告行。"""
    return build_hierarchy_rows(build_similarity_rows(records), records)


def run_stage1(
    input_dir: Path,
    work_dir: Path,
    run_dir: Path,
    folder_depth: int,
    config: AuditConfig,
    fields: tuple[str, ...],
    build_report_rows,
) -> int:
    """执行原始质量审核，并在运行池中移动 Stage 1 候选。"""
    records = build_clean_records(input_dir, folder_depth=folder_depth)
    for record in records:
        score_raw_record(record, config.input, config.raw_quality)
    noise_limit = mark_raw_candidates(records, config.raw_quality)
    candidates = [record for record in records if record.state in {"candidate", "unscorable"}]
    stage_dir = build_stage_dir(run_dir, "stage1")
    write_csv(stage_dir / "stage1_raw_scores.csv", build_report_rows(records), fields)
    write_csv(stage_dir / "stage1_candidates.csv", build_report_rows(candidates), fields)
    for record in candidates:
        move_stage_candidate(record, work_dir, "stage1")
    write_json(
        stage_dir / "run.json",
        {"stage": "stage1", "records": len(records), "candidates": len(candidates), "noise_limit": noise_limit},
    )
    return len(records)


def collect_prefix_groups(records):
    """按属与类别前缀收集已完成预处理的光谱。"""
    groups = {}
    for record in records:
        if record.spectrum is not None:
            groups.setdefault((record.group, parse_folder_prefix(record.folder)), []).append(record)
    return groups


def collect_multi_folder_records(records) -> list:
    """收集同属同前缀下具有多个文件夹的记录。"""
    groups = collect_prefix_groups(records)
    selected_keys = {
        key
        for key, group in groups.items()
        if len({record.folder for record in group}) > 1
    }
    return [
        record
        for record in records
        if (record.group, parse_folder_prefix(record.folder)) in selected_keys
    ]


def run_stage2(
    input_dir: Path,
    work_dir: Path,
    run_dir: Path,
    profile,
    config: AuditConfig,
    folder_depth: int,
    is_multi_folder_only: bool,
    fields: tuple[str, ...],
    build_report_rows,
) -> int:
    """按文件夹执行 Stage 2 近邻审核，并写出评分和候选表。"""
    records = build_clean_records(input_dir, folder_depth=folder_depth)
    preprocess_similarity_records(records, profile, config.input, config.cleaning)
    selected_records = collect_multi_folder_records(records) if is_multi_folder_only else records
    folder_count = score_folder_neighbor_groups(selected_records, config.neighbor)
    candidates = [record for record in selected_records if record.state == "candidate"]
    stage_dir = build_stage_dir(run_dir, "stage2")
    write_csv(stage_dir / "stage2_scores.csv", build_report_rows(selected_records), fields)
    write_csv(stage_dir / "stage2_candidates.csv", build_report_rows(candidates), fields)
    for record in candidates:
        move_stage_candidate(record, work_dir, "stage2")
    write_json(
        stage_dir / "run.json",
        {
            "stage": "stage2",
            "records": len(selected_records),
            "folders": folder_count,
            "candidates": len(candidates),
        },
    )
    return len(selected_records)


def run_stage3(
    input_dir: Path,
    work_dir: Path,
    run_dir: Path,
    profile,
    config: AuditConfig,
    folder_depth: int,
    fields: tuple[str, ...],
    build_report_rows,
) -> int:
    """按同属同前缀全量分组执行 Stage 3 近邻审核。"""
    records = build_clean_records(input_dir, folder_depth=folder_depth)
    preprocess_similarity_records(records, profile, config.input, config.cleaning)
    groups = collect_prefix_groups(records)
    for group in groups.values():
        score_neighbor_group(group, config.neighbor)
    candidates = [record for record in records if record.state == "candidate"]
    stage_dir = build_stage_dir(run_dir, "stage3")
    write_csv(stage_dir / "stage3_scores.csv", build_report_rows(records), fields)
    write_csv(stage_dir / "stage3_candidates.csv", build_report_rows(candidates), fields)
    for record in candidates:
        move_stage_candidate(record, work_dir, "stage3")
    write_json(
        stage_dir / "run.json",
        {
            "stage": "stage3",
            "records": len(records),
            "groups": len(groups),
            "candidates": len(candidates),
        },
    )
    return len(records)


def run_clean_dir(
    dataset_key: str = "alldata",
    config: AuditConfig | None = None,
) -> Path:
    """完成层级数据集三阶段审核，并在成功后统一提交候选。"""
    audit_config = resolve_audit_config(config)
    profile = get_profile(dataset_key)
    dataset_dir = get_dataset_dir(profile)
    run_dir = build_audit_run_dir(dataset_dir)
    work_dir = build_work_dir(run_dir, dataset_dir / profile.root_init)
    input_dir = work_dir / "init"
    stage1_record_count = run_stage1(
        input_dir,
        work_dir,
        run_dir,
        folder_depth=2,
        config=audit_config,
        fields=HIERARCHY_RAW_FIELDS,
        build_report_rows=build_hierarchy_raw_rows,
    )
    stage2_record_count = run_stage2(
        input_dir,
        work_dir,
        run_dir,
        profile,
        audit_config,
        folder_depth=2,
        is_multi_folder_only=True,
        fields=HIERARCHY_SIMILARITY_FIELDS,
        build_report_rows=build_hierarchy_similarity_rows,
    )
    stage3_record_count = run_stage3(
        input_dir,
        work_dir,
        run_dir,
        profile,
        audit_config,
        folder_depth=2,
        fields=HIERARCHY_SIMILARITY_FIELDS,
        build_report_rows=build_hierarchy_similarity_rows,
    )
    candidate_rows = build_candidate_rows(work_dir)
    write_csv(run_dir / "commit_plan.csv", candidate_rows, COMMIT_FIELDS)
    moved_count = commit_candidate_rows(candidate_rows, dataset_dir, profile.root_init)
    write_json(
        run_dir / "run.json",
        {
            "stage1": stage1_record_count,
            "stage2": stage2_record_count,
            "stage3": stage3_record_count,
            "candidates_committed": moved_count,
        },
    )
    return run_dir


def build_cs_clean_run_dir(dataset_dir: Path) -> Path:
    """创建保存本次 CS 审核报告和运行池的目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dataset_dir / "clean_runs" / f"{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_cs_clean_dir(
    profile,
    dataset_dir: Path,
    input_config,
    build_config: DataBuildConfig | None = None,
) -> Path:
    """完成 CS Stage 1 与文件夹内 Stage 2 后统一提交候选。"""
    run_dir = build_cs_clean_run_dir(dataset_dir)
    work_dir = build_work_dir(run_dir, dataset_dir / profile.root_init)
    input_dir = work_dir / "init"
    audit_config = AuditConfig(input=input_config, cleaning=resolve_build_config(build_config))
    stage1_record_count = run_stage1(
        input_dir,
        work_dir,
        run_dir,
        folder_depth=1,
        config=audit_config,
        fields=CS_RAW_FIELDS,
        build_report_rows=build_raw_rows,
    )
    stage2_record_count = run_stage2(
        input_dir,
        work_dir,
        run_dir,
        profile,
        audit_config,
        folder_depth=1,
        is_multi_folder_only=False,
        fields=CS_SIMILARITY_FIELDS,
        build_report_rows=build_similarity_rows,
    )
    candidate_rows = build_candidate_rows(work_dir)
    write_csv(run_dir / "commit_plan.csv", candidate_rows, COMMIT_FIELDS)
    moved_count = commit_candidate_rows(candidate_rows, dataset_dir, profile.root_init)
    write_json(
        run_dir / "run.json",
        {
            "stage1": stage1_record_count,
            "stage2": stage2_record_count,
            "candidates_committed": moved_count,
        },
    )
    return run_dir
