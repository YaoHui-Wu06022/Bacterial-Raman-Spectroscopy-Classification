"""audit clean 的运行池编排与成功后统一提交。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from raman_temp.audit.config import AuditConfig, resolve_audit_config
from raman_temp.audit.raw_quality import run_stage1
from raman_temp.audit.reports import append_delta_log, build_audit_run_dir, read_delta, write_csv, write_delta, write_json
from raman_temp.audit.shift import APPLIED_SHIFT_STATUSES, apply_folder_shift, apply_shift_plan, build_shift_plan, build_unresolved_folder_keys
from raman_temp.audit.similarity import run_stage2, run_stage3
from raman_temp.audit.test_sync import (
    COMMIT_FIELDS,
    build_candidate_commit_rows,
    build_work_pool,
    commit_candidate_rows,
    parse_audit_source_folder,
    sync_work_test_init,
)
from raman_temp.data.profiles import get_dataset_dir, get_profile
from raman_temp.data.test_transfer import build_test_transfer


def load_test_transfer_sources(manifest_path: Path) -> dict[tuple[str, str], str]:
    """从 `*t` manifest 解析主数据文件夹到测试菌 CS 来源的映射。"""
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {
            (row["target_genus"], row["target_folder"]): row["source_folder"]
            for row in rows
            if row.get("target_genus") and row.get("target_folder") and row.get("source_folder")
        }


def build_work_delta_values(
    pool_init_dir: Path,
    data_delta: dict[tuple[str, str], float],
    test_delta: dict[tuple[str, str], float],
    test_transfer_sources: dict[tuple[str, str], str],
) -> dict[tuple[str, str], float]:
    """为运行池中的常规、测试副本和 `*t` 文件夹建立累计平移状态。"""
    values = data_delta.copy()
    for genus_dir in sorted(path for path in pool_init_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            key = (genus_dir.name, folder_dir.name)
            source_folder = parse_audit_source_folder(folder_dir.name)
            if source_folder is None:
                source_folder = test_transfer_sources.get(key)
            if source_folder is not None:
                values[key] = test_delta.get((".", source_folder), 0.0)
    return values


def apply_test_pool_shift_steps(
    plan: list[dict[str, object]],
    work_test_init_dir: Path,
) -> dict[str, float]:
    """将测试菌临时副本的平移步骤同步到运行池中的测试菌来源谱。"""
    totals: dict[str, float] = {}
    for row in plan:
        source_folder = parse_audit_source_folder(str(row["folder"]))
        if source_folder is None:
            continue
        step = float(row["step_delta"])
        if abs(step) > 1e-9:
            apply_folder_shift(work_test_init_dir / source_folder, step)
        totals[source_folder] = float(row["target_delta"])
    return totals


def align_test_training_copies(
    plan: list[dict[str, object]],
    pool_init_dir: Path,
    test_transfer_sources: dict[tuple[str, str], str],
    test_totals: dict[str, float],
) -> None:
    """使运行池 `*t` 副本与其测试菌来源保持相同累计平移。"""
    plan_by_key = {(str(row["genus"]), str(row["folder"])): row for row in plan}
    for key, source_folder in test_transfer_sources.items():
        if source_folder not in test_totals or key not in plan_by_key:
            continue
        row = plan_by_key[key]
        desired = test_totals[source_folder]
        correction = desired - float(row["target_delta"])
        if abs(correction) > 1e-9:
            apply_folder_shift(pool_init_dir / key[0] / key[1], correction)
        row["target_delta"] = desired
        row["step_delta"] = float(row["step_delta"]) + correction


def build_delta_outputs(
    plan: list[dict[str, object]],
    test_totals: dict[str, float],
) -> tuple[dict[tuple[str, str], tuple[str, float]], dict[tuple[str, str], tuple[str, float]]]:
    """从最终平移计划构造主数据与测试菌根目录的 delta 写入结构。"""
    data_values = {}
    for row in plan:
        if parse_audit_source_folder(str(row["folder"])) is None:
            data_values[(str(row["genus"]), str(row["folder"]))] = (str(row["prefix"]), float(row["target_delta"]))
    test_values = {
        (".", folder): (folder, delta)
        for folder, delta in test_totals.items()
    }
    return data_values, test_values


def build_automatic_delta_log_rows(plan: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """按平移计划拆分主数据和测试菌的自动平移日志记录。"""
    data_rows = []
    test_rows = []
    now = datetime.now().isoformat(timespec="seconds")
    for row in plan:
        step = float(row["step_delta"])
        if row["status"] not in APPLIED_SHIFT_STATUSES or abs(step) < 1e-9:
            continue
        entry = {
            "time": now,
            "genus": row["genus"],
            "folder": row["folder"],
            "prefix": row["prefix"],
            "step_delta": f"{step:+g}",
            "cumulative_delta": f"{float(row['target_delta']):+g}",
            "files_changed": row.get("files_changed", 0),
            "note": f"anchor_model={row['anchor_model']}; center={row['anchor_peak_cm']}; quality={row['anchor_quality']}; target=1002",
        }
        source_folder = parse_audit_source_folder(str(row["folder"]))
        if source_folder is not None:
            entry["genus"] = "."
            entry["folder"] = source_folder
            entry["prefix"] = str(row["prefix"])
            test_rows.append(entry)
        else:
            data_rows.append(entry)
    return data_rows, test_rows


def run_clean(
    dataset_key: str = "alldata",
    test_key: str = "test",
    config: AuditConfig | None = None,
) -> Path:
    """在运行池执行完整清洗，并在全部 Stage 成功后提交真实数据修改。"""
    audit_config = resolve_audit_config(config)
    dataset_profile = get_profile(dataset_key)
    test_profile = get_profile(test_key)
    dataset_dir = get_dataset_dir(dataset_profile)
    test_dir = get_dataset_dir(test_profile)
    run_dir = build_audit_run_dir(dataset_dir)
    pool = build_work_pool(run_dir, dataset_dir / dataset_profile.root_init, test_dir / test_profile.root_init)
    data_delta = read_delta(dataset_dir / "delta.txt")
    test_delta = read_delta(test_dir / "delta.txt")
    transfer_sources = load_test_transfer_sources(test_dir / "test_transfer_manifest.csv")

    stage1 = run_stage1(pool.pool_init_dir, pool.work_dir, run_dir, audit_config)
    current_delta = build_work_delta_values(pool.pool_init_dir, data_delta, test_delta, transfer_sources)
    plan = build_shift_plan(pool.pool_init_dir, current_delta, audit_config)
    apply_shift_plan(pool.pool_init_dir, plan)
    test_totals = {
        folder: delta
        for (genus, folder), delta in test_delta.items()
        if genus == "."
    }
    test_totals.update(apply_test_pool_shift_steps(plan, pool.test_init_dir))
    align_test_training_copies(plan, pool.pool_init_dir, transfer_sources, test_totals)
    write_csv(run_dir / "shift_plan.csv", plan, tuple(plan[0]) if plan else ())
    unresolved = build_unresolved_folder_keys(plan)
    stage2 = run_stage2(pool.pool_init_dir, pool.work_dir, run_dir, dataset_profile, unresolved, audit_config)
    stage3 = run_stage3(pool.pool_init_dir, pool.work_dir, run_dir, dataset_profile, unresolved, audit_config)

    candidate_rows = build_candidate_commit_rows(pool.work_dir)
    write_csv(run_dir / "commit_plan.csv", candidate_rows, COMMIT_FIELDS)
    data_values, test_values = build_delta_outputs(plan, test_totals)
    commit_count = commit_candidate_rows(candidate_rows, dataset_dir, test_dir)
    synced_count = sync_work_test_init(pool.test_init_dir, test_dir / test_profile.root_init, candidate_rows)
    write_delta(dataset_dir / "delta.txt", data_values)
    write_delta(test_dir / "delta.txt", test_values)
    data_log_rows, test_log_rows = build_automatic_delta_log_rows(plan)
    if data_log_rows:
        append_delta_log(dataset_dir / "delta_log.txt", data_log_rows)
    if test_log_rows:
        append_delta_log(test_dir / "delta_log.txt", test_log_rows)
    transfer_result = build_test_transfer(
        test_dir / test_profile.root_init,
        dataset_dir / dataset_profile.root_init,
        test_dir / "test_transfer_manifest.csv",
        test_dir / "test_transfer_folder_map.csv",
    )
    write_json(
        run_dir / "run.json",
        {"stage1": stage1.record_count, "stage2": stage2.record_count, "stage3": stage3.record_count, "candidates_committed": commit_count, "test_spectra_synced": synced_count, "test_transfer_files": transfer_result.transferred_count},
    )
    return run_dir
