"""按当前磁盘重建流水线记录（走向树 + 页面状态）。

元数据（``folder_lineage.json`` / ``state.json``）被删除或与磁盘分叉后，用本模块一次性重建：

- 走向树：以 ``cosmic_data`` 活动区、``delete/manual``、``delete/folder_delete`` 的目录内容
  为准（``initialize_folder_lineage`` 本来就是磁盘派生、幂等的）；
- 日期级仿射参数：按 ``shift_data/<日期>/shift/affine.json`` 中状态为 ``manual_approved``
  的报告回填到该日期所有文件夹，使下游页面的门禁与磁盘一致；
- 产物路径：``outputs`` 中指向已不存在目录的条目一律清空（init 迁到 init_test、来源整夹删除后
  的残留路径会让下游误判产物仍在）；
- 全量 init 摘要：按存活记录的 ``outputs`` 与磁盘计数重写 ``dataset_builds.full_init``，
  其中 ``test_exports`` 直接取自走向记录——``init_test`` 的目录名只有 ``{前缀}{日期}_{序号}``，
  来源名不落盘，无法从目录结构反推；
- 页面状态：目录清单取「活动区 + 人工删除区」的文件夹（整夹删除的已不在待审核范围），
  提交记录取 ``delete/manual`` 当前的全部相对路径。

**重建不了的部分**：清洗阶段逐文件夹的审核进度、以及每次移动的操作时间戳——磁盘上没有
留下痕迹。因此清洗完成度按派生规则给出（每个有走向记录的日期都视为清洗完成）。
本模块不移动、不删除任何光谱文件。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ramanv2.common.naming import build_natural_key
from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.data.profiles import resolve_test_root
from ramanv2.pipeline.lineage import (
    initialize_folder_lineage,
    lineage_path,
    save_folder_lineage,
    sync_pipeline_state_from_lineage,
)
from ramanv2.pipeline.state import (
    load_pipeline_state,
    pipeline_state_path,
    save_pipeline_state,
)


MANUAL_DELETE_DIR = "delete/manual"
FOLDER_DELETED_STATUS = "folder_deleted"
INIT_DIR_NAME = "init"


def approved_affine_reports(shift_root: Path | str) -> dict[str, dict[str, object]]:
    """读出 ``shift_data`` 下所有状态为 ``manual_approved`` 的日期级仿射报告。"""
    root = Path(shift_root).resolve()
    reports: dict[str, dict[str, object]] = {}
    if not root.is_dir():
        return reports
    for date_dir in sorted(root.iterdir(), key=lambda item: build_natural_key(item.name)):
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        report = load_calibration_report(root, date_dir.name)
        if report is not None and str(report.get("status", "")) == "manual_approved":
            reports[date_dir.name] = report
    return reports


def _backfill_affine(
    payload: dict[str, object],
    reports: dict[str, dict[str, object]],
    recorded_at: str,
) -> int:
    """把已确认的日期级参数写回该日期所有文件夹的 affine 记录。"""
    folders = payload.get("folders", {})
    if not isinstance(folders, dict):
        return 0
    filled = 0
    for record in folders.values():
        if not isinstance(record, dict):
            continue
        report = reports.get(str(record.get("date", "")))
        if report is None:
            continue
        scale = report.get("scale", report.get("base_scale"))
        offset = report.get("offset", report.get("base_offset"))
        if scale is None or offset is None:
            continue
        approved_at = str(report.get("approved_at") or recorded_at)
        record["affine"] = {
            "status": "approved",
            "version": approved_at,
            "scale": float(scale),
            "offset": float(offset),
            "approved_at": approved_at,
        }
        record["updated_at"] = recorded_at
        filled += 1
    return filled


def _reset_stale_affine(
    payload: dict[str, object],
    reports: dict[str, dict[str, object]],
    recorded_at: str,
) -> int:
    """把磁盘上已无 ``manual_approved`` 报告的日期重置为待校准。

    重建是「磁盘的函数」而非「在旧记录上叠加」：若某日期的 ``affine.json`` 已不存在或不是
    ``manual_approved``，残留的 approved 必须清掉，否则下游会以为该日期有可用参数。
    """
    folders = payload.get("folders", {})
    if not isinstance(folders, dict):
        return 0
    cleared = 0
    for record in folders.values():
        if not isinstance(record, dict):
            continue
        if str(record.get("date", "")) in reports:
            continue
        affine = record.get("affine")
        if isinstance(affine, dict) and str(affine.get("status", "")) == "approved":
            record["affine"] = {"status": "pending", "version": None, "scale": None, "offset": None}
            record["updated_at"] = recorded_at
            cleared += 1
    return cleared


def _manual_deleted_paths(data_root: Path) -> set[str]:
    """人工删除区当前的全部相对路径，等价于清洗提交记录。"""
    base = data_root / MANUAL_DELETE_DIR
    if not base.is_dir():
        return set()
    return {
        path.relative_to(base).as_posix()
        for path in base.rglob("*.arc_data")
        if path.is_file()
    }


def _folder_catalog(payload: dict[str, object]) -> dict[str, set[str]]:
    """目录清单只收活动区与人工删除区的文件夹；整夹删除的已不在待审核范围。"""
    folders = payload.get("folders", {})
    catalog: dict[str, set[str]] = {}
    if not isinstance(folders, dict):
        return catalog
    for record in folders.values():
        if not isinstance(record, dict):
            continue
        if str(record.get("folder_status", "")) == FOLDER_DELETED_STATUS:
            continue
        date_name = str(record.get("date", ""))
        folder_name = str(record.get("folder", ""))
        if not date_name or not folder_name:
            continue
        catalog.setdefault(date_name, set()).add(folder_name)
    return catalog


def _count_arc_files(path: Path) -> int:
    return sum(1 for item in path.glob("*.arc_data") if item.is_file())


def _direct_init_folders(init_root: Path, referenced: set[Path]) -> tuple[list[str], int]:
    """列出直接放进 ``init``、没有 cosmic 来源的组。

    这类组（例如只有处理结果、没有原始采集目录的 ``20261027`` 批次）不参与走向树，
    因此不会出现在任何 ``outputs.init`` 里；单独记一份，让 ``init`` 的账目能对平。
    """
    names: list[str] = []
    files = 0
    if not init_root.is_dir():
        return names, files
    for group in sorted(init_root.glob("*/*"), key=lambda item: build_natural_key(item.as_posix())):
        if not group.is_dir() or group.resolve() in referenced:
            continue
        count = _count_arc_files(group)
        if not count:
            continue
        names.append(group.relative_to(init_root).as_posix())
        files += count
    return names, files


def _prune_missing_outputs(payload: dict[str, object]) -> dict[str, int]:
    """清空指向已不存在路径的 ``outputs`` 条目。

    重建以磁盘为准：``init`` 被整体迁到 ``init_test``、或来源被整夹删除后，残留的旧路径
    会让下游误以为产物仍在——``collect_init_export_sources`` 会把不存在的目录当成可导出来源，
    ``sync_pipeline_state_from_lineage`` 也会错误地把日期算成已生成。
    """
    folders = payload.get("folders", {})
    counts = {"init": 0, "test": 0, "shift": 0, "profiles": 0}
    if not isinstance(folders, dict):
        return counts
    for record in folders.values():
        if not isinstance(record, dict):
            continue
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for name in ("init", "test", "shift"):
            value = outputs.get(name)
            if value and not Path(value).is_dir():
                outputs[name] = None
                counts[name] += 1
        profiles = outputs.get("profiles")
        if isinstance(profiles, dict):
            for key in [key for key, value in profiles.items() if not (value and Path(value).is_dir())]:
                profiles.pop(key, None)
                counts["profiles"] += 1
    return counts


def _rebuild_full_init_report(
    payload: dict[str, object],
    init_root: Path,
    test_root: Path | None,
) -> dict[str, object]:
    """按走向记录里的 ``outputs`` 与磁盘计数重写 ``dataset_builds.full_init``。

    ``test_exports`` 是「来源 → init_test 目标」的唯一权威映射：``init_test`` 的目录名由
    ``{前缀}{日期}_{序号}`` 生成，来源名不会留在磁盘上，因此本函数只做汇总，不重新推导路由。
    """
    folders = payload.get("folders", {})
    test_exports: list[dict[str, object]] = []
    date_stats: dict[str, dict[str, int]] = {}
    test_folders: set[str] = set()
    selected: list[str] = []
    referenced: set[Path] = set()
    source_files = 0
    init_files = 0
    test_files = 0
    if isinstance(folders, dict):
        for source_id, record in folders.items():
            if not isinstance(record, dict):
                continue
            if str(record.get("folder_status", "")) == FOLDER_DELETED_STATUS:
                continue
            date_name = str(record.get("date", ""))
            stats = date_stats.setdefault(
                date_name,
                {"source_files": 0, "output_files": 0, "folder_count": 0, "test_files": 0},
            )
            stats["folder_count"] += 1
            active = int(record.get("active_file_count") or 0)
            stats["source_files"] += active
            source_files += active
            outputs = record.get("outputs")
            if not isinstance(outputs, dict):
                continue
            init_path = outputs.get("init")
            if init_path:
                referenced.add(Path(init_path).resolve())
                count = _count_arc_files(Path(init_path))
                init_files += count
                stats["output_files"] += count
            test_path = outputs.get("test")
            if test_path:
                target = Path(test_path)
                count = _count_arc_files(target)
                test_files += count
                stats["output_files"] += count
                stats["test_files"] += count
                test_folders.add(target.name)
                selected.append(str(source_id))
                test_exports.append(
                    {
                        "source": str(source_id),
                        "source_name": str(record.get("folder", "")),
                        "target": target.name,
                        "genus": str(record.get("genus", "")),
                        "date": date_name,
                        "file_count": count,
                    }
                )
    dates = sorted(date_stats, key=build_natural_key)
    direct_folders, direct_files = _direct_init_folders(init_root, referenced)
    return {
        "status": "ready",
        "mode": "full",
        "rebuilt_from_disk": True,
        "source_dates": dates,
        "eligible_dates": dates,
        "skipped_dates": [],
        "date_status": {name: {"status": "eligible"} for name in dates},
        "conflict_dates": [],
        "source_files": source_files,
        "output_files": init_files + test_files,
        "init_source_files": source_files - test_files,
        "init_output_files": init_files,
        "direct_init_folders": direct_folders,
        "direct_init_files": direct_files,
        "init_disk_files": init_files + direct_files,
        "date_stats": date_stats,
        "profile_stats": {},
        "generated_profiles": [],
        "init_root": str(init_root),
        "test_root": str(test_root) if test_root is not None else None,
        "test_files": test_files,
        "test_folders": sorted(test_folders, key=build_natural_key),
        "test_folder_count": len(test_folders),
        "test_output_files": test_files,
        "test_source_files": test_files,
        "test_exports": sorted(test_exports, key=lambda entry: build_natural_key(str(entry["source"]))),
        "selected_test_folders": sorted(selected, key=build_natural_key),
    }


def rebuild_pipeline_records(
    data_root: Path | str,
    shift_root: Path | str,
    mapping: object | None = None,
    state_path: Path | str | None = None,
    *,
    init_root: Path | str | None = None,
    test_root: Path | str | None = None,
) -> dict[str, object]:
    """按当前磁盘重写 ``folder_lineage.json`` 与 ``state.json``，返回统计摘要。"""
    root = Path(data_root).resolve()
    shift = Path(shift_root).resolve()
    init_target = Path(init_root).resolve() if init_root is not None else root.parent / INIT_DIR_NAME
    test_target = Path(test_root).resolve() if test_root is not None else resolve_test_root(root.parent.parent)
    target_state = (
        Path(state_path) if state_path is not None else pipeline_state_path(root.parent.parent)
    )
    recorded_at = datetime.now(timezone.utc).isoformat()

    payload = initialize_folder_lineage(root, mapping)
    reports = approved_affine_reports(shift)
    _reset_stale_affine(payload, reports, recorded_at)
    affine_backfilled = _backfill_affine(payload, reports, recorded_at)
    pruned = _prune_missing_outputs(payload)
    payload["updated_at"] = recorded_at
    save_folder_lineage(root, payload)

    full_report = _rebuild_full_init_report(
        payload, init_target, test_target if test_target.is_dir() else None
    )
    catalog = _folder_catalog(payload)
    committed = _manual_deleted_paths(root)
    state = load_pipeline_state(target_state)
    state.dataset_builds["full_init"] = full_report
    sync_pipeline_state_from_lineage(root, state)
    state.cleaning.folder_catalog = catalog
    state.cleaning.committed_paths = committed
    # 清洗进度无法从磁盘恢复；这里与 lineage 的派生结果保持一致，避免两套完成度互相覆盖。
    state.cleaning.completed_folder_keys = {
        f"{date_name}/{folder_name}"
        for date_name, folder_names in catalog.items()
        for folder_name in folder_names
    }
    state.cleaning.marked_paths = set()
    state.cleaning.mark_reasons = {}
    state.cleaning.seeded_folder_keys = set()
    state.cleaning.folder_page_indices = {}
    state.cleaning.current_date_name = None
    state.cleaning.current_folder_key = None
    save_pipeline_state(target_state, state)

    folders = payload.get("folders", {})
    folder_records = folders.values() if isinstance(folders, dict) else ()
    status_counts: dict[str, int] = {}
    for record in folder_records:
        if isinstance(record, dict):
            key = str(record.get("folder_status", "unknown"))
            status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "lineage_path": str(lineage_path(root)),
        "state_path": str(target_state),
        "folder_count": sum(status_counts.values()),
        "status_counts": status_counts,
        "date_count": len(catalog),
        "approved_affine_dates": len(reports),
        "affine_backfilled_folders": affine_backfilled,
        "pruned_outputs": pruned,
        "committed_paths": len(committed),
        "cleaning_completed_dates": len(state.cleaning.completed_date_names),
        "init_generated_dates": len(state.init_generated_dates),
        "test_folder_count": int(full_report["test_folder_count"]),
        "test_files": int(full_report["test_files"]),
        "init_files": int(full_report["init_output_files"]),
        "direct_init_folders": list(full_report["direct_init_folders"]),
        "direct_init_files": int(full_report["direct_init_files"]),
        "init_disk_files": int(full_report["init_disk_files"]),
        "source_files": int(full_report["source_files"]),
    }


__all__ = ["approved_affine_reports", "rebuild_pipeline_records"]
