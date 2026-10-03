"""校正后数据的文件夹级离群审核与安全移动。"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ramanv2.audit.config import AuditConfig, resolve_audit_config
from ramanv2.audit.records import CleanRecord
from ramanv2.audit.similarity import preprocess_similarity_records, score_neighbor_group
from ramanv2.common.naming import build_natural_key
from ramanv2.core.paths import PROJECT_ROOT


SHIFTED_ROOT = PROJECT_ROOT / "dataset" / "shift_data"
SPECIAL_FOLDER_NAMES = {"小球", "shift", "delete", "audit_runs", ".raman_pipeline"}
REPORT_FIELDS = (
    "date",
    "folder",
    "relative_path",
    "state",
    "reasons",
    "reference_count",
    "neighbor_count",
    "neighbor_corr",
    "rmse",
    "corr_limit",
    "rmse_limit",
    "move_status",
)


def _resolve_date_dirs(root: Path, date_name: str | None) -> list[Path]:
    if date_name is not None:
        date_dir = root / date_name
        if not date_dir.is_dir() or not date_name.isdigit():
            raise FileNotFoundError(f"日期目录不存在：{date_dir}")
        return [date_dir]
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: path.name,
    )


def _resolve_folder_dirs(
    root: Path,
    date_name: str | None,
    folder_name: str | None,
) -> list[Path]:
    folders = []
    for date_dir in _resolve_date_dirs(root, date_name):
        if folder_name is not None:
            folder_dir = date_dir / folder_name
            if folder_dir.is_dir() and folder_name not in SPECIAL_FOLDER_NAMES:
                folders.append(folder_dir)
            continue
        folders.extend(
            path
            for path in sorted(date_dir.iterdir(), key=lambda item: build_natural_key(item.name))
            if path.is_dir() and path.name not in SPECIAL_FOLDER_NAMES
        )
    return folders


def _build_records(root: Path, folder_dir: Path) -> list[CleanRecord]:
    date_name = folder_dir.parent.name
    records = []
    for path in sorted(folder_dir.glob("*.arc_data"), key=lambda item: build_natural_key(item.name)):
        records.append(
            CleanRecord(
                path=path,
                rel_path=path.relative_to(root).as_posix(),
                group=date_name,
                folder=folder_dir.name,
            )
        )
    return records


def _item_payload(record: CleanRecord) -> dict[str, object]:
    return {
        "date": record.group,
        "folder": record.folder,
        "relative_path": record.rel_path,
        "state": record.state,
        "reasons": list(record.reasons),
        "reference_count": record.reference_count,
        "neighbor_count": record.neighbor_count,
        "neighbor_corr": None if record.neighbor_corr != record.neighbor_corr else record.neighbor_corr,
        "rmse": None if record.rmse != record.rmse else record.rmse,
        "corr_limit": None if record.corr_limit != record.corr_limit else record.corr_limit,
        "rmse_limit": None if record.rmse_limit != record.rmse_limit else record.rmse_limit,
        "move_status": "pending" if record.state == "candidate" else "not_applicable",
    }


def _write_csv(path: Path, items: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for item in items:
                row = dict(item)
                row["reasons"] = ";".join(str(reason) for reason in row["reasons"])
                writer.writerow(row)
        _replace_report_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_report_file(temporary: Path, target: Path) -> None:
    """Windows 下报告文件可能短暂被读取，替换时做有限重试。"""
    for attempt, delay in enumerate((0.0, 0.05, 0.2, 0.5, 1.0)):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 4:
                raise


def _write_report(report: dict[str, object]) -> None:
    run_dir = Path(str(report["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    items = list(report["items"])
    temporary = run_dir / f".summary.json.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _replace_report_file(temporary, run_dir / "summary.json")
    finally:
        temporary.unlink(missing_ok=True)
    _write_csv(run_dir / "scores.csv", items)
    _write_csv(
        run_dir / "candidates.csv",
        [item for item in items if item["state"] == "candidate"],
    )


def analyze_folder(
    source_root: Path | str = SHIFTED_ROOT,
    date_name: str | None = None,
    folder_name: str | None = None,
    config: AuditConfig | None = None,
    profile_id: str = "GN",
) -> dict[str, object]:
    """分析一个日期文件夹；不修改源目录。"""
    reports = analyze_folders(source_root, date_name, folder_name, config, profile_id)
    if len(reports["folders"]) != 1:
        raise ValueError("analyze_folder 必须只匹配一个普通文件夹")
    return reports


def analyze_folders(
    source_root: Path | str = SHIFTED_ROOT,
    date_name: str | None = None,
    folder_name: str | None = None,
    config: AuditConfig | None = None,
    profile_id: str = "GN",
) -> dict[str, object]:
    """独立分析日期/文件夹，生成候选报告但不移动文件。"""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"校正后数据目录不存在：{root}")
    audit_config = resolve_audit_config(config)
    run_dir = root / "audit_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_dir = run_dir.with_name(f"{run_dir.name}_{uuid4().hex[:8]}")
    items: list[dict[str, object]] = []
    folders: list[str] = []
    for folder_dir in _resolve_folder_dirs(root, date_name, folder_name):
        folders.append(folder_dir.relative_to(root).as_posix())
        records = _build_records(root, folder_dir)
        preprocess_similarity_records(records, profile_id, audit_config.input, audit_config.cleaning)
        score_neighbor_group(
            [record for record in records if record.spectrum is not None],
            audit_config.neighbor,
        )
        items.extend(_item_payload(record) for record in records)
    report: dict[str, object] = {
        "status": "analyzed",
        "source_root": str(root),
        "run_dir": str(run_dir),
        "folders": folders,
        "candidate_count": sum(item["state"] == "candidate" for item in items),
        "item_count": len(items),
        "minimum_references": audit_config.neighbor.minimum_references,
        "thresholds": {"mad_multiplier": 3.5, "correlation": "low", "rmse": "high"},
        "items": items,
    }
    _write_report(report)
    return report


def apply_folder_outliers(report: dict[str, object]) -> dict[str, object]:
    """将报告中的候选移动到 delete/<日期>/<文件夹>，失败时回滚。"""
    root = Path(str(report["source_root"])).resolve()
    moved: list[tuple[Path, Path]] = []
    candidates = [item for item in report["items"] if item["state"] == "candidate"]
    try:
        for item in candidates:
            source = root / str(item["relative_path"])
            target = root / "delete" / str(item["date"]) / str(item["folder"]) / source.name
            if not source.is_file():
                if target.is_file():
                    item["state"] = "already_moved"
                    item["move_status"] = "already_moved"
                    continue
                raise FileNotFoundError(f"候选文件不存在：{source}")
            if target.exists():
                raise FileExistsError(f"候选目标已存在：{target}")
        for item in candidates:
            if item["state"] == "already_moved":
                continue
            source = root / str(item["relative_path"])
            target = root / "delete" / str(item["date"]) / str(item["folder"]) / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
            item["state"] = "moved"
            item["move_status"] = "moved"
    except Exception as error:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        delete_root = root / "delete"
        for directory in sorted(
            (path for path in delete_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        if delete_root.is_dir() and not any(delete_root.iterdir()):
            delete_root.rmdir()
        for item in candidates:
            if item["state"] == "moved":
                item["state"] = "candidate"
            if item["state"] == "candidate":
                item["move_status"] = "failed"
        report["status"] = "move_failed"
        report["error"] = str(error)
        _write_report(report)
        raise
    report["status"] = "applied"
    report["moved_count"] = len(moved) + sum(item["state"] == "already_moved" for item in candidates)
    _write_report(report)
    return report


def run_folder_audit(
    source_root: Path | str = SHIFTED_ROOT,
    date_name: str | None = None,
    folder_name: str | None = None,
    config: AuditConfig | None = None,
    profile_id: str = "GN",
) -> dict[str, object]:
    """执行一次明确确认的文件夹审核并自动移动候选。"""
    return apply_folder_outliers(
        analyze_folders(source_root, date_name, folder_name, config, profile_id)
    )


__all__ = (
    "SHIFTED_ROOT",
    "analyze_folder",
    "analyze_folders",
    "apply_folder_outliers",
    "run_folder_audit",
)
