"""Cosmic 源头文件夹走向记录与日期级摘要。

记录文件以 cosmic_data 下的日期/文件夹为稳定主键，不依赖 init 的序号目录。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ramanv2.common.naming import build_natural_key, is_test_source_folder, parse_folder_prefix, parse_test_folder_prefix


LINEAGE_FILE_NAME = "folder_lineage.json"
LINEAGE_VERSION = 1
MANUAL_DELETE_RELATIVE = Path("delete") / "manual"
SPECIAL_NAMES = {"delete", "shift", "audit", "audit_runs", ".raman_pipeline", "小球"}


def lineage_path(data_root: Path | str) -> Path:
    root = Path(data_root).resolve()
    return root.parent / ".raman_pipeline" / LINEAGE_FILE_NAME


def load_folder_lineage(data_root: Path | str) -> dict[str, object]:
    path = lineage_path(data_root)
    if not path.is_file():
        return {"version": LINEAGE_VERSION, "source_root": str(Path(data_root).resolve()), "folders": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != LINEAGE_VERSION:
        raise ValueError("folder_lineage.json 版本不受支持")
    folders = payload.get("folders", {})
    if not isinstance(folders, dict):
        raise ValueError("folder_lineage.json.folders 必须是对象")
    return payload


def save_folder_lineage(data_root: Path | str, payload: dict[str, object]) -> None:
    path = lineage_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prefix_for_folder(folder_name: str) -> str:
    if is_test_source_folder(folder_name):
        return parse_test_folder_prefix(folder_name).upper()
    return parse_folder_prefix(folder_name, uppercase_enable=True)


def _record_template(date_name: str, folder_name: str) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "source_id": f"{date_name}/{folder_name}",
        "date": date_name,
        "folder": folder_name,
        "prefix": _prefix_for_folder(folder_name),
        "genus": "",
        "files": {},
        "folder_status": "active",
        "cleaning": {"status": "pending", "history": []},
        "affine": {"status": "pending", "version": None, "scale": None, "offset": None},
        "compensation": {"value": 0.0, "history": []},
        "audit": {"status": "pending", "history": []},
        "outputs": {"shift": None, "init": None, "profiles": {}, "csdata": None, "prefix_groups": []},
        "history": [{"operation": "lineage_initialized", "recorded_at": now}],
        "updated_at": now,
    }


def _ensure_folder(payload: dict[str, object], date_name: str, folder_name: str) -> dict[str, object]:
    folders = payload.setdefault("folders", {})
    if not isinstance(folders, dict):
        raise ValueError("folder_lineage.json.folders 必须是对象")
    key = f"{date_name}/{folder_name}"
    record = folders.get(key)
    if not isinstance(record, dict):
        record = _record_template(date_name, folder_name)
        folders[key] = record
    return record


def initialize_folder_lineage(
    data_root: Path | str,
    mapping: object | None = None,
) -> dict[str, object]:
    """从 cosmic 活动区和 delete/manual 建立或刷新源头记录。

    小球目录只用于日期级仿射校准，不进入样本走向树。
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"cosmic 数据目录不存在：{root}")
    payload = load_folder_lineage(root)
    payload["version"] = LINEAGE_VERSION
    payload["source_root"] = str(root)
    folders = payload.setdefault("folders", {})
    if not isinstance(folders, dict):
        raise ValueError("folder_lineage.json.folders 必须是对象")

    prefix_to_genus: dict[str, str] = {}
    if mapping is not None and hasattr(mapping, "genera"):
        prefix_to_genus = {
            str(prefix).upper(): str(genus)
            for genus, prefixes in getattr(mapping, "genera").items()
            for prefix in prefixes
        }

    discovered: set[str] = set()

    def scan_base(base: Path, status: str, imported: bool = False) -> None:
        if not base.is_dir():
            return
        for date_dir in sorted((item for item in base.iterdir() if item.is_dir() and item.name.isdigit()), key=lambda item: build_natural_key(item.name)):
            for folder_dir in sorted((item for item in date_dir.iterdir() if item.is_dir()), key=lambda item: build_natural_key(item.name)):
                if folder_dir.name in SPECIAL_NAMES:
                    continue
                files = sorted(
                    (item for item in folder_dir.rglob("*.arc_data") if item.is_file()),
                    key=lambda item: build_natural_key(item.relative_to(folder_dir).as_posix()),
                )
                if not files:
                    continue
                record = _ensure_folder(payload, date_dir.name, folder_dir.name)
                key = f"{date_dir.name}/{folder_dir.name}"
                discovered.add(key)
                record["genus"] = prefix_to_genus.get(str(record.get("prefix", "")).upper(), record.get("genus", ""))
                record["folder_status"] = "active" if status == "active" else "manual_deleted"
                file_records = record.setdefault("files", {})
                if not isinstance(file_records, dict):
                    file_records = {}
                    record["files"] = file_records
                for file_path in files:
                    relative = file_path.relative_to(folder_dir).as_posix()
                    file_records.setdefault(
                        relative,
                        {
                            "status": status,
                            "source_path": f"{date_dir.name}/{folder_dir.name}/{relative}",
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                            "reason": "imported_history" if imported else None,
                        },
                    )

    scan_base(root, "active")
    scan_base(root / MANUAL_DELETE_RELATIVE, "manual_deleted", imported=True)

    for key, record in folders.items():
        if not isinstance(record, dict):
            continue
        file_records = record.get("files", {})
        if isinstance(file_records, dict):
            active_count = sum(1 for item in file_records.values() if isinstance(item, dict) and item.get("status") == "active")
            deleted_count = sum(1 for item in file_records.values() if isinstance(item, dict) and item.get("status") == "manual_deleted")
            record["active_file_count"] = active_count
            record["deleted_file_count"] = deleted_count
        if key not in discovered:
            if record.get("folder_status") == "active":
                record["folder_status"] = "folder_deleted"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()

    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_folder_lineage(root, payload)
    return payload


def record_manual_move(
    data_root: Path | str,
    relative_paths: list[str] | tuple[str, ...] | set[str],
    reason: str,
) -> dict[str, object]:
    payload = load_folder_lineage(data_root)
    now = datetime.now(timezone.utc).isoformat()
    for item in relative_paths:
        normalized = str(item).replace("\\", "/")
        parts = normalized.split("/")
        if len(parts) < 3:
            continue
        date_name, folder_name = parts[0], parts[1]
        relative = "/".join(parts[2:])
        record = _ensure_folder(payload, date_name, folder_name)
        files = record.setdefault("files", {})
        if not isinstance(files, dict):
            files = {}
            record["files"] = files
        files[relative] = {
            "status": "manual_deleted",
            "source_path": normalized,
            "deleted_path": f"delete/manual/{normalized}",
            "reason": reason,
            "recorded_at": now,
        }
        record["folder_status"] = "folder_deleted" if reason in {"folder_remove", "date_remove", "prefix_folder_remove"} else "active"
        record.setdefault("cleaning", {}).setdefault("history", []).append(
            {"operation": reason, "path": relative, "recorded_at": now}
        )
        record.setdefault("history", []).append({"operation": reason, "path": relative, "recorded_at": now})
        record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def record_restore(data_root: Path | str, relative_paths: list[str] | tuple[str, ...] | set[str]) -> dict[str, object]:
    payload = load_folder_lineage(data_root)
    now = datetime.now(timezone.utc).isoformat()
    for item in relative_paths:
        normalized = str(item).replace("\\", "/")
        parts = normalized.split("/")
        if len(parts) < 3:
            continue
        record = _ensure_folder(payload, parts[0], parts[1])
        files = record.setdefault("files", {})
        if isinstance(files, dict):
            relative = "/".join(parts[2:])
            current = files.get(relative, {})
            files[relative] = {**(current if isinstance(current, dict) else {}), "status": "active", "restored_at": now}
        record["folder_status"] = "active"
        record.setdefault("cleaning", {}).setdefault("history", []).append(
            {"operation": "restore", "path": "/".join(parts[2:]), "recorded_at": now}
        )
        record.setdefault("history", []).append({"operation": "restore", "path": normalized, "recorded_at": now})
        record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def record_folder_compensation(data_root: Path | str, date_name: str, folder_name: str, value: float) -> dict[str, object]:
    payload = load_folder_lineage(data_root)
    record = _ensure_folder(payload, str(date_name), str(folder_name))
    now = datetime.now(timezone.utc).isoformat()
    compensation = record.setdefault("compensation", {"value": 0.0, "history": []})
    if not isinstance(compensation, dict):
        compensation = {"value": 0.0, "history": []}
        record["compensation"] = compensation
    previous = float(compensation.get("value", 0.0))
    current = float(value)
    if previous != current:
        compensation.setdefault("history", []).append({"value": previous, "new_value": current, "recorded_at": now})
        compensation["value"] = current
        record.setdefault("history", []).append({"operation": "folder_compensation", "value": current, "recorded_at": now})
    record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def record_affine_approval(
    data_root: Path | str,
    date_name: str,
    version: str | int | None,
    scale: float,
    offset: float,
) -> dict[str, object]:
    payload = initialize_folder_lineage(data_root)
    now = datetime.now(timezone.utc).isoformat()
    folders = payload.get("folders", {})
    if isinstance(folders, dict):
        for key, record in folders.items():
            if not isinstance(record, dict) or not str(key).startswith(f"{date_name}/"):
                continue
            record["affine"] = {
                "status": "approved",
                "version": version,
                "scale": float(scale),
                "offset": float(offset),
                "approved_at": now,
            }
            record.setdefault("history", []).append({"operation": "affine_approved", "recorded_at": now})
            record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def record_audit_result(
    data_root: Path | str,
    date_name: str,
    folder_name: str,
    status: str,
    reason: str | None = None,
) -> dict[str, object]:
    payload = initialize_folder_lineage(data_root)
    record = _ensure_folder(payload, str(date_name), str(folder_name))
    now = datetime.now(timezone.utc).isoformat()
    audit = record.setdefault("audit", {})
    previous_history = audit.get("history", []) if isinstance(audit, dict) else []
    record["audit"] = {
        "status": str(status),
        "reason": reason,
        "recorded_at": now,
        "history": [*previous_history, {"status": status, "reason": reason, "recorded_at": now}],
    }
    record.setdefault("history", []).append({"operation": "audit", "status": status, "recorded_at": now})
    record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def folder_compensation(data_root: Path | str, date_name: str, folder_name: str) -> float:
    payload = load_folder_lineage(data_root)
    record = payload.get("folders", {}).get(f"{date_name}/{folder_name}", {})
    compensation = record.get("compensation", {}) if isinstance(record, dict) else {}
    return float(compensation.get("value", 0.0)) if isinstance(compensation, dict) else 0.0


def record_output_path(
    data_root: Path | str,
    date_name: str,
    folder_name: str,
    *,
    init_path: str | None = None,
    profile_paths: dict[str, str] | None = None,
    csdata_path: str | None = None,
    shift_path: str | None = None,
) -> dict[str, object]:
    payload = load_folder_lineage(data_root)
    record = _ensure_folder(payload, str(date_name), str(folder_name))
    outputs = record.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        record["outputs"] = outputs
    if init_path is not None:
        outputs["init"] = str(init_path)
    if profile_paths:
        current = outputs.setdefault("profiles", {})
        if isinstance(current, dict):
            current.update({str(key): str(value) for key, value in profile_paths.items()})
    if csdata_path is not None:
        outputs["csdata"] = str(csdata_path)
    if shift_path is not None:
        outputs["shift"] = str(shift_path)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["updated_at"] = record["updated_at"]
    save_folder_lineage(data_root, payload)
    return payload


def summarize_lineage(data_root: Path | str) -> dict[str, list[str]]:
    payload = initialize_folder_lineage(data_root)
    folders = payload.get("folders", {})
    summary = {"cleaning_completed_dates": set(), "affine_completed_dates": set(), "audit_completed_dates": set(), "init_generated_dates": set()}
    if not isinstance(folders, dict):
        return {key: [] for key in summary}
    by_date: dict[str, list[dict[str, object]]] = {}
    for record in folders.values():
        if isinstance(record, dict):
            by_date.setdefault(str(record.get("date", "")), []).append(record)
    for date_name, records in by_date.items():
        if records and all(record.get("folder_status") in {"active", "manual_deleted", "folder_deleted"} for record in records):
            summary["cleaning_completed_dates"].add(date_name)
        if records and all(isinstance(record.get("affine"), dict) and (record.get("affine") or {}).get("status") == "approved" for record in records):
            summary["affine_completed_dates"].add(date_name)
        if records and all(isinstance(record.get("audit"), dict) and (record.get("audit") or {}).get("status") == "applied" for record in records):
            summary["audit_completed_dates"].add(date_name)
        if records and all(isinstance(record.get("outputs"), dict) and (record.get("outputs") or {}).get("init") for record in records):
            summary["init_generated_dates"].add(date_name)
    return {key: sorted(values, key=build_natural_key) for key, values in summary.items()}


def sync_pipeline_state_from_lineage(data_root: Path | str, state: object) -> object:
    """只从已有走向文件刷新日期摘要，不重新扫描源数据。"""
    payload = load_folder_lineage(data_root)
    folders = payload.get("folders", {})
    if not isinstance(folders, dict):
        return state
    by_date: dict[str, list[dict[str, object]]] = {}
    for record in folders.values():
        if isinstance(record, dict):
            by_date.setdefault(str(record.get("date", "")), []).append(record)
    affine_dates: set[str] = set()
    audit_dates: set[str] = set()
    init_dates: set[str] = set()
    cleaning_dates: set[str] = set()
    for date_name, records in by_date.items():
        if records and all(item.get("folder_status") in {"active", "manual_deleted", "folder_deleted"} for item in records):
            cleaning_dates.add(date_name)
        if records and all(isinstance(item.get("affine"), dict) and item["affine"].get("status") == "approved" for item in records):
            affine_dates.add(date_name)
        audit_records = [
            item for item in records
            if item.get("folder_status") != "folder_deleted" and not is_test_source_folder(str(item.get("folder", "")))
        ]
        if audit_records and all(isinstance(item.get("audit"), dict) and item["audit"].get("status") == "applied" for item in audit_records):
            audit_dates.add(date_name)
        output_records = [item for item in records if item.get("folder_status") != "folder_deleted"]
        if output_records and all(
            isinstance(item.get("outputs"), dict)
            and ((item["outputs"].get("init") if not is_test_source_folder(str(item.get("folder", ""))) else item["outputs"].get("csdata")))
            for item in output_records
        ):
            init_dates.add(date_name)
    if hasattr(state, "cleaning"):
        state.cleaning.completed_date_names = cleaning_dates
    if hasattr(state, "affine_completed_dates"):
        state.affine_completed_dates = affine_dates
    if hasattr(state, "audit_completed_dates"):
        state.audit_completed_dates = audit_dates
    if hasattr(state, "init_generated_dates"):
        state.init_generated_dates = init_dates
    return state


def list_lineage_folders(
    data_root: Path | str,
    genus: str | None = None,
    prefix: str | None = None,
) -> list[dict[str, object]]:
    payload = initialize_folder_lineage(data_root)
    folders = payload.get("folders", {})
    result = []
    if not isinstance(folders, dict):
        return result
    for record in folders.values():
        if not isinstance(record, dict) or record.get("folder_status") == "folder_deleted":
            continue
        if genus is not None and str(record.get("genus", "")) != str(genus):
            continue
        if prefix is not None and str(record.get("prefix", "")).upper() != str(prefix).upper():
            continue
        result.append(dict(record))
    return sorted(result, key=lambda item: build_natural_key(str(item.get("source_id", ""))))


def remove_source_folder(
    data_root: Path | str,
    source_id: str,
    *,
    shift_root: Path | str | None = None,
) -> dict[str, object]:
    """从 cosmic 源头整夹删除，并按记录清理已知下游目录。"""
    root = Path(data_root).resolve()
    parts = str(source_id).replace("\\", "/").split("/", 1)
    if len(parts) != 2:
        raise ValueError("source_id 必须是 日期/文件夹")
    date_name, folder_name = parts
    source_folder = root / date_name / folder_name
    manual_folder = root / "delete" / "manual" / date_name / folder_name
    if not source_folder.is_dir():
        raise FileNotFoundError(f"源文件夹不存在：{source_folder}")
    files = sorted((item for item in source_folder.rglob("*.arc_data") if item.is_file()), key=lambda item: build_natural_key(item.relative_to(source_folder).as_posix()))
    if not files:
        raise ValueError(f"源文件夹没有光谱：{source_folder}")
    if manual_folder.exists():
        raise FileExistsError(f"人工删除目标已存在：{manual_folder}")
    relative_paths = [f"{date_name}/{folder_name}/{item.relative_to(source_folder).as_posix()}" for item in files]
    try:
        manual_folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_folder), str(manual_folder))
        record_manual_move(root, relative_paths, "prefix_folder_remove")
        removed = remove_downstream_outputs(root, source_id, shift_root=shift_root)
        return {"status": "ready", "source_id": source_id, "moved_paths": relative_paths, **removed}
    except Exception:
        if manual_folder.exists() and not source_folder.exists():
            source_folder.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(manual_folder), str(source_folder))
        record_restore(root, relative_paths)
        raise


def _prepare_audit_cleanup(
    shift_root: Path,
    source_id: str,
) -> list[tuple[Path, Path]]:
    """暂存包含源文件夹的审核报告，删除该文件夹的报告记录。"""
    audit_root = shift_root / "audit_runs"
    if not audit_root.is_dir():
        return []
    backups: list[tuple[Path, Path]] = []
    try:
        for run_dir in sorted((item for item in audit_root.iterdir() if item.is_dir()), key=lambda item: build_natural_key(item.name)):
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                continue
            try:
                report = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            folders = report.get("folders", [])
            if not isinstance(folders, list):
                continue
            relevant = {
                str(item).replace("\\", "/")
                for item in folders
                if str(item).replace("\\", "/") == source_id
                or str(item).replace("\\", "/").startswith(f"{source_id}/")
            }
            if not relevant:
                continue
            backup = audit_root / f".{run_dir.name}.lineage_backup_{uuid4().hex}"
            run_dir.replace(backup)
            backups.append((run_dir, backup))
            remaining = [item for item in folders if str(item).replace("\\", "/") not in relevant]
            if not remaining:
                continue
            shutil.copytree(backup, run_dir)
            report["folders"] = remaining
            items = report.get("items", [])
            if isinstance(items, list):
                report["items"] = [
                    item
                    for item in items
                    if not (
                        isinstance(item, dict)
                        and (
                            str(item.get("relative_path", "")).replace("\\", "/") == source_id
                            or str(item.get("relative_path", "")).replace("\\", "/").startswith(f"{source_id}/")
                        )
                    )
                ]
                report["item_count"] = len(report["items"])
                report["candidate_count"] = sum(item.get("state") == "candidate" for item in report["items"] if isinstance(item, dict))
            temporary = summary_path.with_name(f".{summary_path.name}.{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(summary_path)
    except Exception:
        for target, backup in reversed(backups):
            if target.exists():
                shutil.rmtree(target)
            if backup.exists():
                backup.replace(target)
        raise
    return backups


def remove_downstream_outputs(
    data_root: Path | str,
    source_id: str,
    *,
    shift_root: Path | str | None = None,
) -> dict[str, object]:
    """按走向记录删除一个源文件夹的 shift、init、Profile 和 CSdata 目录。"""
    root = Path(data_root).resolve()
    payload = load_folder_lineage(root)
    record = payload.get("folders", {}).get(str(source_id), {})
    outputs = record.get("outputs", {}) if isinstance(record, dict) else {}
    output_paths: list[Path] = []
    if isinstance(outputs, dict):
        for value in (outputs.get("shift"), outputs.get("init"), outputs.get("csdata")):
            if value:
                output_paths.append(Path(str(value)))
        profiles = outputs.get("profiles", {})
        if isinstance(profiles, dict):
            output_paths.extend(Path(str(value)) for value in profiles.values() if value)
    backups: list[tuple[Path, Path]] = []
    audit_backups: list[tuple[Path, Path]] = []
    if shift_root is not None:
        parts = str(source_id).split("/", 1)
        if len(parts) == 2:
            shift_base = Path(shift_root).resolve()
            output_paths.extend(
                path
                for path in (shift_base / parts[0] / parts[1], shift_base / "delete" / parts[0] / parts[1])
                if path not in output_paths
            )
    try:
        for path in output_paths:
            if not path.exists():
                continue
            backup = path.with_name(f".{path.name}.lineage_backup_{uuid4().hex}")
            path.replace(backup)
            backups.append((path, backup))
    except Exception:
        for target, backup in reversed(backups):
            if backup.exists() and not target.exists():
                backup.replace(target)
        raise
    try:
        if shift_root is not None:
            audit_backups = _prepare_audit_cleanup(Path(shift_root).resolve(), str(source_id))
        if isinstance(record, dict):
            record["outputs"] = {"shift": None, "init": None, "profiles": {}, "csdata": None, "prefix_groups": []}
            record["audit"] = {"status": "removed", "reason": "source_folder_removed", "history": []}
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_folder_lineage(root, payload)
    except Exception:
        for target, backup in reversed(audit_backups):
            if target.exists():
                shutil.rmtree(target)
            if backup.exists():
                backup.replace(target)
        for target, backup in reversed(backups):
            if backup.exists() and not target.exists():
                backup.replace(target)
        raise
    for _, backup in backups:
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()
    for _, backup in audit_backups:
        if backup.exists():
            shutil.rmtree(backup)
    return {"removed_outputs": [str(path) for path in output_paths]}


def remove_downstream_files(
    data_root: Path | str,
    relative_paths: list[str] | tuple[str, ...] | set[str],
    *,
    shift_root: Path | str | None = None,
) -> dict[str, object]:
    """删除单谱在各下游目录中的对应文件，保留同文件夹其他光谱。"""
    root = Path(data_root).resolve()
    payload = load_folder_lineage(root)
    removed: list[str] = []
    backups: list[tuple[Path, Path]] = []
    try:
        for item in relative_paths:
            normalized = str(item).replace("\\", "/")
            parts = normalized.split("/")
            if len(parts) < 3:
                continue
            record = payload.get("folders", {}).get("/".join(parts[:2]), {})
            outputs = record.get("outputs", {}) if isinstance(record, dict) else {}
            targets: list[Path] = []
            if isinstance(outputs, dict):
                for value in (outputs.get("init"), outputs.get("csdata")):
                    if value:
                        targets.append(Path(str(value)) / parts[-1])
                profiles = outputs.get("profiles", {})
                if isinstance(profiles, dict):
                    targets.extend(Path(str(value)) / parts[-1] for value in profiles.values() if value)
            if shift_root is not None:
                shift_base = Path(shift_root).resolve()
                targets.extend((shift_base / parts[0] / parts[1] / parts[-1], shift_base / "delete" / parts[0] / parts[1] / parts[-1]))
            for target in targets:
                if target.is_file():
                    backup = target.with_name(f".{target.name}.lineage_backup_{uuid4().hex}")
                    target.replace(backup)
                    backups.append((target, backup))
                    removed.append(str(target))
        save_folder_lineage(root, payload)
    except Exception:
        for target, backup in reversed(backups):
            if backup.exists() and not target.exists():
                backup.replace(target)
        raise
    for _, backup in backups:
        backup.unlink(missing_ok=True)
    return {"removed_files": removed}


__all__ = [
    "LINEAGE_FILE_NAME",
    "folder_compensation",
    "initialize_folder_lineage",
    "lineage_path",
    "load_folder_lineage",
    "record_folder_compensation",
    "record_affine_approval",
    "record_audit_result",
    "record_manual_move",
    "record_restore",
    "record_output_path",
    "list_lineage_folders",
    "remove_source_folder",
    "remove_downstream_outputs",
    "remove_downstream_files",
    "save_folder_lineage",
    "summarize_lineage",
    "sync_pipeline_state_from_lineage",
]
