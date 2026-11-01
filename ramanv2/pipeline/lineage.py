"""Cosmic 源头文件夹走向记录与日期级摘要。

记录文件以 cosmic_data 下的日期/文件夹为稳定主键，不依赖 init 的序号目录。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ramanv2.common.publishing import preserve_paths, write_text_atomic
from ramanv2.common.naming import (
    NON_TRAINING_FOLDER_NAMES,
    PIPELINE_RESERVED_FOLDER_NAMES,
    build_natural_key,
    parse_folder_prefix,
)


LINEAGE_FILE_NAME = "folder_lineage.json"
LINEAGE_VERSION = 1
MANUAL_DELETE_RELATIVE = Path("delete") / "manual"
FOLDER_DELETE_RELATIVE = Path("delete") / "folder_delete"
SPECIAL_NAMES = PIPELINE_RESERVED_FOLDER_NAMES | NON_TRAINING_FOLDER_NAMES


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
    write_text_atomic(
        lineage_path(data_root),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _record_template(date_name: str, folder_name: str) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "source_id": f"{date_name}/{folder_name}",
        "date": date_name,
        "folder": folder_name,
        "prefix": parse_folder_prefix(folder_name, uppercase_enable=True),
        "genus": "",
        "files": {},
        "folder_status": "active",
        "cleaning": {"status": "pending", "history": []},
        "affine": {"status": "pending", "version": None, "scale": None, "offset": None},
        "outputs": {"shift": None, "init": None, "profiles": {}, "test": None, "prefix_groups": []},
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
    date_names: set[str] | None = None,
) -> dict[str, object]:
    """从 cosmic 活动区、人工删除区和整夹删除区建立或刷新源头记录。

    小球目录只用于日期级仿射校准，不进入样本走向树。
    指定 date_names 时只扫描和刷新这些日期，保留其他日期的记录。
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
        for date_dir in sorted((item for item in base.iterdir() if item.is_dir() and item.name.isdigit() and (date_names is None or item.name in date_names)), key=lambda item: build_natural_key(item.name)):
            for folder_dir in sorted((item for item in date_dir.iterdir() if item.is_dir()), key=lambda item: build_natural_key(item.name)):
                if folder_dir.name in SPECIAL_NAMES or folder_dir.name.startswith("."):
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
                if status == "active":
                    record["folder_status"] = "active"
                elif status == "folder_deleted":
                    record["folder_status"] = "folder_deleted"
                elif record.get("folder_status") != "folder_deleted":
                    record["folder_status"] = "manual_deleted"
                file_records = record.setdefault("files", {})
                if not isinstance(file_records, dict):
                    file_records = {}
                    record["files"] = file_records
                for file_path in files:
                    relative = file_path.relative_to(folder_dir).as_posix()
                    file_records.setdefault(
                        relative,
                        {
                            "status": "active" if status == "active" else "manual_deleted",
                            "source_path": f"{date_dir.name}/{folder_dir.name}/{relative}",
                            "deleted_path": (
                                f"{FOLDER_DELETE_RELATIVE.as_posix()}/{date_dir.name}/{folder_dir.name}/{relative}"
                                if status == "folder_deleted"
                                else (
                                    f"{MANUAL_DELETE_RELATIVE.as_posix()}/{date_dir.name}/{folder_dir.name}/{relative}"
                                    if status != "active"
                                    else None
                                )
                            ),
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                            "reason": "folder_remove" if status == "folder_deleted" else ("imported_history" if imported else None),
                        },
                    )

    scan_base(root, "active")
    scan_base(root / MANUAL_DELETE_RELATIVE, "manual_deleted", imported=True)
    scan_base(root / FOLDER_DELETE_RELATIVE, "folder_deleted", imported=True)

    for key, record in folders.items():
        if not isinstance(record, dict):
            continue
        if date_names is not None and record.get("date") not in date_names:
            continue
        file_records = record.get("files", {})
        if isinstance(file_records, dict):
            active_count = sum(1 for item in file_records.values() if isinstance(item, dict) and item.get("status") == "active")
            deleted_count = sum(1 for item in file_records.values() if isinstance(item, dict) and item.get("status") == "manual_deleted")
            record["active_file_count"] = active_count
            record["deleted_file_count"] = deleted_count
        if key not in discovered:
            # 小球与派生目录本就不参与扫描，不能被"未发现即整夹删除"误判。
            # 小球与派生目录本就不参与扫描，不能被"未发现即整夹删除"误判。
            if record.get("folder_status") == "active" and str(record.get("folder", "")) not in SPECIAL_NAMES:
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
        record["init_pending"] = True
        record["folder_status"] = "folder_deleted" if reason in {"folder_remove", "date_remove", "prefix_folder_remove"} else "active"
        record.setdefault("cleaning", {}).setdefault("history", []).append(
            {"operation": reason, "path": relative, "recorded_at": now}
        )
        record.setdefault("history", []).append({"operation": reason, "path": relative, "recorded_at": now})
        record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def record_folder_move(
    data_root: Path | str,
    relative_paths: list[str] | tuple[str, ...] | set[str],
) -> dict[str, object]:
    """记录整夹删除，覆盖同一文件夹此前的人工删除状态。"""
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
            "deleted_path": f"{FOLDER_DELETE_RELATIVE.as_posix()}/{normalized}",
            "reason": "folder_remove",
            "recorded_at": now,
        }
        record["init_pending"] = True
        record["folder_status"] = "folder_deleted"
        record.setdefault("history", []).append({"operation": "folder_remove", "path": normalized, "recorded_at": now})
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
        record["init_pending"] = True
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
    *,
    shift_dir: Path | None = None,
) -> dict[str, object]:
    """仅刷新所选日期的仿射记录，重建时一并保存 shift 路径，避免逐夹写盘。"""
    payload = initialize_folder_lineage(data_root, date_names={str(date_name)})
    now = datetime.now(timezone.utc).isoformat()
    folders = payload.get("folders", {})
    if isinstance(folders, dict):
        for key, record in folders.items():
            if not isinstance(record, dict) or not str(key).startswith(f"{date_name}/"):
                continue
            previous = record.get("affine", {})
            if previous.get("scale") != float(scale) or previous.get("offset") != float(offset):
                record["init_pending"] = True
            record["affine"] = {
                "status": "approved",
                "version": version,
                "scale": float(scale),
                "offset": float(offset),
                "approved_at": now,
            }
            if shift_dir is not None:
                outputs = record.setdefault("outputs", {})
                # 整日发布也同步了已删除组的移除，不能只给仍存在的组记时间。
                outputs["shift_updated_at"] = now
                if (shift_dir / str(record["folder"])).is_dir():
                    outputs["shift"] = str(shift_dir / str(record["folder"]))
            record.setdefault("history", []).append({"operation": "affine_approved", "recorded_at": now})
            record["updated_at"] = now
    payload["updated_at"] = now
    save_folder_lineage(data_root, payload)
    return payload


def apply_output_path(
    payload: dict[str, object],
    date_name: str,
    folder_name: str,
    *,
    init_path: str | None = None,
    profile_paths: dict[str, str] | None = None,
    test_path: str | None = None,
    shift_path: str | None = None,
) -> dict[str, object]:
    """在已加载的走向记录中写入输出路径，由调用方统一保存，避免逐组读写大文件。"""
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
    if test_path is not None:
        outputs["test"] = str(test_path)
    if shift_path is not None:
        outputs["shift"] = str(shift_path)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    if shift_path is not None:
        outputs["shift_updated_at"] = record["updated_at"]
    payload["updated_at"] = record["updated_at"]
    return payload


def record_output_path(
    data_root: Path | str,
    date_name: str,
    folder_name: str,
    *,
    init_path: str | None = None,
    profile_paths: dict[str, str] | None = None,
    test_path: str | None = None,
    shift_path: str | None = None,
) -> dict[str, object]:
    payload = load_folder_lineage(data_root)
    apply_output_path(
        payload,
        date_name,
        folder_name,
        init_path=init_path,
        profile_paths=profile_paths,
        test_path=test_path,
        shift_path=shift_path,
    )
    save_folder_lineage(data_root, payload)
    return payload


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
    init_dates: set[str] = set()
    cleaning_dates: set[str] = set()
    for date_name, records in by_date.items():
        if records and all(item.get("folder_status") in {"active", "manual_deleted", "folder_deleted"} for item in records):
            cleaning_dates.add(date_name)
        if records and all(isinstance(item.get("affine"), dict) and item["affine"].get("status") == "approved" for item in records):
            affine_dates.add(date_name)
        output_records = [item for item in records if item.get("folder_status") != "folder_deleted"]
        if output_records and not any(item.get("init_pending") for item in records) and all(
            isinstance(item.get("outputs"), dict)
            and (item["outputs"].get("init") or item["outputs"].get("test"))
            for item in output_records
        ):
            init_dates.add(date_name)
    if hasattr(state, "cleaning"):
        state.cleaning.completed_date_names = cleaning_dates
    if hasattr(state, "affine_completed_dates"):
        state.affine_completed_dates = affine_dates
    if hasattr(state, "init_generated_dates"):
        state.init_generated_dates = init_dates
    return state


def has_active_source_folders(data_root: Path | str, date_name: str) -> bool:
    """判断日期下是否仍有普通源文件夹，排除小球和派生目录。"""
    date_dir = Path(data_root).resolve() / str(date_name)
    excluded = SPECIAL_NAMES
    if not date_dir.is_dir():
        return False
    return any(
        folder.is_dir()
        and folder.name not in excluded
        and any(folder.rglob("*.arc_data"))
        for folder in date_dir.iterdir()
    )


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
    root = Path(data_root).resolve()
    for record in folders.values():
        if not isinstance(record, dict) or record.get("folder_status") == "folder_deleted":
            continue
        source_path = root / str(record.get("date", "")) / str(record.get("folder", ""))
        if not source_path.is_dir():
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
    """只归档 cosmic 整夹及其已有单谱归档；失败时恢复文件与走向记录。"""
    del shift_root  # 保留旧调用签名，派生数据由显式重建更新。
    root = Path(data_root).resolve()
    parts = str(source_id).replace("\\\\", "/").split("/")
    if len(parts) != 2:
        raise ValueError("source_id 必须是 日期/文件夹")
    date_name, folder_name = parts
    if len(date_name) != 8 or not date_name.isascii() or not date_name.isdigit() or not folder_name or folder_name in SPECIAL_NAMES or folder_name in {".", ".."}:
        raise ValueError("source_id 必须是 日期/文件夹")
    source = root / date_name / folder_name
    manual = root / MANUAL_DELETE_RELATIVE / date_name / folder_name
    archive = root / FOLDER_DELETE_RELATIVE / date_name / folder_name
    if any(not path.resolve().is_relative_to(root) for path in (source, manual, archive)):
        raise ValueError("归档路径超出 cosmic 目录")
    if archive.exists() and not archive.is_dir():
        raise FileExistsError(f"整夹归档目标不是文件夹：{archive}")
    if not source.is_dir() and not manual.is_dir():
        if not archive.is_dir():
            raise FileNotFoundError(f"源文件夹不存在：{source}")
        paths = [f"{date_name}/{folder_name}/{path.relative_to(archive).as_posix()}" for path in sorted(archive.rglob("*.arc_data"))]
        return {"status": "already_ready", "source_id": source_id, "moved_paths": paths}
    with preserve_paths([source, manual, archive, lineage_path(root)]):
        if manual.exists():
            if source.exists():
                _move_folder_contents(manual, source)
            else:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(manual), str(source))
        files = sorted(source.rglob("*.arc_data"), key=lambda path: build_natural_key(path.relative_to(source).as_posix()))
        if not files:
            raise ValueError(f"源文件夹没有光谱：{source}")
        paths = [f"{date_name}/{folder_name}/{path.relative_to(source).as_posix()}" for path in files]
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            _move_folder_contents(source, archive)
        else:
            shutil.move(str(source), str(archive))
        record_folder_move(root, paths)
    return {"status": "ready", "source_id": source_id, "moved_paths": paths}


def _move_folder_contents(source_folder: Path, target_folder: Path) -> list[tuple[Path, Path]]:
    """将源文件夹内容合并到已有目标，并在冲突前完成预检查。"""
    source_files = sorted(
        (item for item in source_folder.rglob("*") if item.is_file()),
        key=lambda item: build_natural_key(item.relative_to(source_folder).as_posix()),
    )
    conflicts = [
        target_folder / item.relative_to(source_folder)
        for item in source_files
        if (target_folder / item.relative_to(source_folder)).exists()
    ]
    if conflicts:
        names = "; ".join(str(path) for path in conflicts[:5])
        if len(conflicts) > 5:
            names += f"；另有 {len(conflicts) - 5} 个冲突"
        raise FileExistsError(f"人工删除目标存在同名文件：{names}")

    moved: list[tuple[Path, Path]] = []
    try:
        for source in source_files:
            target = target_folder / source.relative_to(source_folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
        shutil.rmtree(source_folder)
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        raise
    return moved


__all__ = [
    "FOLDER_DELETE_RELATIVE",
    "LINEAGE_FILE_NAME",
    "apply_output_path",
    "has_active_source_folders",
    "initialize_folder_lineage",
    "lineage_path",
    "load_folder_lineage",
    "record_affine_approval",
    "record_manual_move",
    "record_folder_move",
    "record_restore",
    "record_output_path",
    "list_lineage_folders",
    "remove_source_folder",
    "save_folder_lineage",
    "sync_pipeline_state_from_lineage",
]
