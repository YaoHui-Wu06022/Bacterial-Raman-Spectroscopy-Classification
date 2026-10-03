"""从人工清洗删除目录恢复普通光谱，并重新执行受影响文件夹的 audit。"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ramanv2.audit.folder import run_folder_audit
from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.common.naming import (
    build_natural_key,
    is_test_source_folder,
    parse_folder_prefix,
    parse_test_folder_prefix,
)
from ramanv2.common.publishing import create_temporary_path, rename_directory
from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import record_audit_result, record_restore
from ramanv2.pipeline.state import load_pipeline_state, save_pipeline_state


MANUAL_DELETE_DIR = "delete/manual"
RESTORE_LOG_NAME = "restore_log.jsonl"
BEAD_DIR_NAME = "小球"


@dataclass(frozen=True)
class DeletedSpectrum:
    """一条人工删除的普通光谱，relative_path 是原始数据相对路径。"""

    relative_path: str
    source_relative_path: str
    date_name: str
    folder_name: str
    file_name: str
    genus: str
    prefix: str


def _prefix_for_folder(folder_name: str) -> str:
    if is_test_source_folder(folder_name):
        return parse_test_folder_prefix(folder_name).upper()
    return parse_folder_prefix(folder_name, uppercase_enable=True)


def _prefix_to_genus(mapping: DatasetMapping) -> dict[str, str]:
    return {
        prefix.upper(): genus
        for genus, prefixes in mapping.genera.items()
        for prefix in prefixes
    }


def scan_deleted_spectra(
    data_root: Path | str,
    profile_catalog: DatasetMapping,
    species_name: str | None = None,
    date_name: str | None = None,
    include_bead: bool = False,
) -> list[DeletedSpectrum]:
    """只扫描 ``cosmic_data/delete/manual``，并忽略小球删除目录。"""
    root = Path(data_root).resolve()
    manual_root = root / MANUAL_DELETE_DIR
    if not manual_root.is_dir():
        return []
    prefix_genus = _prefix_to_genus(profile_catalog)
    records: list[DeletedSpectrum] = []
    for date_dir in sorted((item for item in manual_root.iterdir() if item.is_dir()), key=lambda item: build_natural_key(item.name)):
        if date_name is not None and date_dir.name != str(date_name):
            continue
        if not date_dir.name.isdigit():
            continue
        for folder_dir in sorted((item for item in date_dir.iterdir() if item.is_dir()), key=lambda item: build_natural_key(item.name)):
            if folder_dir.name == BEAD_DIR_NAME and not include_bead:
                continue
            prefix = _prefix_for_folder(folder_dir.name)
            genus = prefix_genus.get(prefix, "")
            if species_name is not None and species_name not in {genus, prefix, folder_dir.name}:
                continue
            for source in sorted(
                (item for item in folder_dir.iterdir() if item.is_file() and item.suffix.lower() == ".arc_data"),
                key=lambda item: build_natural_key(item.name),
            ):
                relative = "/".join((date_dir.name, folder_dir.name, source.name))
                records.append(
                    DeletedSpectrum(
                        relative_path=relative,
                        source_relative_path=f"{MANUAL_DELETE_DIR}/{relative}",
                        date_name=date_dir.name,
                        folder_name=folder_dir.name,
                        file_name=source.name,
                        genus=genus,
                        prefix=prefix,
                    )
                )
    return records


def _load_affine_parameters(shift_root: Path, date_name: str) -> tuple[dict[str, object], float, float]:
    report = load_calibration_report(shift_root, date_name)
    if report is None:
        raise FileNotFoundError(f"日期没有仿射参数：{date_name}")
    if report.get("status") != "manual_approved":
        raise ValueError(f"日期尚未完成仿射校准：{date_name}")
    scale = report.get("base_scale", report.get("scale"))
    offset = report.get("base_offset", report.get("offset"))
    if scale is None or offset is None:
        raise ValueError(f"日期仿射参数不完整：{date_name}")
    return report, float(scale), float(offset)


def _resolve_init_target(init_root: Path, data_root: Path, record: DeletedSpectrum) -> Path:
    """按已有 canonical 目录优先、删除和当前目录自然排序兜底解析目标。"""
    genus_dir = init_root / record.genus
    prefix_date = f"{record.prefix}{record.date_name}_"
    existing = sorted(
        (item for item in genus_dir.iterdir() if item.is_dir() and item.name.startswith(prefix_date)),
        key=lambda item: build_natural_key(item.name),
    ) if genus_dir.is_dir() else []
    if len(existing) == 1:
        return existing[0]
    names: set[str] = set()
    for parent in (data_root / record.date_name, data_root / MANUAL_DELETE_DIR / record.date_name):
        if parent.is_dir():
            names.update(
                item.name
                for item in parent.iterdir()
                if item.is_dir() and item.name != BEAD_DIR_NAME and _prefix_for_folder(item.name) == record.prefix
            )
    ordered = sorted(names, key=build_natural_key)
    if record.folder_name not in ordered:
        raise ValueError(f"无法确定 init 来源文件夹：{record.relative_path}")
    target = genus_dir / f"{record.prefix}{record.date_name}_{ordered.index(record.folder_name) + 1:02d}"
    if existing and target not in existing:
        raise ValueError(f"init 来源目录映射不唯一：{record.relative_path}")
    return target


def _append_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _audit_profile_id(mapping: DatasetMapping, genus: str) -> str:
    for profile_name in ("GN", "GP", "FUNG"):
        if genus in mapping.datasets.get(profile_name, ()):
            return profile_name
    return "MICRO"


def _restore_directory_backup(target: Path, backup: Path | None) -> None:
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if backup is not None and backup.exists():
        rename_directory(backup, target)


def _restore_bead_spectra(
    data_root: Path,
    state_path: Path | str,
    records: list[DeletedSpectrum],
) -> dict[str, object]:
    """恢复没有 affine 参数日期的小球谱，供后续仿射校准使用。"""
    sources = {record.relative_path: data_root / MANUAL_DELETE_DIR / record.relative_path for record in records}
    targets = {record.relative_path: data_root / record.relative_path for record in records}
    created: list[Path] = []
    try:
        for relative, source in sources.items():
            target = targets[relative]
            if not source.is_file():
                raise FileNotFoundError(f"人工删除小球光谱不存在：{source}")
            if target.exists():
                raise FileExistsError(f"小球原始目标已存在：{target}")
            wavenumbers, intensities = read_arc_data(source)
            if not wavenumbers.size or wavenumbers.size != intensities.size:
                raise ValueError(f"无效小球光谱：{source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            created.append(target)

        state = load_pipeline_state(state_path)
        dates = {record.date_name for record in records}
        for record in records:
            state.cleaning.committed_paths.discard(record.relative_path)
            state.cleaning.marked_paths.discard(record.relative_path)
            state.cleaning.mark_reasons.pop(record.relative_path, None)
            state.cleaning.folder_catalog.setdefault(record.date_name, set()).add(BEAD_DIR_NAME)
            state.cleaning.completed_folder_keys.add(f"{record.date_name}/{BEAD_DIR_NAME}")
        save_pipeline_state(state_path, state)

        for source in sources.values():
            source.unlink()
        for parent in sorted({source.parent for source in sources.values()}, key=lambda item: len(item.parts), reverse=True):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        _append_log(
            Path(state_path).resolve().parent / RESTORE_LOG_NAME,
            {
                "operation": "restore_bead",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "dates": sorted(dates, key=build_natural_key),
                "items": [
                    {
                        "source_manual": str(sources[record.relative_path]),
                        "cosmic": str(targets[record.relative_path]),
                        "date": record.date_name,
                        "folder": BEAD_DIR_NAME,
                        "file": record.file_name,
                    }
                    for record in records
                ],
            },
        )
        return {
            "status": "bead_restored",
            "restored_file_count": len(records),
            "dates": sorted(dates, key=build_natural_key),
            "paths": sorted(sources, key=build_natural_key),
        }
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise


def restore_deleted_spectra(
    data_root: Path | str,
    shift_root: Path | str,
    init_root: Path | str,
    state_path: Path | str,
    profile_catalog: DatasetMapping,
    selected_relative_paths: list[str] | tuple[str, ...] | set[str],
) -> dict[str, object]:
    """从 cosmic 删除区恢复原始谱，重新生成受影响 shift 并更新下游数据。"""
    root = Path(data_root).resolve()
    shift_base = Path(shift_root).resolve()
    init_base = Path(init_root).resolve()
    normalized = set()
    for item in selected_relative_paths:
        path = str(item).replace("\\", "/")
        prefix = f"{MANUAL_DELETE_DIR}/"
        normalized.add(path[len(prefix):] if path.startswith(prefix) else path)
    selected = sorted(normalized, key=build_natural_key)
    if not selected:
        raise ValueError("至少选择一条需要恢复的光谱")
    prefix_genus = _prefix_to_genus(profile_catalog)
    records = {
        item.relative_path: item
        for item in scan_deleted_spectra(root, profile_catalog, include_bead=True)
    }
    missing = [path for path in selected if path not in records]
    if missing:
        raise FileNotFoundError(f"人工删除光谱不存在：{missing}")
    chosen = [records[path] for path in selected]
    bead_records = [record for record in chosen if record.folder_name == BEAD_DIR_NAME]
    if bead_records:
        if len(bead_records) != len(chosen):
            raise ValueError("小球光谱不能与普通采集光谱混合恢复")
        return _restore_bead_spectra(root, state_path, chosen)
    affected_folders = sorted(
        {(record.date_name, record.folder_name) for record in chosen},
        key=lambda item: build_natural_key(f"{item[0]}/{item[1]}"),
    )

    date_parameters: dict[str, tuple[dict[str, object], float, float]] = {}
    source_raw: dict[str, Path] = {}
    raw_targets: dict[str, Path] = {}
    shift_targets: dict[str, Path] = {}
    init_targets: dict[str, Path] = {}
    for record in chosen:
        if not record.genus or prefix_genus.get(record.prefix) != record.genus:
            raise ValueError(f"删除文件夹前缀未映射：{record.folder_name}")
        raw_source = root / MANUAL_DELETE_DIR / record.relative_path
        raw_target = root / record.relative_path
        shift_target = shift_base / record.date_name / record.folder_name / record.file_name
        if not raw_source.is_file():
            raise FileNotFoundError(f"人工删除光谱不存在：{raw_source}")
        if raw_target.exists():
            raise FileExistsError(f"原始目标已存在：{raw_target}")
        if shift_target.exists():
            raise FileExistsError(f"shift 目标已存在：{shift_target}")
        if record.date_name not in date_parameters:
            date_parameters[record.date_name] = _load_affine_parameters(shift_base, record.date_name)
        init_target = _resolve_init_target(init_base, root, record)
        if (init_target / record.file_name).exists():
            raise FileExistsError(f"init 目标已存在：{init_target / record.file_name}")
        source_raw[record.relative_path] = raw_source
        raw_targets[record.relative_path] = raw_target
        shift_targets[record.relative_path] = shift_target
        init_targets[record.relative_path] = init_target

    shift_backups: dict[Path, Path | None] = {}
    init_backups: dict[Path, Path | None] = {}
    created_raw: list[Path] = []
    created_shift: list[Path] = []
    published_init: list[Path] = []
    temporary_init: list[Path] = []
    audit_reports: list[dict[str, object]] = []
    audit_started_dates: set[str] = set()
    try:
        for folder in {target.parent for target in shift_targets.values()}:
            backup = create_temporary_path(folder)
            if folder.exists():
                shutil.copytree(folder, backup)
                shift_backups[folder] = backup
            else:
                shift_backups[folder] = None
        for folder in set(init_targets.values()):
            backup = create_temporary_path(folder)
            if folder.exists():
                shutil.copytree(folder, backup)
                init_backups[folder] = backup
            else:
                init_backups[folder] = None

        # 只从 cosmic_data/delete/manual 恢复原始谱，重新按日期级 affine 生成 shift。
        for record in chosen:
            raw_x, raw_y = read_arc_data(source_raw[record.relative_path])
            if not raw_x.size or raw_x.size != raw_y.size:
                raise ValueError(f"无效光谱：{record.relative_path}")
            raw_targets[record.relative_path].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_raw[record.relative_path], raw_targets[record.relative_path])
            created_raw.append(raw_targets[record.relative_path])
            shift_targets[record.relative_path].parent.mkdir(parents=True, exist_ok=True)
            _, scale, offset = date_parameters[record.date_name]
            corrected = scale * raw_x + offset
            write_arc_data(shift_targets[record.relative_path], corrected, raw_y)
            created_shift.append(shift_targets[record.relative_path])

        # 只重新审核受影响的普通文件夹；小球目录在扫描阶段已排除。
        for date_name, folder_name in affected_folders:
            record = next(item for item in chosen if item.date_name == date_name and item.folder_name == folder_name)
            audit_started_dates.add(date_name)
            report = dict(
                run_folder_audit(
                    shift_base,
                    date_name=date_name,
                    folder_name=folder_name,
                    profile_id=_audit_profile_id(profile_catalog, record.genus),
                )
            )
            audit_reports.append(report)
            record_audit_result(
                root,
                date_name,
                folder_name,
                "applied" if report.get("status") == "applied" else str(report.get("status", "failed")),
                None if report.get("status") == "applied" else str(report.get("error", "audit_failed")),
            )
            if report.get("status") != "applied":
                raise RuntimeError(f"文件夹 audit 未通过：{date_name}/{folder_name}")

        # audit 后的 shift 文件夹整体替换到 init，确保被 audit 再次移除的候选不会进入 init。
        for init_folder in set(init_targets.values()):
            record = next(item for item in chosen if init_targets[item.relative_path] == init_folder)
            shift_folder = shift_base / record.date_name / record.folder_name
            temporary = create_temporary_path(init_folder)
            temporary.mkdir(parents=True, exist_ok=False)
            temporary_init.append(temporary)
            if shift_folder.is_dir():
                for source in sorted(shift_folder.iterdir(), key=lambda item: build_natural_key(item.name)):
                    if source.is_file() and source.suffix.lower() == ".arc_data":
                        target = temporary / source.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            if init_folder.exists():
                shutil.rmtree(init_folder)
            rename_directory(temporary, init_folder)
            published_init.append(init_folder)

        state = load_pipeline_state(state_path)
        affected_dates = {record.date_name for record in chosen}
        for record in chosen:
            state.cleaning.committed_paths.discard(record.relative_path)
            state.cleaning.marked_paths.discard(record.relative_path)
            state.cleaning.mark_reasons.pop(record.relative_path, None)
            state.cleaning.folder_catalog.setdefault(record.date_name, set()).add(record.folder_name)
            state.cleaning.completed_folder_keys.add(f"{record.date_name}/{record.folder_name}")
        state.dataset_builds.pop("profiles", None)
        full_report = state.dataset_builds.get("full_init")
        if isinstance(full_report, dict):
            updated_full = dict(full_report)
            updated_full["last_restore"] = {
                "file_count": len(chosen),
                "dates": sorted(affected_dates, key=build_natural_key),
                "paths": selected,
                "audit_reports": audit_reports,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            state.dataset_builds["full_init"] = updated_full
        save_pipeline_state(state_path, state)

        for path in source_raw.values():
            path.unlink()
        for parent in sorted(
            {path.parent for path in source_raw.values()},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()

        record_restore(root, selected)

        _append_log(
            Path(state_path).resolve().parent / RESTORE_LOG_NAME,
            {
                "operation": "restore",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "audit_reports": audit_reports,
                "items": [
                    {
                        "source_manual": str(source_raw[item.relative_path]),
                        "cosmic": str(raw_targets[item.relative_path]),
                        "shift": str(shift_targets[item.relative_path]),
                        "init": str(init_targets[item.relative_path] / item.file_name),
                        "date": item.date_name,
                        "genus": item.genus,
                        "prefix": item.prefix,
                        "folder": item.folder_name,
                    }
                    for item in chosen
                ],
            },
        )
        return {
            "status": "ready",
            "restored_file_count": len(chosen),
            "dates": sorted(affected_dates, key=build_natural_key),
            "paths": selected,
            "init_targets": sorted({str(path) for path in init_targets.values()}),
            "audit_reports": audit_reports,
        }
    except Exception:
        for path in created_raw + created_shift:
            path.unlink(missing_ok=True)
        for path in temporary_init:
            shutil.rmtree(path, ignore_errors=True)
        for folder, backup in shift_backups.items():
            _restore_directory_backup(folder, backup)
        for folder in published_init:
            _restore_directory_backup(folder, init_backups.get(folder))
        if audit_started_dates:
            state = load_pipeline_state(state_path)
            state.audit_completed_dates.difference_update(audit_started_dates)
            save_pipeline_state(state_path, state)
            for date_name, folder_name in affected_folders:
                record_audit_result(root, date_name, folder_name, "pending", "restore_rollback")
        raise
    finally:
        for backup in (*shift_backups.values(), *init_backups.values()):
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)


def restore_deleted_folder(
    data_root: Path | str,
    shift_root: Path | str,
    init_root: Path | str,
    state_path: Path | str,
    profile_catalog: DatasetMapping,
    source_id: str,
) -> dict[str, object]:
    """只恢复该次整夹删除记录中的文件，不恢复更早的独立删除谱。"""
    from ramanv2.pipeline.lineage import load_folder_lineage

    payload = load_folder_lineage(data_root)
    record = payload.get("folders", {}).get(str(source_id), {})
    files = record.get("files", {}) if isinstance(record, dict) else {}
    if not isinstance(files, dict):
        raise ValueError(f"没有可恢复的整夹删除记录：{source_id}")
    selected = []
    for relative, item in files.items():
        if not isinstance(item, dict) or item.get("status") != "manual_deleted":
            continue
        if item.get("reason") not in {"folder_remove", "date_remove", "prefix_folder_remove"}:
            continue
        selected.append(f"{source_id}/{relative}")
    if not selected:
        raise ValueError(f"没有可恢复的整夹删除记录：{source_id}")
    return restore_deleted_spectra(
        data_root,
        shift_root,
        init_root,
        state_path,
        profile_catalog,
        selected,
    )


__all__ = ["DeletedSpectrum", "RESTORE_LOG_NAME", "restore_deleted_folder", "restore_deleted_spectra", "scan_deleted_spectra"]
