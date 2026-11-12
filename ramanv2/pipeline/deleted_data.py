"""从人工清洗删除目录恢复普通光谱。"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ramanv2.common.arc_data import (
    count_arc_data,
    list_arc_data_paths,
    read_arc_data,
)
from ramanv2.common.naming import (
    BEAD_FOLDER_NAME,
    FOLDER_DELETE_RELATIVE,
    MANUAL_DELETE_RELATIVE,
    build_natural_key,
    parse_folder_prefix,
)
from ramanv2.data.calibration.affine import invalidate_calibration_report
from ramanv2.data.dataset_mapping import DatasetMapping, build_prefix_to_genus
from ramanv2.pipeline.cleaning import discard_removed_date
from ramanv2.records.lineage import (
    record_purge,
    record_restore,
)
from ramanv2.pipeline.state import invalidate_downstream, load_pipeline_state, write_pipeline_state


RESTORE_LOG_NAME = "restore_log.jsonl"


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


@dataclass(frozen=True)
class DeletedFolder:
    """一个整夹删除归档目录。"""

    relative_path: str
    source_relative_path: str
    date_name: str
    folder_name: str
    file_count: int


def _sub_directories(path: Path) -> list[os.DirEntry[str]]:
    """列出目录的直接子目录。

    用 ``os.scandir`` 而非 ``Path.iterdir`` + ``is_dir()``：Windows 上 ``DirEntry`` 的类型
    来自目录枚举结果，不产生逐项 ``stat``，整树扫描比 ``pathlib`` 快约 8 倍。
    """
    with os.scandir(path) as iterator:
        return [entry for entry in iterator if entry.is_dir()]


def scan_deleted_spectra(
    data_root: Path | str,
    dataset_mapping: DatasetMapping,
    species_name: str | None = None,
    date_name: str | None = None,
    include_bead: bool = False,
) -> list[DeletedSpectrum]:
    """只扫描 ``cosmic_data/delete/manual``，并忽略小球删除目录。"""
    root = Path(data_root).resolve()
    manual_root = root / MANUAL_DELETE_RELATIVE.as_posix()
    if not manual_root.is_dir():
        return []
    prefix_genus = build_prefix_to_genus(dataset_mapping)
    records: list[DeletedSpectrum] = []
    for date_entry in sorted(_sub_directories(manual_root), key=lambda entry: build_natural_key(entry.name)):
        if date_name is not None and date_entry.name != str(date_name):
            continue
        if not date_entry.name.isdigit():
            continue
        for folder_entry in sorted(
            _sub_directories(Path(date_entry.path)),
            key=lambda entry: build_natural_key(entry.name),
        ):
            folder_name = folder_entry.name
            if folder_name == BEAD_FOLDER_NAME and not include_bead:
                continue
            prefix = parse_folder_prefix(folder_name, uppercase_enable=True)
            genus = prefix_genus.get(prefix, "")
            if species_name is not None and species_name not in {genus, prefix, folder_name}:
                continue
            for source in list_arc_data_paths(
                Path(folder_entry.path),
                key=lambda path: build_natural_key(path.name),
            ):
                relative = "/".join((date_entry.name, folder_name, source.name))
                records.append(
                    DeletedSpectrum(
                        relative_path=relative,
                        source_relative_path=f"{MANUAL_DELETE_RELATIVE.as_posix()}/{relative}",
                        date_name=date_entry.name,
                        folder_name=folder_name,
                        file_name=source.name,
                        genus=genus,
                        prefix=prefix,
                    )
                )
    return records


def scan_deleted_folders(data_root: Path | str) -> list[DeletedFolder]:
    """只扫描 ``cosmic_data/delete/folder_delete`` 中的完整文件夹。"""
    root = Path(data_root).resolve() / FOLDER_DELETE_RELATIVE
    if not root.is_dir():
        return []
    records: list[DeletedFolder] = []
    for date_entry in sorted(
        (entry for entry in _sub_directories(root) if entry.name.isdigit()),
        key=lambda entry: build_natural_key(entry.name),
    ):
        for folder_entry in sorted(
            _sub_directories(Path(date_entry.path)),
            key=lambda entry: build_natural_key(entry.name),
        ):
            file_count = count_arc_data(Path(folder_entry.path), recursive=True)
            if not file_count:
                continue
            relative = f"{date_entry.name}/{folder_entry.name}"
            records.append(
                DeletedFolder(
                    relative_path=relative,
                    source_relative_path=f"{FOLDER_DELETE_RELATIVE.as_posix()}/{relative}",
                    date_name=date_entry.name,
                    folder_name=folder_entry.name,
                    file_count=file_count,
                )
            )
    return records


def _append_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _prune_empty_directories(directories: set[Path], root: Path) -> None:
    """删除移动光谱后留下的空文件夹，保留根目录。"""
    for parent in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        while parent != root and root in parent.parents:
            if not parent.is_dir() or any(parent.iterdir()):
                break
            parent.rmdir()
            parent = parent.parent


def _restore_bead_spectra(
    data_root: Path,
    shift_root: Path,
    state_path: Path | str,
    records: list[DeletedSpectrum],
) -> dict[str, object]:
    """恢复人工删除的小球谱，供后续仿射校准使用。

    小球谱变化意味着日期级仿射参数必须重算，因此这里把 ``affine.json`` 置为 pending；
    普通谱的恢复不做这一步。
    """
    sources = {record.relative_path: data_root / MANUAL_DELETE_RELATIVE.as_posix() / record.relative_path for record in records}
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
            state.cleaning.folder_catalog.setdefault(record.date_name, set()).add(BEAD_FOLDER_NAME)
            state.cleaning.completed_folder_keys.add(f"{record.date_name}/{BEAD_FOLDER_NAME}")
        # 小球谱变化后旧仿射参数不再适用，日期回到待校准队列。
        for date_name in dates:
            invalidate_calibration_report(shift_root, date_name)
            state.affine_completed_dates.discard(date_name)
        write_pipeline_state(state_path, state)

        for source in sources.values():
            source.unlink()
        _prune_empty_directories({source.parent for source in sources.values()}, data_root)
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
                        "folder": BEAD_FOLDER_NAME,
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
            "targets": sorted(
                {f"{record.date_name}/{BEAD_FOLDER_NAME}" for record in records},
                key=build_natural_key,
            ),
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
    dataset_mapping: DatasetMapping,
    selected_relative_paths: list[str] | tuple[str, ...] | set[str],
) -> dict[str, object]:
    """把 cosmic 删除区的普通谱移回原位。

    只改 cosmic 与清洗状态：不写 shift、不改仿射参数、不触发审核，审核结论保持原样。
    该文件夹的清洗完成状态保持不变；shift 由仿射校准页的重建入口从 cosmic 整体生成。
    """
    del init_root  # 恢复只作用于 cosmic_data；init 由数据集获取页按需重建。
    root = Path(data_root).resolve()
    shift_base = Path(shift_root).resolve()
    normalized = set()
    for item in selected_relative_paths:
        path = str(item).replace("\\", "/")
        prefix = f"{MANUAL_DELETE_RELATIVE.as_posix()}/"
        normalized.add(path[len(prefix):] if path.startswith(prefix) else path)
    selected = sorted(normalized, key=build_natural_key)
    if not selected:
        raise ValueError("至少选择一条需要恢复的光谱")
    prefix_genus = build_prefix_to_genus(dataset_mapping)
    records = {
        item.relative_path: item
        for item in scan_deleted_spectra(root, dataset_mapping, include_bead=True)
    }
    missing = [path for path in selected if path not in records]
    if missing:
        raise FileNotFoundError(f"人工删除光谱不存在：{missing}")
    chosen = [records[path] for path in selected]
    bead_records = [record for record in chosen if record.folder_name == BEAD_FOLDER_NAME]
    if bead_records:
        if len(bead_records) != len(chosen):
            raise ValueError("小球光谱不能与普通采集光谱混合恢复")
        return _restore_bead_spectra(root, shift_base, state_path, chosen)
    affected_dates = {record.date_name for record in chosen}

    source_raw: dict[str, Path] = {}
    raw_targets: dict[str, Path] = {}
    for record in chosen:
        if not record.genus or prefix_genus.get(record.prefix) != record.genus:
            raise ValueError(f"删除文件夹前缀未映射：{record.folder_name}")
        raw_source = root / MANUAL_DELETE_RELATIVE.as_posix() / record.relative_path
        raw_target = root / record.relative_path
        if not raw_source.is_file():
            raise FileNotFoundError(f"人工删除光谱不存在：{raw_source}")
        if raw_target.exists():
            raise FileExistsError(f"原始目标已存在：{raw_target}")
        source_raw[record.relative_path] = raw_source
        raw_targets[record.relative_path] = raw_target

    created_raw: list[Path] = []
    try:
        # 只从 cosmic_data/delete/manual 把原始谱移回原位，不派生任何 shift 输出。
        for record in chosen:
            raw_x, raw_y = read_arc_data(source_raw[record.relative_path])
            if not raw_x.size or raw_x.size != raw_y.size:
                raise ValueError(f"无效光谱：{record.relative_path}")
            raw_targets[record.relative_path].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_raw[record.relative_path], raw_targets[record.relative_path])
            created_raw.append(raw_targets[record.relative_path])

        record_restore(root, selected)

        state = load_pipeline_state(state_path)
        for record in chosen:
            state.cleaning.committed_paths.discard(record.relative_path)
            state.cleaning.marked_paths.discard(record.relative_path)
            state.cleaning.mark_reasons.pop(record.relative_path, None)
            state.cleaning.folder_catalog.setdefault(record.date_name, set()).add(record.folder_name)
        # 恢复不回退清洗进度、不作废审核结论：只标记 init 待重建。
        invalidate_downstream(state, affected_dates)
        full_report = state.dataset_builds.get("full_init")
        if isinstance(full_report, dict):
            updated_full = dict(full_report)
            updated_full["last_restore"] = {
                "file_count": len(chosen),
                "dates": sorted(affected_dates, key=build_natural_key),
                "paths": selected,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            state.dataset_builds["full_init"] = updated_full
        write_pipeline_state(state_path, state)

        for path in source_raw.values():
            path.unlink()
        _prune_empty_directories({path.parent for path in source_raw.values()}, root)

        _append_log(
            Path(state_path).resolve().parent / RESTORE_LOG_NAME,
            {
                "operation": "restore",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "items": [
                    {
                        "source_manual": str(source_raw[item.relative_path]),
                        "cosmic": str(raw_targets[item.relative_path]),
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
            "targets": sorted(
                {f"{record.date_name}/{record.folder_name}" for record in chosen},
                key=build_natural_key,
            ),
        }
    except Exception:
        for path in created_raw:
            path.unlink(missing_ok=True)
        raise


def purge_deleted_spectra(
    data_root: Path | str,
    state_path: Path | str,
    selected_relative_paths: list[str] | tuple[str, ...] | set[str],
) -> dict[str, object]:
    """从人工删除区永久删除所选光谱，并清理因此变空的目录。

    这是不可恢复操作：文件直接 unlink，随后把走向记录标记为 purged、从清洗提交记录中
    移除。空文件夹会自下而上删除，直到 data_root 为止；活动区不受影响。
    """
    root = Path(data_root).resolve()
    manual_root = root / MANUAL_DELETE_RELATIVE.as_posix()
    normalized = set()
    for item in selected_relative_paths:
        path = str(item).replace("\\", "/")
        prefix = f"{MANUAL_DELETE_RELATIVE.as_posix()}/"
        normalized.add(path[len(prefix):] if path.startswith(prefix) else path)
    selected = sorted(normalized, key=build_natural_key)
    if not selected:
        raise ValueError("至少选择一条需要彻底删除的光谱")
    paths: dict[str, Path] = {}
    for relative in selected:
        path = (manual_root / relative).resolve()
        if not path.is_relative_to(manual_root):
            raise ValueError(f"拒绝处理删除区之外的路径：{relative}")
        if not path.is_file():
            raise FileNotFoundError(f"人工删除光谱不存在：{path}")
        paths[relative] = path

    record_purge(root, selected)
    state = load_pipeline_state(state_path)
    for relative in selected:
        state.cleaning.committed_paths.discard(relative)
    write_pipeline_state(state_path, state)

    for path in paths.values():
        path.unlink()
    _prune_empty_directories({path.parent for path in paths.values()}, root)

    # 整日期已无任何数据（活动或可恢复）时，仿射结论与清洗记录一并作废，
    # 否则该日期会以“待增量建 init”的形式残留在页面上。
    dates = sorted({path.split("/", 1)[0] for path in selected}, key=build_natural_key)
    emptied = [date_name for date_name in dates if not (root / date_name).exists()]
    if emptied:
        for date_name in emptied:
            discard_removed_date(state.cleaning, date_name)
            state.affine_completed_dates.discard(date_name)
        write_pipeline_state(state_path, state)

    _append_log(
        Path(state_path).resolve().parent / RESTORE_LOG_NAME,
        {
            "operation": "purge",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "items": [
                {
                    "source_manual": str(paths[relative]),
                    "date": relative.split("/", 1)[0],
                    "folder": relative.split("/")[1],
                    "file": "/".join(relative.split("/")[2:]),
                }
                for relative in selected
            ],
        },
    )
    return {
        "status": "purged",
        "purged_file_count": len(selected),
        "dates": dates,
        "paths": selected,
    }


def restore_deleted_folder(
    data_root: Path | str,
    shift_root: Path | str,
    init_root: Path | str,
    state_path: Path | str,
    dataset_mapping: DatasetMapping,
    source_id: str,
) -> dict[str, object]:
    """把整夹删除归档整体移回 cosmic_data，并让该文件夹重新进入清洗队列。

    只改 cosmic 与清洗状态：不写 shift、不作废审核结论。shift 由仿射校准页的重建入口
    按已确认参数生成（重建时才作废该日期审核结论），init 由数据集获取页重建。
    """
    del init_root, dataset_mapping  # 整夹恢复不写 shift、不写 init。
    root = Path(data_root).resolve()
    parts = str(source_id).replace("\\", "/").split("/", 1)
    if len(parts) != 2:
        raise ValueError("source_id 必须是 日期/文件夹")
    date_name, folder_name = parts
    archive = root / FOLDER_DELETE_RELATIVE / date_name / folder_name
    target = root / date_name / folder_name
    if not archive.is_dir():
        raise FileNotFoundError(f"整夹删除归档不存在：{archive}")
    if target.exists():
        raise FileExistsError(f"源文件夹目标已存在：{target}")
    files = list_arc_data_paths(
        archive,
        recursive=True,
        key=lambda item: build_natural_key(item.relative_to(archive).as_posix()),
    )
    if not files:
        raise ValueError(f"整夹删除归档没有光谱：{archive}")
    relative_paths = [
        f"{date_name}/{folder_name}/{item.relative_to(archive).as_posix()}"
        for item in files
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(archive), str(target))
    try:
        record_restore(root, relative_paths)
        state = load_pipeline_state(state_path)
        state.cleaning.folder_catalog.setdefault(date_name, set()).add(folder_name)
        # 整夹恢复后文件夹内容已变，必须重新清洗确认；审核结论保持原样（审核只做一次）。
        state.cleaning.completed_folder_keys.discard(f"{date_name}/{folder_name}")
        invalidate_downstream(state, [date_name])
        write_pipeline_state(state_path, state)
    except Exception:
        if target.exists() and not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(archive))
        raise
    return {
        "status": "folder_restored",
        "source_id": source_id,
        "restored_file_count": len(files),
        "dates": [date_name],
        "paths": relative_paths,
        "targets": [f"{date_name}/{folder_name}"],
    }


__all__ = [
    "DeletedFolder",
    "DeletedSpectrum",
    "RESTORE_LOG_NAME",
    "purge_deleted_spectra",
    "restore_deleted_folder",
    "restore_deleted_spectra",
    "scan_deleted_folders",
    "scan_deleted_spectra",
]
