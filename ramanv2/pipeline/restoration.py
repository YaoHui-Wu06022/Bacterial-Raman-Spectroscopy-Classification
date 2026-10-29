"""从人工清洗删除目录恢复普通光谱。

恢复只作用于 ``cosmic_data`` 与清洗状态：源谱写回原位、受影响文件夹按需回到清洗队列。
shift_data 与 init 都是可丢弃的派生数据，由各自的「重建」入口从 cosmic_data 整体生成，
恢复流程不写 shift、不改仿射参数、不作废审核结论。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.naming import build_natural_key, parse_folder_prefix
from ramanv2.data.calibration.affine import invalidate_calibration_report
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.invalidation import invalidate_downstream
from ramanv2.pipeline.lineage import (
    FOLDER_DELETE_RELATIVE,
    record_restore,
)
from ramanv2.pipeline.state import load_pipeline_state, save_pipeline_state


MANUAL_DELETE_DIR = "delete/manual"
FOLDER_DELETE_DIR = FOLDER_DELETE_RELATIVE.as_posix()
RESTORE_LOG_NAME = "restore_log.jsonl"
BEAD_DIR_NAME = "小球"
ARC_DATA_SUFFIX = ".arc_data"


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


def _prefix_to_genus(mapping: DatasetMapping) -> dict[str, str]:
    return {
        prefix.upper(): genus
        for genus, prefixes in mapping.genera.items()
        for prefix in prefixes
    }


def _sub_directories(path: Path) -> list[os.DirEntry[str]]:
    """列出目录的直接子目录。

    用 ``os.scandir`` 而非 ``Path.iterdir`` + ``is_dir()``：Windows 上 ``DirEntry`` 的类型
    来自目录枚举结果，不产生逐项 ``stat``，整树扫描比 ``pathlib`` 快约 8 倍。
    """
    with os.scandir(path) as iterator:
        return [entry for entry in iterator if entry.is_dir()]


def _arc_data_entries(path: Path) -> list[os.DirEntry[str]]:
    """列出目录中直接的 ``.arc_data`` 文件。"""
    with os.scandir(path) as iterator:
        return [
            entry
            for entry in iterator
            if entry.is_file() and entry.name.lower().endswith(ARC_DATA_SUFFIX)
        ]


def _count_arc_data(path: Path) -> int:
    """递归统计目录下的 ``.arc_data`` 数量，等价 ``rglob("*.arc_data")`` + ``is_file``。"""
    total = 0
    pending = [path]
    while pending:
        try:
            with os.scandir(pending.pop()) as iterator:
                for entry in iterator:
                    if entry.is_dir():
                        pending.append(Path(entry.path))
                    elif entry.is_file() and entry.name.lower().endswith(ARC_DATA_SUFFIX):
                        total += 1
        except OSError:
            continue
    return total


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
            if folder_name == BEAD_DIR_NAME and not include_bead:
                continue
            prefix = parse_folder_prefix(folder_name, uppercase_enable=True)
            genus = prefix_genus.get(prefix, "")
            if species_name is not None and species_name not in {genus, prefix, folder_name}:
                continue
            for source in sorted(
                _arc_data_entries(Path(folder_entry.path)),
                key=lambda entry: build_natural_key(entry.name),
            ):
                relative = "/".join((date_entry.name, folder_name, source.name))
                records.append(
                    DeletedSpectrum(
                        relative_path=relative,
                        source_relative_path=f"{MANUAL_DELETE_DIR}/{relative}",
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
            file_count = _count_arc_data(Path(folder_entry.path))
            if not file_count:
                continue
            relative = f"{date_entry.name}/{folder_entry.name}"
            records.append(
                DeletedFolder(
                    relative_path=relative,
                    source_relative_path=f"{FOLDER_DELETE_DIR}/{relative}",
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
        # 小球谱变化后旧仿射参数不再适用，日期回到待校准队列。
        for date_name in dates:
            invalidate_calibration_report(shift_root, date_name)
            state.affine_completed_dates.discard(date_name)
        save_pipeline_state(state_path, state)

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
            "targets": sorted(
                {f"{record.date_name}/{BEAD_DIR_NAME}" for record in records},
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
    profile_catalog: DatasetMapping,
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
        return _restore_bead_spectra(root, shift_base, state_path, chosen)
    affected_dates = {record.date_name for record in chosen}

    source_raw: dict[str, Path] = {}
    raw_targets: dict[str, Path] = {}
    for record in chosen:
        if not record.genus or prefix_genus.get(record.prefix) != record.genus:
            raise ValueError(f"删除文件夹前缀未映射：{record.folder_name}")
        raw_source = root / MANUAL_DELETE_DIR / record.relative_path
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
        save_pipeline_state(state_path, state)

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


def restore_deleted_folder(
    data_root: Path | str,
    shift_root: Path | str,
    init_root: Path | str,
    state_path: Path | str,
    profile_catalog: DatasetMapping,
    source_id: str,
) -> dict[str, object]:
    """把整夹删除归档整体移回 cosmic_data，并让该文件夹重新进入清洗队列。

    只改 cosmic 与清洗状态：不写 shift、不作废审核结论。shift 由仿射校准页的重建入口
    按已确认参数生成（重建时才作废该日期审核结论），init 由数据集获取页重建。
    """
    del init_root, profile_catalog  # 整夹恢复不写 shift、不写 init。
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
    files = sorted(
        (item for item in archive.rglob("*.arc_data") if item.is_file()),
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
        save_pipeline_state(state_path, state)
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
    "restore_deleted_folder",
    "restore_deleted_spectra",
    "scan_deleted_folders",
    "scan_deleted_spectra",
]
