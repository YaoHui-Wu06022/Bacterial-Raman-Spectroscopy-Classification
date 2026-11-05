"""数据清洗流水线的增量扫描、标记和可恢复移动。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ramanv2.common.arc_data import iter_arc_data_paths
from ramanv2.common.naming import PIPELINE_RESERVED_FOLDER_NAMES, build_natural_key
from ramanv2.common.file_ops import (
    move_files,
    preserve_paths,
    remove_empty_parent_directories,
)
from ramanv2.pipeline.lineage import (
    initialize_folder_lineage,
    lineage_path,
    record_manual_move,
    remove_source_folder,
)
from ramanv2.pipeline.state import COMMIT_LOG_NAME, CleaningState, PipelineState, invalidate_downstream, write_pipeline_state


DATE_PATTERN = re.compile(r"^\d{8}$")
SPECIAL_FOLDER_NAMES = PIPELINE_RESERVED_FOLDER_NAMES
BEAD_FOLDER_NAME = "小球"


@dataclass(frozen=True)
class ReviewFolder:
    key: str
    date_name: str
    folder_name: str
    relative_paths: tuple[str, ...]


@dataclass(frozen=True)
class SpectrumItem:
    relative_path: str
    date_name: str
    folder_name: str
    file_name: str


@dataclass(frozen=True)
class CommitResult:
    moved_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]


def _parse_date_name(name: str) -> datetime | None:
    if not DATE_PATTERN.fullmatch(name):
        return None
    try:
        return datetime.strptime(name, "%Y%m%d")
    except ValueError:
        return None


def scan_date_names(root_dir: Path | str) -> list[str]:
    """只读取日期目录名称，供每次页面刷新进行轻量发现。"""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"光谱根目录不存在：{root}")
    dates = []
    for path in root.iterdir():
        parsed = _parse_date_name(path.name)
        if path.is_dir() and parsed is not None:
            dates.append((parsed, path.name))
    return [name for _, name in sorted(dates)]


def _collect_paths(root: Path, folder_dir: Path) -> tuple[str, ...]:
    if not folder_dir.is_dir():
        return ()
    names = sorted(
        (path.name for path in iter_arc_data_paths(folder_dir)),
        key=build_natural_key,
    )
    prefix = folder_dir.relative_to(root).as_posix()
    return tuple(f"{prefix}/{name}" for name in names)


def scan_review_folders(
    root_dir: Path | str,
    date_names: list[str] | set[str] | tuple[str, ...] | None = None,
    folder_catalog: dict[str, set[str]] | None = None,
    completed_folder_keys: set[str] | None = None,
) -> list[ReviewFolder]:
    """只扫描传入日期的直接子目录；目录清单可恢复已被移走的文件夹。"""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"光谱根目录不存在：{root}")
    catalog = folder_catalog or {}
    selected_dates = set(date_names) if date_names is not None else set(scan_date_names(root))
    completed = completed_folder_keys or set()
    folders = []
    for date_name in sorted(selected_dates):
        date_dir = root / date_name
        physical_names = (
            {
                path.name
                for path in date_dir.iterdir()
                if path.is_dir() and path.name not in SPECIAL_FOLDER_NAMES
            }
            if date_dir.is_dir()
            else set()
        )
        names = physical_names | set(catalog.get(date_name, set()))
        for folder_name in sorted(names, key=build_natural_key):
            key = f"{date_name}/{folder_name}"
            # 已完成文件夹不再列举光谱：全库门禁检查会对所有日期调用本函数。
            if key in completed:
                continue
            folders.append(
                ReviewFolder(
                    key=key,
                    date_name=date_name,
                    folder_name=folder_name,
                    relative_paths=_collect_paths(root, date_dir / folder_name),
                )
            )
    return folders


def sync_folder_catalog(state: CleaningState, folders: list[ReviewFolder]) -> None:
    for folder in folders:
        state.folder_catalog.setdefault(folder.date_name, set()).add(folder.folder_name)


def refresh_completed_dates(state: CleaningState) -> None:
    state.completed_date_names = {
        date_name
        for date_name, folder_names in state.folder_catalog.items()
        if folder_names and all(
            f"{date_name}/{folder_name}" in state.completed_folder_keys
            for folder_name in folder_names
        )
    }


def discard_removed_folder(state: CleaningState, folder: ReviewFolder) -> None:
    """移除已物理删除文件夹的清洗记录，避免目录清单继续制造待审核项。"""
    folder_names = state.folder_catalog.get(folder.date_name)
    if folder_names is not None:
        folder_names.discard(folder.folder_name)
        if not folder_names:
            state.folder_catalog.pop(folder.date_name, None)
    key = folder.key
    state.completed_folder_keys.discard(key)
    state.folder_page_indices.pop(key, None)
    prefix = f"{key}/"
    state.marked_paths = {path for path in state.marked_paths if not path.startswith(prefix)}
    state.mark_reasons = {
        path: reasons
        for path, reasons in state.mark_reasons.items()
        if not path.startswith(prefix)
    }
    refresh_completed_dates(state)


def reopen_date_review(
    state: CleaningState,
    date_name: str,
    folders: list[ReviewFolder],
) -> None:
    """撤销整日审核完成状态，让新增文件夹和光谱重新进入审核队列。"""
    prefix = f"{date_name}/"
    folder_names = {folder.folder_name for folder in folders}
    if folder_names:
        state.folder_catalog[date_name] = folder_names
    else:
        state.folder_catalog.pop(date_name, None)
    state.completed_folder_keys = {
        key for key in state.completed_folder_keys if not key.startswith(prefix)
    }
    state.folder_page_indices = {
        key: value
        for key, value in state.folder_page_indices.items()
        if not key.startswith(prefix)
    }
    state.marked_paths = {path for path in state.marked_paths if not path.startswith(prefix)}
    state.mark_reasons = {
        path: reasons
        for path, reasons in state.mark_reasons.items()
        if not path.startswith(prefix)
    }
    state.completed_date_names.discard(date_name)
    state.current_date_name = date_name if folder_names else None
    state.current_folder_key = None
    refresh_completed_dates(state)


def discard_removed_date(state: CleaningState, date_name: str) -> None:
    """移除已物理删除日期的全部清洗记录。"""
    state.folder_catalog.pop(date_name, None)
    prefix = f"{date_name}/"
    state.completed_folder_keys = {
        key for key in state.completed_folder_keys if not key.startswith(prefix)
    }
    state.folder_page_indices = {
        key: value
        for key, value in state.folder_page_indices.items()
        if not key.startswith(prefix)
    }
    state.marked_paths = {path for path in state.marked_paths if not path.startswith(prefix)}
    state.mark_reasons = {
        path: reasons
        for path, reasons in state.mark_reasons.items()
        if not path.startswith(prefix)
    }
    state.completed_date_names.discard(date_name)
    refresh_completed_dates(state)


def reconcile_cleaning_records(root_dir: Path | str, state: CleaningState) -> bool:
    """按当前磁盘清理已删除日期和文件夹的历史审核记录。"""
    root = Path(root_dir).resolve()
    changed = False
    for date_name in list(state.folder_catalog):
        date_dir = root / date_name
        if not date_dir.is_dir():
            discard_removed_date(state, date_name)
            changed = True
            continue
        physical_names = {
            path.name
            for path in date_dir.iterdir()
            if path.is_dir() and path.name not in SPECIAL_FOLDER_NAMES
        }
        for folder_name in list(state.folder_catalog.get(date_name, set())):
            if folder_name not in physical_names:
                discard_removed_folder(
                    state,
                    ReviewFolder(f"{date_name}/{folder_name}", date_name, folder_name, ()),
                )
                changed = True
    before_dates = set(state.completed_date_names)
    refresh_completed_dates(state)
    return changed or before_dates != state.completed_date_names


def mark_paths(state: CleaningState, paths: set[str] | list[str] | tuple[str, ...], reason: str) -> None:
    for path in paths:
        normalized = str(path).replace("\\", "/")
        state.marked_paths.add(normalized)
        state.mark_reasons.setdefault(normalized, set()).add(reason)


def unmark_paths(state: CleaningState, paths: set[str] | list[str] | tuple[str, ...]) -> None:
    for path in paths:
        normalized = str(path).replace("\\", "/")
        state.marked_paths.discard(normalized)
        state.mark_reasons.pop(normalized, None)


def toggle_mark_path(state: CleaningState, relative_path: str) -> bool:
    """切换单条光谱的待移除状态，返回切换后的状态。"""
    normalized = str(relative_path).replace("\\", "/")
    if normalized in state.marked_paths:
        unmark_paths(state, [normalized])
        return False
    mark_paths(state, [normalized], "manual_selection")
    return True


def folder_has_pending(state: CleaningState, folder: ReviewFolder) -> bool:
    return bool(state.marked_paths.intersection(folder.relative_paths))


def complete_folder(state: CleaningState, folder: ReviewFolder) -> None:
    if folder_has_pending(state, folder):
        raise ValueError("该文件夹还有未提交的待移除光谱")
    state.folder_catalog.setdefault(folder.date_name, set()).add(folder.folder_name)
    state.completed_folder_keys.add(folder.key)
    state.current_folder_key = None
    refresh_completed_dates(state)


def _remove_empty_parents(source: Path, root: Path) -> None:
    remove_empty_parent_directories(source.parent, root)


def commit_paths(
    root_dir: Path | str,
    state: PipelineState,
    state_path: Path | str,
    paths: set[str] | list[str] | tuple[str, ...],
    reason: str | None = None,
) -> CommitResult:
    """把指定光谱移入可恢复目录，并立刻保存状态和提交日志。"""
    root = Path(root_dir).resolve()
    target_root = root / "delete" / "manual"
    normalized = sorted({str(path).replace("\\\\", "/") for path in paths})
    moved, missing, moves = [], [], {}
    purged: list[str] = []
    for relative in normalized:
        source, target = (root / relative).resolve(), (target_root / relative).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(target_root) or source.is_relative_to(root / "delete"):
            raise ValueError(f"拒绝处理源目录之外的路径：{relative}")
        if source.is_file():
            # 小球谱只服务日期级仿射，删掉即永久删除，不进入人工删除区。
            parts = relative.split("/")
            if len(parts) >= 2 and parts[1] == BEAD_FOLDER_NAME:
                purged.append(relative)
            else:
                moves[source] = target
                moved.append(relative)
        elif not target.is_file():
            missing.append(relative)
    dates = {relative.split("/", 1)[0] for relative in moved + purged}
    snapshots = [lineage_path(root), Path(state_path), Path(state_path).parent / COMMIT_LOG_NAME]
    snapshots.extend(root / date for date in dates)
    snapshots.extend(target_root / date for date in dates)
    working = PipelineState.from_payload(state.to_payload())
    cleaning = working.cleaning
    with preserve_paths(snapshots):
        initialize_folder_lineage(root)
        move_files(moves)
        if moved:
            record_manual_move(root, moved, reason or "manual_selection")
        # 小球谱直接永久删除，不写人工删除记录；失败时由日期级快照整体回滚。
        for relative in purged:
            (root / relative).unlink()
            _remove_empty_parents(root / relative, root)
        reasons = {}
        purged_set = set(purged)
        for relative in normalized:
            if relative in missing:
                continue
            cleaning.marked_paths.discard(relative)
            if relative in purged_set:
                cleaning.committed_paths.discard(relative)
            else:
                cleaning.committed_paths.add(relative)
            path_reasons = set(cleaning.mark_reasons.pop(relative, set()))
            if reason:
                path_reasons.add(reason)
            reasons[relative] = sorted(path_reasons)
        for relative in moved:
            _remove_empty_parents(root / relative, root)
        invalidate_downstream(working, dates)
        refresh_completed_dates(cleaning)
        write_pipeline_state(state_path, working)
        if moved or purged:
            log_path = Path(state_path).parent / COMMIT_LOG_NAME
            with log_path.open("a", encoding="utf-8") as file:
                timestamp = datetime.now().astimezone().isoformat()
                payload = {"committed_at": timestamp, "paths": moved, "reasons": reasons}
                if purged:
                    payload["purged_paths"] = purged
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    state.__dict__.update(working.__dict__)
    missing_set = set(missing)
    return CommitResult(tuple(relative for relative in normalized if relative not in missing_set), tuple(missing))


def commit_folder_items(
    root_dir: Path | str,
    state: PipelineState,
    state_path: Path | str,
    folder: ReviewFolder,
) -> CommitResult:
    """整夹归档到 cosmic/delete/folder_delete，只标记 init 待重建。"""
    root = Path(root_dir).resolve()
    working = PipelineState.from_payload(state.to_payload())
    source = root / folder.date_name / folder.folder_name
    manual = root / "delete" / "manual" / folder.date_name / folder.folder_name
    archive = root / "delete" / "folder_delete" / folder.date_name / folder.folder_name
    if any(not path.resolve().is_relative_to(root) for path in (source, manual, archive)):
        raise ValueError("删除路径超出 cosmic 目录")
    with preserve_paths([source, manual, archive, lineage_path(root), Path(state_path)]):
        result = remove_source_folder(root, folder.key)
        paths = tuple(result["moved_paths"])
        working.cleaning.marked_paths.difference_update(paths)
        working.cleaning.committed_paths.update(paths)
        for path in paths:
            working.cleaning.mark_reasons.pop(path, None)
        discard_removed_folder(working.cleaning, folder)
        invalidate_downstream(working, [folder.date_name])
        write_pipeline_state(state_path, working)
    state.__dict__.update(working.__dict__)
    return CommitResult(paths, ())


def commit_date_items(
    root_dir: Path | str,
    state: PipelineState,
    state_path: Path | str,
    folders: list[ReviewFolder] | tuple[ReviewFolder, ...],
) -> CommitResult:
    """立即移除当前日期的普通与小球光谱：普通谱进人工删除区，小球谱直接永久删除。"""
    paths = [path for folder in folders for path in folder.relative_paths]
    mark_paths(state.cleaning, paths, "date_remove")
    result = commit_paths(root_dir, state, state_path, paths, reason="date_remove")
    root = Path(root_dir).resolve()
    date_names = {folder.date_name for folder in folders}
    for folder in folders:
        remove_empty_parent_directories(root / folder.date_name / folder.folder_name, root)
    for date_name in date_names:
        remove_empty_parent_directories(root / date_name, root)
        discard_removed_date(state.cleaning, date_name)
    # 整日期移除后该日期已消失，其 shift 输出由重建入口清理；审核结论保持原样。
    invalidate_downstream(state, date_names)
    # commit_paths 已经保存过移动记录；目录清理后再保存一次，确保状态落盘顺序完整。
    write_pipeline_state(state_path, state)
    return result


__all__ = [
    "CommitResult",
    "ReviewFolder",
    "SPECIAL_FOLDER_NAMES",
    "SpectrumItem",
    "commit_date_items",
    "commit_folder_items",
    "commit_paths",
    "complete_folder",
    "discard_removed_date",
    "discard_removed_folder",
    "folder_has_pending",
    "mark_paths",
    "refresh_completed_dates",
    "reconcile_cleaning_records",
    "reopen_date_review",
    "scan_date_names",
    "scan_review_folders",
    "sync_folder_catalog",
    "toggle_mark_path",
    "unmark_paths",
]
