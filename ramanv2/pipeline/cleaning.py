"""数据清洗流水线的增量扫描、标记和可恢复移动。"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ramanv2.common.naming import build_natural_key
from ramanv2.pipeline.lineage import (
    initialize_folder_lineage,
    load_folder_lineage,
    record_manual_move,
    record_restore,
    remove_downstream_files,
    remove_downstream_outputs,
)
from ramanv2.pipeline.state import COMMIT_LOG_NAME, CleaningState, PipelineState, save_pipeline_state


DATE_PATTERN = re.compile(r"^\d{8}$")
DEFAULT_CELL_PATTERN = re.compile(r"^cell[12](?:_|$)", re.IGNORECASE)
SPECIAL_FOLDER_NAMES = {"delete", "shift", "audit_runs", ".raman_pipeline"}


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
    paths = [path for path in folder_dir.glob("*.arc_data") if path.is_file()]
    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted(paths, key=lambda item: build_natural_key(item.name))
    )


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
            folders.append(
                ReviewFolder(
                    key=key,
                    date_name=date_name,
                    folder_name=folder_name,
                    relative_paths=_collect_paths(root, date_dir / folder_name),
                )
            )
    return [folder for folder in folders if folder.key not in completed]


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
    state.seeded_folder_keys.discard(key)
    state.folder_page_indices.pop(key, None)
    prefix = f"{key}/"
    state.marked_paths = {path for path in state.marked_paths if not path.startswith(prefix)}
    state.mark_reasons = {
        path: reasons
        for path, reasons in state.mark_reasons.items()
        if not path.startswith(prefix)
    }
    refresh_completed_dates(state)


def discard_removed_date(state: CleaningState, date_name: str) -> None:
    """移除已物理删除日期的全部清洗记录。"""
    state.folder_catalog.pop(date_name, None)
    prefix = f"{date_name}/"
    state.completed_folder_keys = {
        key for key in state.completed_folder_keys if not key.startswith(prefix)
    }
    state.seeded_folder_keys = {
        key for key in state.seeded_folder_keys if not key.startswith(prefix)
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


def seed_default_marks(folders: list[ReviewFolder], state: CleaningState) -> int:
    """进入文件夹时仅为该文件夹标记 cell1/cell2，跳过小球目录。"""
    count = 0
    for folder in folders:
        if folder.key in state.seeded_folder_keys or folder.folder_name == "小球":
            continue
        matched = [
            path
            for path in folder.relative_paths
            if DEFAULT_CELL_PATTERN.match(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        ]
        mark_paths(state, matched, "default_cell1_cell2")
        state.seeded_folder_keys.add(folder.key)
        count += len(matched)
    return count


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
    _remove_empty_directories(source.parent, root)


def _remove_empty_directories(directory: Path, root: Path) -> None:
    """删除移动光谱后留下的空文件夹，兼容采集目录的只读属性。"""
    parent = directory
    while parent != root and root in parent.parents:
        if not parent.is_dir():
            parent = parent.parent
            continue
        try:
            has_entries = any(parent.iterdir())
        except PermissionError:
            parent.chmod(0o700)
            has_entries = any(parent.iterdir())
        if has_entries:
            break
        try:
            parent.rmdir()
        except PermissionError:
            # 采集目录可能带有 Windows 只读属性，先清除后再删除。
            parent.chmod(0o700)
            parent.rmdir()
        parent = parent.parent


def commit_paths(
    root_dir: Path | str,
    state: PipelineState,
    state_path: Path | str,
    paths: set[str] | list[str] | tuple[str, ...],
    reason: str | None = None,
) -> CommitResult:
    """把指定光谱移入可恢复目录，并立刻保存状态和提交日志。"""
    cleaning = state.cleaning
    root = Path(root_dir).resolve()
    # 每次源头操作前刷新走向记录，确保历史删除和当前活动文件都可追踪。
    initialize_folder_lineage(root)
    target_root = root / "delete" / "manual"
    moved: list[str] = []
    missing: list[str] = []
    reasons: dict[str, list[str]] = {}
    cleanup_sources: list[Path] = []

    normalized_paths = sorted({str(path).replace("\\", "/") for path in paths})
    lineage = load_folder_lineage(root)
    folders = lineage.get("folders", {})
    if isinstance(folders, dict):
        approved_dates = {
            str(record.get("date", ""))
            for relative_path in normalized_paths
            for record in [folders.get("/".join(relative_path.split("/")[:2]), {})]
            if isinstance(record, dict)
            and isinstance(record.get("affine"), dict)
            and record["affine"].get("status") == "approved"
        }
        if approved_dates:
            raise RuntimeError(
                "日期已完成仿射校正，清洗页不能继续修改源谱；请使用 prefix 图的源头整夹删除入口："
                + ", ".join(sorted(approved_dates, key=build_natural_key))
            )
    for relative_path in normalized_paths:
        source = (root / relative_path).resolve()
        target = (target_root / relative_path).resolve()
        if root not in source.parents or root not in target.parents:
            raise ValueError(f"拒绝处理根目录之外的路径：{relative_path}")
        if not source.is_file():
            if target.is_file():
                cleaning.marked_paths.discard(relative_path)
                cleaning.committed_paths.add(relative_path)
                reasons[relative_path] = sorted(cleaning.mark_reasons.pop(relative_path, set()))
                cleanup_sources.append(source)
            else:
                missing.append(relative_path)
            continue
        if target.exists():
            raise FileExistsError(f"移除目标已存在：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(relative_path)
        cleaning.marked_paths.discard(relative_path)
        cleaning.committed_paths.add(relative_path)
        path_reasons = set(cleaning.mark_reasons.pop(relative_path, set()))
        if reason:
            path_reasons.add(reason)
        reasons[relative_path] = sorted(path_reasons)
        cleanup_sources.append(source)

    for source in cleanup_sources:
        _remove_empty_parents(source, root)
    if moved:
        try:
            record_manual_move(root, moved, reason or "manual_selection")
            remove_downstream_files(
                root,
                moved,
                shift_root=root.parent / "shift_data",
            )
        except Exception:
            # 走向记录写入失败时恢复已经移动的源文件，避免文件和记录分叉。
            for relative_path in reversed(moved):
                source = root / relative_path
                target = target_root / relative_path
                if target.is_file() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), str(source))
            record_restore(root, moved)
            raise
    refresh_completed_dates(cleaning)
    save_pipeline_state(state_path, state)
    if reasons:
        log_path = Path(state_path).parent / COMMIT_LOG_NAME
        with log_path.open("a", encoding="utf-8") as file:
            timestamp = datetime.now().astimezone().isoformat()
            file.write(json.dumps({"committed_at": timestamp, "paths": moved, "reasons": reasons}, ensure_ascii=False) + "\n")
    return CommitResult(tuple(moved), tuple(missing))


def commit_marked_items(root_dir: Path | str, state: PipelineState, state_path: Path | str) -> CommitResult:
    return commit_paths(root_dir, state, state_path, state.cleaning.marked_paths)


def commit_folder_items(
    root_dir: Path | str,
    state: PipelineState,
    state_path: Path | str,
    folder: ReviewFolder,
) -> CommitResult:
    """立即移动当前文件夹全部光谱，供文件夹移除操作使用。"""
    mark_paths(state.cleaning, folder.relative_paths, "folder_remove")
    result = commit_paths(root_dir, state, state_path, folder.relative_paths)
    remove_downstream_outputs(root_dir, folder.key, shift_root=Path(root_dir).resolve().parent / "shift_data")
    discard_removed_folder(state.cleaning, folder)
    save_pipeline_state(state_path, state)
    return result


def commit_date_items(
    root_dir: Path | str,
    state: PipelineState,
    state_path: Path | str,
    folders: list[ReviewFolder] | tuple[ReviewFolder, ...],
) -> CommitResult:
    """立即移动当前日期普通文件夹和小球文件夹的全部光谱。"""
    paths = [path for folder in folders for path in folder.relative_paths]
    mark_paths(state.cleaning, paths, "date_remove")
    result = commit_paths(root_dir, state, state_path, paths)
    for folder in folders:
        remove_downstream_outputs(root_dir, folder.key, shift_root=Path(root_dir).resolve().parent / "shift_data")
    root = Path(root_dir).resolve()
    date_names = {folder.date_name for folder in folders}
    for folder in folders:
        _remove_empty_directories(root / folder.date_name / folder.folder_name, root)
    for date_name in date_names:
        _remove_empty_directories(root / date_name, root)
        discard_removed_date(state.cleaning, date_name)
    # commit_paths 已经保存过移动记录；目录清理后再保存一次，确保状态落盘顺序完整。
    save_pipeline_state(state_path, state)
    return result


__all__ = [
    "CommitResult",
    "DEFAULT_CELL_PATTERN",
    "ReviewFolder",
    "SpectrumItem",
    "commit_marked_items",
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
    "scan_date_names",
    "scan_review_folders",
    "seed_default_marks",
    "sync_folder_catalog",
    "toggle_mark_path",
    "unmark_paths",
]
