"""临时产物的原子发布工具。

业务模块负责生成和校验内容，本模块只负责在最终路径上安全替换文件或目录。
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def _remove_readonly(function, path, error) -> None:
    """快照会继承 Windows 只读属性；仅在权限错误时清除属性并重试删除。"""
    if not isinstance(error[1], PermissionError):
        raise error[1]
    Path(path).chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    function(path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, onerror=_remove_readonly)
    else:
        try:
            path.unlink(missing_ok=True)
        except PermissionError as error:
            _remove_readonly(os.unlink, path, (type(error), error, error.__traceback__))


def create_temporary_path(target_path: Path | str) -> Path:
    """为目标文件或目录创建同级临时路径，调用方负责写入内容。"""
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f".{target.name}.{uuid4().hex}.tmp")


def rename_directory(source_path: Path | str, target_path: Path | str) -> None:
    """在 Windows 下重命名目录并短暂重试文件锁。"""
    source = Path(source_path).resolve()
    target = Path(target_path).resolve()
    for attempt, delay in enumerate((0.0, 0.05, 0.2, 0.5, 1.0)):
        if delay:
            time.sleep(delay)
        try:
            os.rename(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise


def publish_file(temp_path: Path | str, target_path: Path | str) -> Path:
    """原子替换一个文件，并在成功后清理临时路径。"""
    temporary = Path(temp_path).resolve()
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)
    return target


def publish_directory(temp_path: Path | str, target_path: Path | str) -> Path:
    """用临时目录替换目标目录，失败时保留旧目录。"""
    temporary = Path(temp_path).resolve()
    target = Path(target_path).resolve()
    backup = target.with_name(f".{target.name}.{uuid4().hex}.backup")
    rollback_failed = False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists():
            rename_directory(target, backup)
        rename_directory(temporary, target)
    except Exception:
        if backup.exists():
            if target.exists():
                _remove_path(target)
            try:
                rename_directory(backup, target)
            except Exception:
                rollback_failed = True
                raise
        raise
    finally:
        if backup.exists() and not rollback_failed:
            _remove_path(backup)
        if temporary.exists():
            _remove_path(temporary)
    return target


@contextmanager
def preserve_paths(paths: list[Path]):
    """跨源谱和派生产物操作失败时恢复快照；只用于明确列出的非嵌套目标。"""
    targets = list(dict.fromkeys(path.resolve() for path in paths))
    if any(path == path.parent for path in targets):
        raise ValueError("不能为文件系统根目录创建操作快照")
    if any(parent in child.parents for parent in targets for child in targets):
        raise ValueError("操作快照目标不能互相包含")
    snapshots: list[tuple[Path, Path, bool]] = []
    rollback_failed = False
    try:
        for target in targets:
            backup = create_temporary_path(target)
            existed = target.exists()
            snapshots.append((target, backup, existed))
            if target.is_dir():
                shutil.copytree(target, backup)
            elif existed:
                shutil.copy2(target, backup)
        try:
            yield
        except Exception:
            try:
                for target, backup, existed in reversed(snapshots):
                    if target.exists():
                        _remove_path(target)
                    if existed:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(backup, target)
            except Exception:
                rollback_failed = True  # 保留剩余快照，不能在回滚失败时清掉唯一备份。
                raise
            raise
    finally:
        if not rollback_failed:
            for _, backup, _ in snapshots:
                if backup.exists():
                    _remove_path(backup)


def publish_files(files: dict[Path | str, Path | str]) -> None:
    """替换多个文件；任一替换失败时恢复已发布文件。"""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    pairs = [(Path(target).resolve(), Path(temp).resolve()) for target, temp in files.items()]
    try:
        for target, temporary in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.{uuid4().hex}.backup")
                os.replace(target, backup)
                backups.append((target, backup))
            os.replace(temporary, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        for target, backup in reversed(backups):
            if backup.exists() and not target.exists():
                os.replace(backup, target)
        raise
    finally:
        for _, backup in backups:
            backup.unlink(missing_ok=True)
        for _, temporary in pairs:
            temporary.unlink(missing_ok=True)


def move_files(files: dict[Path, Path]) -> None:
    """预检查全部归档目标，再移动文件；中途失败则恢复已移动的源文件。"""
    for source, target in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"源文件不存在：{source}")
        if target.exists():
            raise FileExistsError(f"归档目标已存在：{target}")
    moved = []
    created_dirs = set()
    try:
        for source, target in files.items():
            parent = target.parent
            while not parent.exists():
                created_dirs.add(parent)
                parent = parent.parent
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(source))
        for directory in sorted(created_dirs, key=lambda path: len(path.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise


__all__ = [
    "create_temporary_path",
    "rename_directory",
    "publish_directory",
    "publish_file",
    "publish_files",
    "preserve_paths",
    "move_files",
]
