"""临时产物的原子发布工具。

业务模块负责生成和校验内容，本模块只负责在最终路径上安全替换文件或目录。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from uuid import uuid4


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


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
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists():
            rename_directory(target, backup)
        rename_directory(temporary, target)
    except Exception:
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
        if backup.exists():
            rename_directory(backup, target)
        raise
    finally:
        if backup.exists():
            _remove_path(backup)
        if temporary.exists():
            _remove_path(temporary)
    return target


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


__all__ = [
    "create_temporary_path",
    "rename_directory",
    "publish_directory",
    "publish_file",
    "publish_files",
]
