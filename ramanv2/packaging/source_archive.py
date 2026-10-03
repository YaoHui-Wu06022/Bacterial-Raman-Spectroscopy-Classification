"""构建不包含缓存的 ramanv2 源码 ZIP。"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ramanv2.common.publishing import create_temporary_path, publish_file
from ramanv2.core.paths import PROJECT_ROOT


def build_package_archive(package_dir: Path, archive_path: Path) -> Path:
    """原子生成源码 ZIP，归档路径以源码包目录名为根。"""
    source_dir = package_dir.resolve()
    target_path = archive_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = create_temporary_path(target_path)
    try:
        with ZipFile(temp_path, "w", compression=ZIP_DEFLATED) as archive:
            for source_path in sorted(source_dir.rglob("*")):
                if not source_path.is_file() or _exclude_from_package(source_path, source_dir):
                    continue
                relative_path = source_path.relative_to(source_dir)
                archive.write(source_path, (Path(source_dir.name) / relative_path).as_posix())
        publish_file(temp_path, target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target_path


def run_command(_args) -> int:
    """执行顶层源码 ZIP 命令。"""
    package_dir = Path(__file__).resolve().parents[1]
    archive_path = build_package_archive(package_dir, PROJECT_ROOT / "ramanv2.zip")
    print(archive_path)
    return 0


def _exclude_from_package(source_path: Path, package_dir: Path) -> bool:
    """排除 Python 缓存文件。"""
    relative_parts = source_path.relative_to(package_dir).parts
    return "__pycache__" in relative_parts or source_path.suffix == ".pyc"


__all__ = ["build_package_archive", "run_command"]
