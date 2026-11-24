"""源码与实验目录的 ZIP 归档。"""

from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ramanv2.common.file_ops import create_temporary_path, file_sha256, publish_file
from ramanv2.common.paths import PROJECT_ROOT


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


def _exclude_from_package(source_path: Path, package_dir: Path) -> bool:
    """排除 Python 缓存文件。"""
    relative_parts = source_path.relative_to(package_dir).parts
    return "__pycache__" in relative_parts or source_path.suffix == ".pyc"


def publish_experiment_archive(
    experiment_dir: Path | str,
    *,
    package_enable: bool = True,
    output_root: Path | str = PROJECT_ROOT,
) -> Path | None:
    """把实验目录压缩到项目根目录，并返回压缩包路径。"""
    source = Path(experiment_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"实验目录不存在：{source}")
    print(f"experiment_dir = {source}")
    if not package_enable:
        return None
    archive_path = Path(output_root).resolve() / f"{source.name}.zip"
    temporary_base = create_temporary_path(archive_path)
    temporary_archive: Path | None = None
    try:
        temporary_archive = Path(
            shutil.make_archive(
                str(temporary_base),
                "zip",
                root_dir=source.parent,
                base_dir=source.name,
            )
        )
        publish_file(temporary_archive, archive_path)
    finally:
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)
    print(f"experiment_archive = {archive_path}")
    print(f"size = {archive_path.stat().st_size / 2**20:.1f} MiB")
    print(f"sha256 = {file_sha256(archive_path)}")
    return archive_path


__all__ = ["build_package_archive", "publish_experiment_archive"]