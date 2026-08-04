"""仅导出常规 ramanv2 包的压缩产物。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ramanv2.core.paths import PROJECT_ROOT


def build_package_archive(package_dir: Path, archive_path: Path) -> Path:
    """构建不含 Stanford 扩展与缓存文件的 ramanv2 压缩包。"""
    source_dir = package_dir.resolve()
    target_path = archive_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}_",
        suffix=".zip",
        dir=target_path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with ZipFile(temp_path, "w", compression=ZIP_DEFLATED) as archive:
            for source_path in sorted(source_dir.rglob("*")):
                if not source_path.is_file() or _exclude_from_package(source_path, source_dir):
                    continue
                relative_path = source_path.relative_to(source_dir)
                archive.write(source_path, (Path(source_dir.name) / relative_path).as_posix())
        temp_path.replace(target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target_path


def run_command(_args) -> int:
    """执行顶层 zip 命令并输出压缩包路径。"""
    package_dir = Path(__file__).resolve().parent
    archive_path = build_package_archive(package_dir, PROJECT_ROOT / "ramanv2.zip")
    print(archive_path)
    return 0


def _exclude_from_package(source_path: Path, package_dir: Path) -> bool:
    """判断单个源文件是否属于不应发布的包内内容。"""
    relative_parts = source_path.relative_to(package_dir).parts
    return (
        "__pycache__" in relative_parts
        or source_path.suffix == ".pyc"
        or relative_parts[:2] == ("extensions", "stanford_finetune")
    )
