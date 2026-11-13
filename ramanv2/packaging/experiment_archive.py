"""把完整实验目录打包为可传输的 ZIP。"""

from __future__ import annotations

import shutil
from pathlib import Path

from ramanv2.common.file_ops import file_sha256
from ramanv2.common.paths import PROJECT_ROOT


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
    archive_path.unlink(missing_ok=True)
    shutil.make_archive(
        str(archive_path.with_suffix("")),
        "zip",
        root_dir=source.parent,
        base_dir=source.name,
    )
    print(f"experiment_archive = {archive_path}")
    print(f"size = {archive_path.stat().st_size / 2**20:.1f} MiB")
    print(f"sha256 = {file_sha256(archive_path)}")
    return archive_path


__all__ = ["publish_experiment_archive"]
