"""为 Colab 导出 profile 数据 ZIP 和 ramanv2 源码 ZIP。"""

from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.publishing import create_temporary_path, publish_file, publish_files
from ramanv2.core.paths import PROJECT_ROOT
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.packaging.source_archive import build_package_archive


def build_profile_data_archive(
    profile_id: str,
    output_path: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """将 profile 的 init 目录原样打包为 data.zip。"""
    profile = get_profile(profile_id)
    source_init = get_dataset_dir(profile, project_root) / profile.root_init
    if not source_init.is_dir():
        raise FileNotFoundError(f"缺少 profile init 目录：{source_init}")

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = create_temporary_path(target)
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        archive_path, sample_count = _build_data_archive(
            profile,
            source_init,
            temporary / target.name,
        )
        publish_file(archive_path, target)
        shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _data_report(profile, source_init, target, sample_count)


def build_colab_bundle(
    profile_id: str,
    output_dir: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """原子生成只含目录数据的 data.zip 和源码 ramanv2.zip。"""
    profile = get_profile(profile_id)
    source_init = get_dataset_dir(profile, project_root) / profile.root_init
    if not source_init.is_dir():
        raise FileNotFoundError(f"缺少 profile init 目录：{source_init}")

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = create_temporary_path(target_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        _, sample_count = _build_data_archive(profile, source_init, temporary_dir / "data.zip")
        package_path = temporary_dir / "ramanv2.zip"
        build_package_archive(Path(project_root).resolve() / "ramanv2", package_path)
        final_paths = {
            target_dir / "data.zip": temporary_dir / "data.zip",
            target_dir / "ramanv2.zip": package_path,
        }
        publish_files(final_paths)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    shutil.rmtree(temporary_dir, ignore_errors=True)
    data_path = target_dir / "data.zip"
    package_path = target_dir / "ramanv2.zip"
    return {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "source_init": str(source_init.resolve()),
        "sample_count": sample_count,
        "data_archive": {
            "output_path": str(data_path),
            "file_size": data_path.stat().st_size,
        },
        "package_archive": {
            "output_path": str(package_path),
            "file_size": package_path.stat().st_size,
        },
    }


def _build_data_archive(profile, source_init: Path, output_path: Path) -> tuple[Path, int]:
    """校验并压缩 init 下的原始 arc_data 文件。"""
    files = sorted(path for path in source_init.rglob("*.arc_data") if path.is_file())
    if not files:
        raise RuntimeError(f"init 目录中没有 .arc_data 光谱：{source_init}")
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for source_path in files:
            wavenumbers, intensities = read_arc_data(source_path)
            if wavenumbers.size < 2 or intensities.size != wavenumbers.size:
                raise ValueError(f"无效光谱文件：{source_path}")
            relative = source_path.relative_to(source_init)
            archive_name = (Path("dataset") / profile.dataset_name / "init" / relative).as_posix()
            archive.write(source_path, archive_name)
    return output_path, len(files)


def _data_report(profile, source_init: Path, output_path: Path, sample_count: int) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "source_init": str(source_init.resolve()),
        "output_path": str(output_path),
        "sample_count": sample_count,
        "file_size": output_path.stat().st_size,
    }


__all__ = ["build_colab_bundle", "build_profile_data_archive"]
