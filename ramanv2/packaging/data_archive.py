"""为远程训练平台导出 init/init_test 数据 TAR.XZ 和 ramanv2 源码 ZIP。"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.publishing import create_temporary_path, publish_file, publish_files
from ramanv2.core.paths import PROJECT_ROOT
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.packaging.source_archive import build_package_archive


STAGE_INIT = "init"
STAGE_INIT_TEST = "init_test"
# xz preset 1（快速模式）相对默认 6 约快 9 倍，体积约增 1.5 倍；
# 对纯文本 arc_data 数据，压缩率差异小，主要收益是本地生成耗时。
XZ_PRESET = 1


def build_profile_data_archive(
    profile_id: str,
    output_path: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """将 profile 的 init 与已存在的 profile 测试集打包为 data.tar.xz。"""
    profile = get_profile(profile_id)
    stage_dirs = _resolve_stage_dirs(profile, project_root)

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = create_temporary_path(target)
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        archive_path, sample_counts = _build_data_archive(
            stage_dirs,
            temporary / target.name,
        )
        publish_file(archive_path, target)
        shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _data_report(profile, stage_dirs, target, sample_counts)


def build_platform_bundle(
    profile_id: str,
    output_dir: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """原子生成只含目录数据的 data.tar.xz 和源码 ramanv2.zip。"""
    profile = get_profile(profile_id)
    stage_dirs = _resolve_stage_dirs(profile, project_root)

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = create_temporary_path(target_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        _, sample_counts = _build_data_archive(stage_dirs, temporary_dir / "data.tar.xz")
        package_path = temporary_dir / "ramanv2.zip"
        build_package_archive(Path(project_root).resolve() / "ramanv2", package_path)
        final_paths = {
            target_dir / "data.tar.xz": temporary_dir / "data.tar.xz",
            target_dir / "ramanv2.zip": package_path,
        }
        publish_files(final_paths)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    shutil.rmtree(temporary_dir, ignore_errors=True)
    data_path = target_dir / "data.tar.xz"
    package_path = target_dir / "ramanv2.zip"
    return {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "source_init": str(stage_dirs[STAGE_INIT][1].resolve()),
        "source_dirs": {
            name: str(source_dir.resolve())
            for name, (_archive_root, source_dir) in stage_dirs.items()
        },
        "sample_count": sum(sample_counts.values()),
        "sample_counts": sample_counts,
        "data_archive": {
            "output_path": str(data_path),
            "file_size": data_path.stat().st_size,
        },
        "package_archive": {
            "output_path": str(package_path),
            "file_size": package_path.stat().st_size,
        },
    }


def _resolve_stage_dirs(
    profile,
    project_root: Path | str,
) -> dict[str, tuple[Path, Path]]:
    """返回打包阶段及其 (归档根, 源目录)。

    ``init`` 必需；profile 测试集存在时一并打包，缺失时跳过。两者都只在
    ``dataset/<profile>/`` 下取，全局测试母本 ``dataset/init_test`` 不进入数据包。
    """
    dataset_dir = get_dataset_dir(profile, project_root)
    profile_root = Path("dataset") / profile.dataset_name
    init_dir = dataset_dir / profile.root_init
    if not init_dir.is_dir():
        raise FileNotFoundError(f"缺少 profile init 目录：{init_dir}")
    stages: dict[str, tuple[Path, Path]] = {
        STAGE_INIT: (profile_root / profile.root_init, init_dir),
    }
    profile_test_dir = dataset_dir / profile.root_init_test
    if profile_test_dir.is_dir():
        stages[STAGE_INIT_TEST] = (
            profile_root / profile.root_init_test,
            profile_test_dir,
        )
    return stages


def _build_data_archive(
    stage_dirs: dict[str, tuple[Path, Path]],
    output_path: Path,
) -> tuple[Path, dict[str, int]]:
    """校验并压缩各阶段原始 arc_data 文件，返回每个阶段的样本数。"""
    sample_counts: dict[str, int] = {}
    with tarfile.open(output_path, "w:xz", preset=XZ_PRESET, dereference=True) as archive:
        for stage_name, (archive_root, source_dir) in stage_dirs.items():
            files = sorted(path for path in source_dir.rglob("*.arc_data") if path.is_file())
            if not files:
                if stage_name == STAGE_INIT:
                    raise RuntimeError(f"init 目录中没有 .arc_data 光谱：{source_dir}")
                continue
            for source_path in files:
                wavenumbers, intensities = read_arc_data(source_path)
                if wavenumbers.size < 2 or intensities.size != wavenumbers.size:
                    raise ValueError(f"无效光谱文件：{source_path}")
                relative = source_path.relative_to(source_dir)
                archive.add(
                    source_path,
                    arcname=(archive_root / relative).as_posix(),
                    recursive=False,
                )
            sample_counts[stage_name] = len(files)
    return output_path, sample_counts


def _data_report(
    profile,
    stage_dirs: dict[str, tuple[Path, Path]],
    output_path: Path,
    sample_counts: dict[str, int],
) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "source_init": str(stage_dirs[STAGE_INIT][1].resolve()),
        "source_dirs": {
            name: str(source_dir.resolve())
            for name, (_archive_root, source_dir) in stage_dirs.items()
        },
        "output_path": str(output_path),
        "sample_count": sum(sample_counts.values()),
        "sample_counts": sample_counts,
        "file_size": output_path.stat().st_size,
    }


__all__ = ["build_platform_bundle", "build_profile_data_archive"]
