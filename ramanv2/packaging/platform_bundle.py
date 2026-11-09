"""为远程训练平台导出数据 TAR.XZ 与源码 ZIP 平台包。"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from ramanv2.common.arc_data import list_arc_data_paths, read_arc_data
from ramanv2.common.file_ops import create_temporary_path, publish_file, publish_files
from ramanv2.common.paths import PROJECT_ROOT
from ramanv2.data.profiles import DatasetProfile, DatasetSelection, resolve_selection
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
    selection = resolve_selection(profile_id=profile_id, project_root=project_root)
    stage_dirs = _resolve_stage_dirs(selection)
    config_files = _resolve_config_files(selection)

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = create_temporary_path(target)
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        archive_path, sample_counts = _build_data_archive(
            stage_dirs,
            config_files,
            temporary / target.name,
        )
        publish_file(archive_path, target)
        shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _data_report(selection.profile, stage_dirs, config_files, target, sample_counts)


def build_platform_bundle(
    profile_id: str,
    output_dir: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """原子生成只含目录数据的 data.tar.xz 和源码 ramanv2.zip。"""
    selection = resolve_selection(profile_id=profile_id, project_root=project_root)
    stage_dirs = _resolve_stage_dirs(selection)
    config_files = _resolve_config_files(selection)

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = create_temporary_path(target_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        _, sample_counts = _build_data_archive(
            stage_dirs,
            config_files,
            temporary_dir / "data.tar.xz",
        )
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
        "profile_id": selection.profile.profile_id,
        "dataset_name": selection.profile.dataset_name,
        "source_init": str(stage_dirs[STAGE_INIT][1].resolve()),
        "source_dirs": {
            name: str(source_dir.resolve())
            for name, (_archive_root, source_dir) in stage_dirs.items()
        },
        "config_files": {
            archive_path.as_posix(): str(source_path.resolve())
            for archive_path, source_path in config_files
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


def _archive_root(profile: DatasetProfile) -> Path:
    """返回数据集在归档中的根目录；扁平数据集没有 profile 子目录。"""
    root = Path("dataset") / profile.dataset_dir_name
    return root / profile.subdir if profile.subdir else root


def _resolve_stage_dirs(
    selection: DatasetSelection,
) -> dict[str, tuple[Path, Path]]:
    """返回打包阶段及其 (归档根, 源目录)。

    ``init`` 必需；profile 测试集存在时一并打包，缺失时跳过。
    """
    profile = selection.profile
    dataset_dir = selection.dataset_dir
    archive_root = _archive_root(profile)
    init_dir = dataset_dir / profile.root_init
    if not init_dir.is_dir():
        raise FileNotFoundError(f"缺少 profile init 目录：{init_dir}")
    stages: dict[str, tuple[Path, Path]] = {
        STAGE_INIT: (archive_root / profile.root_init, init_dir),
    }
    profile_test_dir = dataset_dir / profile.root_init_test
    if profile_test_dir.is_dir():
        stages[STAGE_INIT_TEST] = (
            archive_root / profile.root_init_test,
            profile_test_dir,
        )
    return stages


def _resolve_config_files(
    selection: DatasetSelection,
) -> list[tuple[Path, Path]]:
    """返回当前数据集需要随包发布的 JSON 配置，不包含其他数据集。"""
    archive_root = _archive_root(selection.profile)
    spec = selection.spec
    candidates = [
        (archive_root / "dataset.json", spec.config_path),
        (archive_root / "hierarchy.json", spec.hierarchy_path),
    ]
    return [(archive_path, source) for archive_path, source in candidates if source.is_file()]


def _build_data_archive(
    stage_dirs: dict[str, tuple[Path, Path]],
    config_files: list[tuple[Path, Path]],
    output_path: Path,
) -> tuple[Path, dict[str, int]]:
    """校验并压缩各阶段原始 arc_data 文件与数据集 JSON 配置。"""
    sample_counts: dict[str, int] = {}
    with tarfile.open(output_path, "w:xz", preset=XZ_PRESET, dereference=True) as archive:
        for stage_name, (archive_root, source_dir) in stage_dirs.items():
            files = list_arc_data_paths(source_dir, recursive=True)
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
        for archive_path, source_path in config_files:
            archive.add(
                source_path,
                arcname=archive_path.as_posix(),
                recursive=False,
            )
    return output_path, sample_counts


def _data_report(
    profile,
    stage_dirs: dict[str, tuple[Path, Path]],
    config_files: list[tuple[Path, Path]],
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
        "config_files": {
            archive_path.as_posix(): str(source_path.resolve())
            for archive_path, source_path in config_files
        },
        "output_path": str(output_path),
        "sample_count": sum(sample_counts.values()),
        "sample_counts": sample_counts,
        "file_size": output_path.stat().st_size,
    }


__all__ = ["build_platform_bundle", "build_profile_data_archive"]
