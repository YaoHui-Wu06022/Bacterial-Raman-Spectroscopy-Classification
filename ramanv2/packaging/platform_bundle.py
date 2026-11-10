"""为远程训练平台导出光谱数据 npz 与源码 ZIP 平台包。"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ramanv2.common.arc_data import list_arc_data_paths, read_arc_data, write_arc_data
from ramanv2.common.file_ops import (
    create_temporary_path,
    publish_directory,
    publish_file,
    publish_files,
    write_text_atomic,
)
from ramanv2.common.paths import PROJECT_ROOT
from ramanv2.data.profiles import DatasetProfile, DatasetSelection, resolve_selection
from ramanv2.packaging.source_archive import build_package_archive


STAGE_INIT = "init"
STAGE_INIT_TEST = "init_test"
DATA_ARCHIVE_NAME = "data.npz"
ARCHIVE_VERSION = 1


def build_profile_data_archive(
    profile_id: str,
    output_path: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """将 profile 的 init 与测试集原始光谱打包为 data.npz。"""
    selection = resolve_selection(profile_id=profile_id, project_root=project_root)
    stage_dirs = _resolve_stage_dirs(selection)
    config_files = _resolve_config_files(selection)

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = create_temporary_path(target)
    try:
        sample_counts = _build_data_archive(stage_dirs, config_files, temporary)
        publish_file(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return _data_report(selection.profile, stage_dirs, config_files, target, sample_counts)


def build_platform_bundle(
    profile_id: str,
    output_dir: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """原子生成只含光谱数据的 data.npz 和源码 ramanv2.zip。"""
    selection = resolve_selection(profile_id=profile_id, project_root=project_root)
    stage_dirs = _resolve_stage_dirs(selection)
    config_files = _resolve_config_files(selection)

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = create_temporary_path(target_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        sample_counts = _build_data_archive(
            stage_dirs,
            config_files,
            temporary_dir / DATA_ARCHIVE_NAME,
        )
        package_path = temporary_dir / "ramanv2.zip"
        build_package_archive(Path(project_root).resolve() / "ramanv2", package_path)
        final_paths = {
            target_dir / DATA_ARCHIVE_NAME: temporary_dir / DATA_ARCHIVE_NAME,
            target_dir / "ramanv2.zip": package_path,
        }
        publish_files(final_paths)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    shutil.rmtree(temporary_dir, ignore_errors=True)
    data_path = target_dir / DATA_ARCHIVE_NAME
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


def restore_data_archive(
    archive_path: Path | str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    """把 data.npz 恢复为 dataset 下的 init/init_test 文本光谱与配置文件。"""
    source_path = Path(archive_path).resolve()
    root_dir = Path(project_root).resolve()
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            version = int(archive["version"])
            if version != ARCHIVE_VERSION:
                raise ValueError(f"数据包版本不兼容：{version}")
            stage_names = [str(item) for item in archive["stage_names"]]
            stage_roots = [str(item) for item in archive["stage_roots"]]
            sample_stages = np.asarray(archive["sample_stages"], dtype=np.int64)
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
            paths = [str(item) for item in archive["paths"]]
            wavenumbers = np.asarray(archive["wavenumbers"], dtype=np.float32)
            intensities = np.asarray(archive["intensities"], dtype=np.float32)
            config_paths = [str(item) for item in archive["config_paths"]]
            config_contents = [str(item) for item in archive["config_contents"]]
    except (KeyError, ValueError, OSError, BadZipFile) as error:
        raise ValueError(f"数据包无效：{source_path}") from error
    if not stage_roots or len(stage_roots) != len(stage_names):
        raise ValueError(f"数据包缺少阶段信息：{source_path}")
    if offsets.size != len(paths) + 1 or sample_stages.size != len(paths):
        raise ValueError(f"数据包索引长度不一致：{source_path}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != intensities.size:
        raise ValueError(f"数据包偏移量无效：{source_path}")
    if wavenumbers.size != intensities.size:
        raise ValueError(f"数据包波数与强度长度不一致：{source_path}")

    temporary_dirs: dict[int, tuple[Path, Path]] = {}
    try:
        for stage_index, stage_root in enumerate(stage_roots):
            target_dir = _resolve_archive_path(root_dir, stage_root)
            temporary_dir = create_temporary_path(target_dir)
            temporary_dir.mkdir(parents=True, exist_ok=False)
            temporary_dirs[stage_index] = (temporary_dir, target_dir)
        for sample_index, sample_path in enumerate(paths):
            stage_index = int(sample_stages[sample_index])
            if stage_index not in temporary_dirs:
                raise ValueError(f"数据包阶段索引无效：{stage_index}")
            start = int(offsets[sample_index])
            end = int(offsets[sample_index + 1])
            target_path = _resolve_archive_path(temporary_dirs[stage_index][0], sample_path)
            write_arc_data(
                target_path,
                wavenumbers[start:end],
                intensities[start:end],
                fmt="%.8f",
            )
        for temporary_dir, target_dir in temporary_dirs.values():
            publish_directory(temporary_dir, target_dir)
        for config_path, content in zip(config_paths, config_contents):
            write_text_atomic(_resolve_archive_path(root_dir, config_path), content)
    finally:
        for temporary_dir, _target_dir in temporary_dirs.values():
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

    sample_counts = {
        stage_name: int(np.count_nonzero(sample_stages == stage_index))
        for stage_index, stage_name in enumerate(stage_names)
    }
    return {
        "sample_count": len(paths),
        "sample_counts": sample_counts,
        "stage_roots": stage_roots,
        "config_paths": config_paths,
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
    """返回数据集根下的配置文件及其归档目标路径。"""
    dataset_archive_root = Path("dataset") / selection.spec.directory.name
    candidates = [
        (dataset_archive_root / "dataset.json", selection.spec.config_path),
        (dataset_archive_root / "hierarchy.json", selection.spec.hierarchy_path),
    ]
    return [(archive_path, source) for archive_path, source in candidates if source.is_file()]


def _build_data_archive(
    stage_dirs: dict[str, tuple[Path, Path]],
    config_files: list[tuple[Path, Path]],
    output_path: Path | str,
) -> dict[str, int]:
    """把各阶段光谱写入单个 data.npz，并附带数据集 JSON 配置。"""
    stage_names: list[str] = []
    stage_roots: list[str] = []
    sample_stages: list[int] = []
    sample_paths: list[str] = []
    offsets: list[int] = [0]
    wavenumber_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []
    sample_counts: dict[str, int] = {}
    for stage_name, (stage_root, source_dir) in stage_dirs.items():
        files = list_arc_data_paths(source_dir, recursive=True)
        if not files:
            if stage_name == STAGE_INIT:
                raise RuntimeError(f"init 目录中没有 .arc_data 光谱：{source_dir}")
            continue
        stage_index = len(stage_names)
        stage_names.append(stage_name)
        stage_roots.append(stage_root.as_posix())
        for source_path in files:
            wavenumbers, intensities = read_arc_data(source_path)
            if wavenumbers.size < 2 or intensities.size != wavenumbers.size:
                raise ValueError(f"无效光谱文件：{source_path}")
            wavenumber_parts.append(np.asarray(wavenumbers, dtype=np.float32))
            intensity_parts.append(np.asarray(intensities, dtype=np.float32))
            offsets.append(offsets[-1] + wavenumbers.size)
            sample_paths.append(source_path.relative_to(source_dir).as_posix())
            sample_stages.append(stage_index)
        sample_counts[stage_name] = len(files)
    if not sample_paths:
        raise RuntimeError("数据包没有可打包的光谱")

    config_paths: list[str] = []
    config_contents: list[str] = []
    for archive_path, source_path in config_files:
        config_paths.append(archive_path.as_posix())
        config_contents.append(source_path.read_text(encoding="utf-8"))

    with Path(output_path).open("wb") as stream:
        np.savez_compressed(
            stream,
            version=np.asarray(ARCHIVE_VERSION),
            created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
            stage_names=np.asarray(stage_names, dtype=np.str_),
            stage_roots=np.asarray(stage_roots, dtype=np.str_),
            sample_stages=np.asarray(sample_stages, dtype=np.int16),
            offsets=np.asarray(offsets, dtype=np.int64),
            paths=np.asarray(sample_paths, dtype=np.str_),
            wavenumbers=np.concatenate(wavenumber_parts),
            intensities=np.concatenate(intensity_parts),
            config_paths=np.asarray(config_paths, dtype=np.str_),
            config_contents=np.asarray(config_contents, dtype=np.str_),
        )
    return sample_counts


def _resolve_archive_path(root_dir: Path, relative_text: str) -> Path:
    """解析数据包内的相对路径，拒绝越出项目根目录的目标。"""
    if not str(relative_text).strip():
        raise ValueError("数据包包含空路径")
    target = (root_dir / relative_text).resolve()
    if target != root_dir and root_dir not in target.parents:
        raise ValueError(f"数据包路径越界：{relative_text}")
    return target


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


__all__ = [
    "DATA_ARCHIVE_NAME",
    "build_platform_bundle",
    "build_profile_data_archive",
    "restore_data_archive",
]
