"""常规数据集的 train/test 离线构建。"
构建总在同级临时目录完成；成功校验后才发布为正式产物，并保留原产物备份。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
from pathlib import Path
import re

import numpy as np

from ramanv2.core.paths import resolve_path
from ramanv2.core.config import InputConfig
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.preprocess import preprocess_single_spectrum

from ramanv2.data.config import DataBuildConfig, build_cosmic_ray_options, resolve_build_config
from ramanv2.common.arc_data import write_arc_data
from ramanv2.common.publishing import create_temporary_path, publish_directory

from ramanv2.data.io import iter_init_groups


MAX_BUILD_WORKERS = 8


def _physical_clean_group(
    profile,
    input_config: InputConfig,
    build_config: DataBuildConfig,
    samples,
    reference_wavenumbers: np.ndarray | None = None,
):
    """将一个原始叶子目录的光谱清洗到统一轴。"""
    reference_axis = (
        build_wn_ref(
            input_config.cut_min,
            input_config.cut_max,
            input_config.target_points,
        )
        if reference_wavenumbers is None
        else _validate_reference_wavenumbers(reference_wavenumbers, input_config)
    )
    options = build_cosmic_ray_options(profile.profile_id, build_config)
    cleaned = []
    for filename, wavenumbers, intensities in samples:
        if not wavenumbers.size or not intensities.size:
            continue
        output_axis, output_spectrum, _ = preprocess_single_spectrum(
            wavenumbers,
            intensities,
            cut_min=input_config.cut_min,
            cut_max=input_config.cut_max,
            reference_wavenumbers=reference_axis,
            bad_bands=input_config.bad_bands,
            baseline_lam=build_config.baseline_lam,
            baseline_asls_p=build_config.baseline_asls_p,
            baseline_max_iter=build_config.baseline_max_iter,
            baseline_fit_min=build_config.baseline_fit_min,
            baseline_fit_max=build_config.baseline_fit_max,
            baseline_method=build_config.baseline_method,
            **options,
        )
        if output_axis is not None:
            cleaned.append((filename, output_axis, output_spectrum))
    return cleaned


def _resolve_build_worker_count() -> int:
    """限制离线构建进程数，避免占满训练环境的全部 CPU。"""
    return min(MAX_BUILD_WORKERS, max(1, os.cpu_count() or 1))


def _iter_physical_clean_groups(
    profile,
    input_config: InputConfig,
    build_config: DataBuildConfig,
    groups,
    reference_wavenumbers: np.ndarray | None,
):
    """按输入顺序并行清洗叶子目录，限制在途任务以控制内存。"""
    worker_count = _resolve_build_worker_count()
    if worker_count == 1:
        for relative_dir, leaf_name, raw_samples in groups:
            yield relative_dir, leaf_name, _physical_clean_group(
                profile,
                input_config,
                build_config,
                raw_samples,
                reference_wavenumbers,
            )
        return

    pending = deque()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for relative_dir, leaf_name, raw_samples in groups:
            future = executor.submit(
                _physical_clean_group,
                profile,
                input_config,
                build_config,
                raw_samples,
                reference_wavenumbers,
            )
            pending.append((relative_dir, leaf_name, future))
            if len(pending) >= worker_count * 2:
                ready_relative_dir, ready_leaf_name, ready_future = pending.popleft()
                yield ready_relative_dir, ready_leaf_name, ready_future.result()
        while pending:
            ready_relative_dir, ready_leaf_name, ready_future = pending.popleft()
            yield ready_relative_dir, ready_leaf_name, ready_future.result()


def _validate_reference_wavenumbers(
    reference_wavenumbers: np.ndarray,
    input_config: InputConfig,
) -> np.ndarray:
    """校验调用方提供的外部统一波数轴，避免改变常规构建默认值。"""
    axis = np.asarray(reference_wavenumbers, dtype=np.float64)
    if axis.ndim != 1 or axis.size != input_config.target_points:
        raise ValueError("外部参考波数轴长度与 InputConfig.target_points 不一致")
    if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("外部参考波数轴必须是有限严格递增的一维数组")
    return axis


def _target_group_name(relative_dir: Path, leaf_name: str) -> Path:
    """按物种简称合并来源文件夹。"""
    matched = re.match(r"([A-Za-z]+)([+-])?", str(leaf_name))
    target = matched.group(1) + (matched.group(2) or "") if matched else leaf_name
    return relative_dir.parent / target if relative_dir != Path(".") else Path(target)


def resolve_train_group_name(relative_dir: Path, leaf_name: str) -> Path:
    """返回 init 来源目录经前缀合并后的 train 相对类别路径。"""
    return _target_group_name(relative_dir, leaf_name)


def _ensure_name_prefix(prefix: str, filename: str) -> str:
    """为合并来源补齐目录名前缀，避免同名光谱覆盖。"""
    marker = f"{prefix}_"
    return filename if filename.startswith(marker) else f"{marker}{filename}"


def _write_group(output_root: Path, relative_dir: Path, samples):
    """将完成清洗的同一类别光谱写入临时构建目录。"""
    target_dir = output_root / relative_dir
    for filename, wavenumbers, intensities in samples:
        write_arc_data(
            target_dir / filename,
            wavenumbers,
            np.asarray(intensities, dtype=np.float32),
            fmt="%.3f",
        )


def build_train(
    profile,
    base_dir: Path | str,
    config: DataBuildConfig | None = None,
    input_config: InputConfig = InputConfig(),
    reference_wavenumbers: np.ndarray | None = None,
):
    """从 init 构建 train。"""
    config = resolve_build_config(config)
    base_dir = Path(base_dir)
    input_path = resolve_path(profile.root_init, base_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing init folder: {input_path}")
    output_path = resolve_path(profile.root_train_clean, base_dir)
    figure_path = resolve_path(profile.root_train_fig, base_dir)
    temp_output = create_temporary_path(output_path)
    temp_figure = create_temporary_path(figure_path)
    if temp_output.exists() or temp_figure.exists():
        raise FileExistsError("随机生成的临时构建路径已存在，请重试")
    temp_output.mkdir(parents=True)
    temp_figure.mkdir(parents=True)
    merged = {}
    skipped_sources = 0
    try:
        worker_count = _resolve_build_worker_count()
        print(f"- Preprocess workers: {worker_count}")
        for relative_dir, leaf_name, physical_samples in _iter_physical_clean_groups(
            profile,
            input_config,
            config,
            iter_init_groups(input_path),
            reference_wavenumbers,
        ):
            if len(physical_samples) < config.min_samples_per_class:
                skipped_sources += 1
                continue
            target_relative = _target_group_name(relative_dir, leaf_name)
            renamed_samples = [
                (_ensure_name_prefix(leaf_name, filename), wavenumbers, intensities)
                for filename, wavenumbers, intensities in physical_samples
            ]
            merged.setdefault(target_relative.as_posix(), []).extend(renamed_samples)

        built_groups = 0
        for relative_text, samples in sorted(merged.items()):
            if len(samples) < config.min_samples_per_class:
                continue
            if len(samples) < config.min_samples_per_class:
                continue
            _write_group(temp_output, Path(relative_text), samples)
            built_groups += 1

        if built_groups == 0:
            raise RuntimeError("训练集构建未生成任何类别，拒绝发布空产物")
        publish_directory(temp_output, output_path)
        publish_directory(temp_figure, figure_path)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        shutil.rmtree(temp_figure, ignore_errors=True)
        raise

    print("\nTraining dataset preprocessing finished:")
    print(f"- Final train spectra: {output_path}")
    print(f"- Groups built: {built_groups}")
    print(f"- Skipped source groups: {skipped_sources}")


def build_test(
    profile,
    base_dir: Path | str,
    config: DataBuildConfig | None = None,
    input_config: InputConfig = InputConfig(),
    reference_wavenumbers: np.ndarray | None = None,
):
    """从 init_test 构建独立 test。"""
    config = resolve_build_config(config)
    base_dir = Path(base_dir)
    input_path = resolve_path(profile.root_init_test, base_dir)
    output_path = resolve_path(profile.root_test, base_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing init_test folder: {input_path}")
    temp_output = create_temporary_path(output_path)
    temp_output.mkdir(parents=True)
    groups_built = 0
    spectra_built = 0
    try:
        for relative_dir, leaf_name, raw_samples in iter_init_groups(input_path):
            cleaned = _physical_clean_group(
                profile,
                input_config,
                config,
                raw_samples,
                reference_wavenumbers,
            )
            if not cleaned:
                continue
            _write_group(temp_output, relative_dir, cleaned)
            groups_built += 1
            spectra_built += len(cleaned)
        if groups_built == 0:
            raise RuntimeError("测试集构建未生成任何分组，拒绝发布空产物")
        publish_directory(temp_output, output_path)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise

    print("\nTest dataset preprocessing finished:")
    print(f"- Final test spectra: {output_path}")
    print(f"- Groups built: {groups_built}")
    print(f"- Spectra built: {spectra_built}")
