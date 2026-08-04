"""常规数据集的 train/test 离线构建。"
构建总在同级临时目录完成；成功校验后才发布为正式产物，并保留原产物备份。"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
import re

import numpy as np

from ramanv2.core.paths import resolve_path
from ramanv2.core.config import InputConfig
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.preprocess import preprocess_single_spectrum

from .config import DataBuildConfig, resolve_build_config, resolve_cosmic_options
from .io import iter_init_groups, resolve_init_input, write_arc_data


def _build_temp_path(output_path: Path) -> Path:
    """为一次构建分配唯一同级临时目录，不触碰既有产物。"""
    return output_path.parent / f"{output_path.name}_building_{uuid.uuid4().hex[:8]}"


def _backup_path(output_path: Path) -> Path:
    """为已发布产物分配唯一备份路径，保证原结果可恢复。"""
    return output_path.parent / f"{output_path.name}_previous_{uuid.uuid4().hex[:8]}"


def _publish_directory(temp_path: Path, output_path: Path):
    """校验后的临时目录替换正式目录；旧目录保留为同级备份。"""
    if not temp_path.is_dir():
        raise FileNotFoundError(f"Build temp directory missing: {temp_path}")
    backup_path = None
    if output_path.exists():
        backup_path = _backup_path(output_path)
        output_path.replace(backup_path)
    try:
        temp_path.replace(output_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    return backup_path


def _publish_file(temp_path: Path, output_path: Path):
    """发布单个构建日志，同时保留此前日志备份。"""
    if not temp_path.is_file():
        raise FileNotFoundError(f"Build temp file missing: {temp_path}")
    backup_path = None
    if output_path.exists():
        backup_path = _backup_path(output_path)
        output_path.replace(backup_path)
    try:
        temp_path.replace(output_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    return backup_path


def _pca_reconstruction_error(spectra, components=0.95, is_centered: bool = True):
    """使用 PCA 重构误差识别同一类别中的异常训练谱。"""
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[0] < 2:
        return 0, np.zeros(spectra.shape[0], dtype=np.float32)
    mean = spectra.mean(axis=0, keepdims=True) if is_centered else 0.0
    centered = spectra - mean
    left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return 0, np.zeros(spectra.shape[0], dtype=np.float32)
    variance = singular_values**2 / max(spectra.shape[0] - 1, 1)
    total_variance = float(variance.sum())
    if total_variance <= 0:
        return 0, np.zeros(spectra.shape[0], dtype=np.float32)
    if isinstance(components, float) and 0 < components <= 1:
        component_count = int(np.searchsorted(np.cumsum(variance) / total_variance, components) + 1)
    else:
        component_count = int(components)
    component_count = max(1, min(component_count, min(spectra.shape)))
    reconstructed = (left[:, :component_count] * singular_values[:component_count]) @ right[:component_count]
    if is_centered:
        reconstructed = reconstructed + mean
    errors = np.mean((spectra - reconstructed) ** 2, axis=1).astype(np.float32)
    return component_count, errors


def _apply_pca_filter(samples, config: DataBuildConfig, label_display: str, log_path: Path):
    """使用 PCA 重构误差删除异常谱，并仅在 PCA 启用时写入日志。"""
    filenames, axes, spectra = zip(*samples)
    spectra_array = np.vstack(spectra)
    component_count, errors = _pca_reconstruction_error(
        spectra_array,
        components=config.pca_components,
        is_centered=config.pca_center_enable,
    )
    ratio = float(np.clip(config.pca_outlier_ratio, 0.0, 1.0))
    if ratio <= 0.0 or component_count == 0:
        keep_mask = np.ones(len(samples), dtype=bool)
        threshold = float("inf")
    else:
        threshold = float(np.quantile(errors, 1.0 - ratio))
        keep_mask = errors <= threshold
    removed = np.flatnonzero(~keep_mask)
    if removed.size:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"[{label_display}] removed {removed.size} samples, threshold={threshold:.6f}\n")
            for index in removed:
                file.write(f"  {filenames[index]}\t{float(errors[index]):.6f}\n")
    return [sample for sample, keep in zip(samples, keep_mask) if keep], component_count, int(removed.size)


def _physical_clean_group(
    profile,
    input_config: InputConfig,
    build_config: DataBuildConfig,
    samples,
    label_display: str,
    reference_wavenumbers: np.ndarray | None = None,
):
    """将一个原始叶子目录的光谱清洗到统一轴，不执行 PCA。"""
    reference_axis = (
        build_wn_ref(
            input_config.cut_min,
            input_config.cut_max,
            input_config.target_points,
        )
        if reference_wavenumbers is None
        else _validate_reference_wavenumbers(reference_wavenumbers, input_config)
    )
    options = resolve_cosmic_options(profile, build_config, label_display)
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
    """按旧规则合并叶子目录前缀相同的类别。"""
    matched = re.match(r"([A-Za-z]+)([+-])?", str(leaf_name))
    target = matched.group(1) + (matched.group(2) or "") if matched else leaf_name
    return relative_dir.parent / target if relative_dir != Path(".") else Path(target)


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
    """从 init 构建 train；PCA 关闭时不创建或修改 PCA 日志。"""
    config = resolve_build_config(config)
    base_dir = Path(base_dir)
    input_path = resolve_init_input(base_dir, profile)
    output_path = resolve_path(profile.root_train_clean, base_dir)
    figure_path = resolve_path(profile.root_train_fig, base_dir)
    temp_output = _build_temp_path(output_path)
    temp_figure = _build_temp_path(figure_path)
    pca_log_path = resolve_path(profile.pca_log_name, base_dir)
    temp_pca_log = None
    if temp_output.exists() or temp_figure.exists():
        raise FileExistsError("随机生成的临时构建路径已存在，请重试")
    temp_output.mkdir(parents=True)
    temp_figure.mkdir(parents=True)
    if config.pca_enable:
        temp_pca_log = pca_log_path.parent / f"{pca_log_path.name}_building_{uuid.uuid4().hex[:8]}"
        temp_pca_log.write_text("", encoding="utf-8")

    merged = {}
    skipped_sources = 0
    try:
        for relative_dir, leaf_name, raw_samples in iter_init_groups(input_path):
            label_display = relative_dir.as_posix() if relative_dir != Path(".") else leaf_name
            physical_samples = _physical_clean_group(
                profile,
                input_config,
                config,
                raw_samples,
                label_display,
                reference_wavenumbers,
            )
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
            label_display = relative_text
            if config.pca_enable:
                samples, component_count, removed = _apply_pca_filter(
                    samples, config, label_display, temp_pca_log
                )
                print(f"  PCA outlier removal: k={component_count}, removed={removed}")
            if len(samples) < config.min_samples_per_class:
                continue
            _write_group(temp_output, Path(relative_text), samples)
            built_groups += 1

        if built_groups == 0:
            raise RuntimeError("训练集构建未生成任何类别，拒绝发布空产物")
        _publish_directory(temp_output, output_path)
        _publish_directory(temp_figure, figure_path)
        if temp_pca_log is not None:
            _publish_file(temp_pca_log, pca_log_path)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        shutil.rmtree(temp_figure, ignore_errors=True)
        if temp_pca_log is not None:
            temp_pca_log.unlink(missing_ok=True)
        raise

    print("\nTraining dataset preprocessing finished:")
    print(f"- Final train spectra: {output_path}")
    print(f"- Groups built: {built_groups}")
    print(f"- Skipped source groups: {skipped_sources}")
    if config.pca_enable:
        print(f"- PCA log: {pca_log_path}")


def build_test(
    profile,
    base_dir: Path | str,
    config: DataBuildConfig | None = None,
    input_config: InputConfig = InputConfig(),
    reference_wavenumbers: np.ndarray | None = None,
):
    """从 init_test 构建独立 test；测试集从不执行 PCA 筛除。"""
    config = resolve_build_config(config)
    base_dir = Path(base_dir)
    input_path = resolve_path(profile.root_init_test, base_dir)
    output_path = resolve_path(profile.root_test, base_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing init_test folder: {input_path}")
    temp_output = _build_temp_path(output_path)
    temp_output.mkdir(parents=True)
    groups_built = 0
    spectra_built = 0
    try:
        for relative_dir, leaf_name, raw_samples in iter_init_groups(input_path):
            label_display = relative_dir.as_posix() if relative_dir != Path(".") else leaf_name
            cleaned = _physical_clean_group(
                profile,
                input_config,
                config,
                raw_samples,
                label_display,
                reference_wavenumbers,
            )
            if not cleaned:
                continue
            _write_group(temp_output, relative_dir, cleaned)
            groups_built += 1
            spectra_built += len(cleaned)
        if groups_built == 0:
            raise RuntimeError("测试集构建未生成任何分组，拒绝发布空产物")
        _publish_directory(temp_output, output_path)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise

    print("\nTest dataset preprocessing finished:")
    print(f"- Final test spectra: {output_path}")
    print(f"- Groups built: {groups_built}")
    print(f"- Spectra built: {spectra_built}")
