"""文件夹级宇宙射线去除的数值算法。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ramanv2.spectra.filters import median_filter_1d


_EPS = 1e-9
# 平台段外侧比值使用固定 3 点，与已验证的批处理脚本一致。
_PLATEAU_OUTSIDE_POINTS = 3


@dataclass(frozen=True)
class FolderCosmicOptions:
    """描述一次文件夹级去尖峰的固定参数。"""

    enable: bool = False
    reference_smooth_window: int = 9
    local_window: int = 7
    z_threshold: float = 7.0
    guard_mad_factor: float = 3.0
    guard_ptp_fraction: float = 0.03
    plateau_z_threshold: float = 6.0
    plateau_run_max: int = 8
    plateau_inside_ratio: float = 1.5
    plateau_outside_ratio: float = 1.25
    run_gap: int = 1
    neighbor_points: int = 4
    max_iterations: int = 4
    min_spectra: int = 8
    small_folder_warning: int = 30


@dataclass(frozen=True)
class FolderCosmicReference:
    """保存文件夹中值谱、真实峰保护和跨谱归一化基准。"""

    median: np.ndarray
    smoothed: np.ndarray
    guard_mask: np.ndarray
    guard_threshold: float
    normalized_median: np.ndarray
    normalized_mad: np.ndarray


@dataclass(frozen=True)
class SpectrumCosmicStats:
    """记录单条谱的替换与诊断统计。"""

    removed_points: int = 0
    removed_runs: int = 0
    plateau_runs: int = 0
    guarded_points: int = 0
    max_z: float = 0.0
    iterations: int = 0
    edge_skipped_runs: int = 0


@dataclass(frozen=True)
class FolderCosmicResult:
    """保存一个文件夹的清洗结果与状态。"""

    status: str
    cleaned: np.ndarray
    stats: tuple[SpectrumCosmicStats, ...]
    reference: FolderCosmicReference | None
    spectrum_count: int


def compute_robust_z(residual, valid_mask=None):
    """使用中位数/MAD 估计稳健 z 分数；无法估计尺度时返回 None。"""
    values = np.asarray(residual, dtype=np.float64)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        values = values[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    center = float(np.median(values))
    centered_abs = np.abs(values - center)
    scale = 1.4826 * float(np.median(centered_abs))
    if scale <= 1e-8:
        scale = float(np.std(values))
    if scale <= 1e-8:
        nonzero = centered_abs[centered_abs > 1e-8]
        if nonzero.size:
            scale = 1.4826 * float(np.median(nonzero))
    if scale <= 1e-8:
        return None
    original = np.asarray(residual, dtype=np.float64)
    return (original - center) / scale


def collect_candidate_runs(candidate_mask, gap: int = 1) -> list[tuple[int, int]]:
    """把候选点按允许间隔合并为连续段，返回闭区间列表。"""
    indices = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    if indices.size == 0:
        return []
    splits = np.split(indices, np.flatnonzero(np.diff(indices) > int(gap) + 1) + 1)
    return [(int(run[0]), int(run[-1])) for run in splits]


def _per_row_median(intensities):
    """返回每条谱的稳健中值，零/负值用 1 兜底以支持归一化。"""
    medians = np.median(np.asarray(intensities, dtype=np.float64), axis=1)
    return np.where(medians > _EPS, medians, 1.0)


def build_folder_cosmic_reference(
    intensities,
    options: FolderCosmicOptions,
    valid_mask=None,
) -> FolderCosmicReference | None:
    """从文件夹内谱构建中值参考、真实峰 guard 和跨谱归一化基准。"""
    values = np.asarray(intensities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return None
    rows = values
    median = np.median(rows, axis=0)
    smoothed = median_filter_1d(median, options.reference_smooth_window, dtype=np.float64)
    elevation = median - smoothed
    mad = 1.4826 * float(np.median(np.abs(elevation - np.median(elevation))))
    span = float(np.ptp(median))
    threshold = max(options.guard_mad_factor * mad, options.guard_ptp_fraction * span)
    threshold = max(threshold, _EPS)
    guard_mask = elevation > threshold
    if valid_mask is not None:
        guard_mask &= np.asarray(valid_mask, dtype=bool)

    row_base = _per_row_median(rows)[:, None]
    normalized_rows = rows / row_base
    normalized_median = np.median(normalized_rows, axis=0)
    normalized_mad = 1.4826 * np.median(np.abs(normalized_rows - normalized_median), axis=0)
    normalized_mad = np.maximum(normalized_mad, 1e-4)
    return FolderCosmicReference(
        median=median,
        smoothed=smoothed,
        guard_mask=guard_mask,
        guard_threshold=float(threshold),
        normalized_median=normalized_median,
        normalized_mad=normalized_mad,
    )


def _fill_run(values, start: int, stop: int, neighbor_points: int) -> bool:
    """用两侧中值线性插值替换一段；无可用邻点时返回 True 表示跳过。"""
    count = stop - start + 1
    left = values[max(start - neighbor_points, 0):start]
    right = values[stop + 1:stop + 1 + neighbor_points]
    if left.size and right.size:
        values[start:stop + 1] = np.linspace(
            float(np.median(left)),
            float(np.median(right)),
            count + 2,
        )[1:-1]
        return False
    if left.size:
        values[start:stop + 1] = float(np.median(left))
        return False
    if right.size:
        values[start:stop + 1] = float(np.median(right))
        return False
    return True


def remove_spectrum_spikes(
    values,
    reference: FolderCosmicReference,
    options: FolderCosmicOptions,
    valid_mask=None,
) -> tuple[np.ndarray, SpectrumCosmicStats]:
    """逐谱检测局部中值离群点，并按连续段插值替换。"""
    cleaned = np.asarray(values, dtype=np.float64).copy()
    valid = np.ones(cleaned.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    guard = reference.guard_mask & valid
    guarded_hit = np.zeros(cleaned.shape, dtype=bool)
    removed_points = 0
    removed_runs = 0
    edge_skipped = 0
    max_z = 0.0
    iterations = 0

    for _ in range(int(options.max_iterations)):
        local = median_filter_1d(cleaned, options.local_window, dtype=np.float64)
        z = compute_robust_z(cleaned - local, valid)
        if z is None:
            break
        above = (z > options.z_threshold) & valid
        guarded_hit |= above & guard
        runs = collect_candidate_runs(above & ~guard, gap=options.run_gap)
        if not runs:
            break
        iterations += 1
        changed = False
        for start, stop in runs:
            max_z = max(max_z, float(np.max(z[start:stop + 1])))
            if _fill_run(cleaned, start, stop, options.neighbor_points):
                edge_skipped += 1
                continue
            removed_points += stop - start + 1
            removed_runs += 1
            changed = True
        if not changed:
            break

    stats = SpectrumCosmicStats(
        removed_points=removed_points,
        removed_runs=removed_runs,
        guarded_points=int(guarded_hit.sum()),
        max_z=max_z,
        iterations=iterations,
        edge_skipped_runs=edge_skipped,
    )
    return cleaned, stats


def remove_spectrum_plateau_leaks(
    values,
    reference: FolderCosmicReference,
    options: FolderCosmicOptions,
    valid_mask=None,
) -> tuple[np.ndarray, SpectrumCosmicStats]:
    """按跨谱归一化中值补充清理平台型/多像素射线。"""
    cleaned = np.asarray(values, dtype=np.float64).copy()
    valid = np.ones(cleaned.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    guard = reference.guard_mask & valid
    guarded_hit = np.zeros(cleaned.shape, dtype=bool)
    removed_points = 0
    removed_runs = 0
    plateau_runs = 0
    edge_skipped = 0
    max_z = 0.0
    iterations = 0

    for _ in range(int(options.max_iterations)):
        base = float(np.median(cleaned[valid])) if valid.any() else 0.0
        if base <= _EPS:
            break
        normalized = cleaned / base
        z = (normalized - reference.normalized_median) / np.maximum(reference.normalized_mad, _EPS)
        above = (z > options.plateau_z_threshold) & valid
        guarded_hit |= above & guard
        ratio = normalized / np.maximum(reference.normalized_median, _EPS)
        runs = collect_candidate_runs(above & ~guard, gap=options.run_gap)
        if not runs:
            break
        iterations += 1
        changed = False
        for start, stop in runs:
            max_z = max(max_z, float(np.max(z[start:stop + 1])))
            if stop - start + 1 > options.plateau_run_max:
                continue
            if guard[start:stop + 1].any():
                continue
            inside = float(np.max(ratio[start:stop + 1]))
            outside = np.concatenate([
                ratio[max(start - _PLATEAU_OUTSIDE_POINTS, 0):start],
                ratio[stop + 1:stop + 1 + _PLATEAU_OUTSIDE_POINTS],
            ])
            if inside < options.plateau_inside_ratio:
                continue
            if outside.size and float(np.max(outside)) > options.plateau_outside_ratio:
                continue
            if _fill_run(cleaned, start, stop, options.neighbor_points):
                edge_skipped += 1
                continue
            removed_points += stop - start + 1
            removed_runs += 1
            plateau_runs += 1
            changed = True
        if not changed:
            break

    stats = SpectrumCosmicStats(
        removed_points=removed_points,
        removed_runs=removed_runs,
        plateau_runs=plateau_runs,
        guarded_points=int(guarded_hit.sum()),
        max_z=max_z,
        iterations=iterations,
        edge_skipped_runs=edge_skipped,
    )
    return cleaned, stats


def remove_folder_cosmic_rays(
    intensities,
    options: FolderCosmicOptions,
    valid_mask=None,
) -> FolderCosmicResult:
    """对同一文件夹的全部谱执行去尖峰，返回清洗结果与统计。"""
    values = np.asarray(intensities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"文件夹光谱必须是二维数组，当前维度为 {values.ndim}")
    clean_copy = values.copy()
    zero_stats = tuple(SpectrumCosmicStats() for _ in range(values.shape[0]))
    if not options.enable:
        return FolderCosmicResult("disabled", clean_copy, zero_stats, None, values.shape[0])
    if values.shape[0] < options.min_spectra:
        return FolderCosmicResult("skipped_small", clean_copy, zero_stats, None, values.shape[0])

    reference = build_folder_cosmic_reference(values, options, valid_mask)
    if reference is None:
        return FolderCosmicResult("skipped_small", clean_copy, zero_stats, None, values.shape[0])

    cleaned = values.copy()
    stats: list[SpectrumCosmicStats] = []
    for index in range(values.shape[0]):
        spike_cleaned, spike_stats = remove_spectrum_spikes(
            values[index], reference, options, valid_mask,
        )
        plateau_cleaned, plateau_stats = remove_spectrum_plateau_leaks(
            spike_cleaned, reference, options, valid_mask,
        )
        cleaned[index] = plateau_cleaned
        stats.append(replace(
            spike_stats,
            removed_points=spike_stats.removed_points + plateau_stats.removed_points,
            removed_runs=spike_stats.removed_runs + plateau_stats.removed_runs,
            plateau_runs=plateau_stats.plateau_runs,
            guarded_points=spike_stats.guarded_points + plateau_stats.guarded_points,
            max_z=max(spike_stats.max_z, plateau_stats.max_z),
            iterations=spike_stats.iterations + plateau_stats.iterations,
            edge_skipped_runs=spike_stats.edge_skipped_runs + plateau_stats.edge_skipped_runs,
        ))

    return FolderCosmicResult("applied", cleaned, tuple(stats), reference, values.shape[0])
