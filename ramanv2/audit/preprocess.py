"""近邻相似性审核共用的单谱预处理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ramanv2.common.arc_data import read_arc_data
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import preprocess_single_spectrum


__all__ = ("ComparisonSpectrum", "preprocess_comparison_spectrum")


@dataclass(frozen=True)
class ComparisonSpectrum:
    """保存统一波数轴、预处理强度和 SNV 比较向量。"""

    wavenumbers: np.ndarray | None
    intensities: np.ndarray | None
    normalized: np.ndarray | None
    skip_reason: str = ""


def preprocess_comparison_spectrum(
    path: Path,
    profile_id: str,
    input_config,
    build_config,
    reference_wavenumbers: np.ndarray | None = None,
) -> ComparisonSpectrum:
    """按训练参数预处理一条谱，生成近邻比较使用的 SNV 向量。"""
    raw_wavenumbers, raw_intensities = read_arc_data(path)
    if not raw_wavenumbers.size or not raw_intensities.size:
        return ComparisonSpectrum(None, None, None, "read_failed")
    axis = (
        build_wn_ref(input_config.cut_min, input_config.cut_max, input_config.target_points)
        if reference_wavenumbers is None
        else reference_wavenumbers
    )
    options = build_config.build_cosmic_ray_options(profile_id)
    wavenumbers, intensities, _ = preprocess_single_spectrum(
        raw_wavenumbers,
        raw_intensities,
        cut_min=input_config.cut_min,
        cut_max=input_config.cut_max,
        reference_wavenumbers=axis,
        bad_bands=input_config.bad_bands,
        baseline_method=build_config.baseline_method,
        baseline_lam=build_config.baseline_lam,
        baseline_asls_p=build_config.baseline_asls_p,
        baseline_max_iter=build_config.baseline_max_iter,
        baseline_fit_min=build_config.baseline_fit_min,
        baseline_fit_max=build_config.baseline_fit_max,
        **options,
    )
    if wavenumbers is None or intensities is None:
        return ComparisonSpectrum(None, None, None, "preprocess_failed")
    return ComparisonSpectrum(
        np.asarray(wavenumbers, dtype=np.float32),
        np.asarray(intensities, dtype=np.float32),
        np.asarray(normalize_spectrum(intensities, "snv"), dtype=np.float32),
    )
