"""Stage2 与 Stage3 共用的审核光谱预处理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman_temp.audit.config import AuditConfig, resolve_audit_config
from raman_temp.spectra.axis import build_wn_ref
from raman_temp.data.config import resolve_cosmic_options
from raman_temp.data.io import read_arc_data
from raman_temp.spectra.normalize import normalize_spectrum
from raman_temp.spectra.preprocess import preprocess_single_spectrum


@dataclass(frozen=True)
class AuditSpectrum:
    """保存审核使用的统一波数轴、处理谱和 SNV 标准化向量。"""

    wavenumbers: np.ndarray | None
    intensities: np.ndarray | None
    normalized: np.ndarray | None
    skip_reason: str = ""
    cosmic_replaced: int = 0


def preprocess_audit_spectrum(
    path: Path,
    profile,
    config: AuditConfig | None = None,
    reference_wavenumbers: np.ndarray | None = None,
    label: str = "",
) -> AuditSpectrum:
    """按训练一致的预处理参数生成供近邻审核比较的 SNV 谱向量。"""
    audit_config = resolve_audit_config(config)
    input_config = audit_config.input
    cleaning_config = audit_config.cleaning
    raw_wavenumbers, raw_intensities = read_arc_data(path)
    if not raw_wavenumbers.size or not raw_intensities.size:
        return AuditSpectrum(None, None, None, "read_failed")
    axis = (
        build_wn_ref(
            input_config.cut_min,
            input_config.cut_max,
            input_config.target_points,
        )
        if reference_wavenumbers is None
        else reference_wavenumbers
    )
    options = resolve_cosmic_options(profile, cleaning_config, label)
    wavenumbers, intensities, cosmic_stats = preprocess_single_spectrum(
        raw_wavenumbers,
        raw_intensities,
        cut_min=input_config.cut_min,
        cut_max=input_config.cut_max,
        reference_wavenumbers=axis,
        bad_bands=input_config.bad_bands,
        baseline_method=cleaning_config.baseline_method,
        baseline_lam=cleaning_config.baseline_lam,
        baseline_asls_p=cleaning_config.baseline_asls_p,
        baseline_max_iter=cleaning_config.baseline_max_iter,
        baseline_fit_min=cleaning_config.baseline_fit_min,
        baseline_fit_max=cleaning_config.baseline_fit_max,
        **options,
    )
    if wavenumbers is None or intensities is None:
        return AuditSpectrum(None, None, None, "preprocess_failed")
    normalized = normalize_spectrum(intensities, "snv")
    return AuditSpectrum(
        np.asarray(wavenumbers, dtype=np.float32),
        np.asarray(intensities, dtype=np.float32),
        np.asarray(normalized, dtype=np.float32),
        cosmic_replaced=int(cosmic_stats),
    )
