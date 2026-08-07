"""共享近邻相似性审核。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ramanv2.audit.records import CleanRecord
from ramanv2.audit.preprocess import preprocess_comparison_spectrum


@dataclass(frozen=True)
class NeighborConfig:
    """定义同组近邻审核所需的最少参考谱数量。"""

    minimum_references: int = 8


def compute_mad_limit(values: np.ndarray, direction: str) -> float:
    """根据缩放 MAD 计算单侧异常阈值。"""
    median = float(np.median(values))
    scale = max(float(np.median(np.abs(values - median))) * 1.4826, 1e-6)
    return median - 3.5 * scale if direction == "low" else median + 3.5 * scale


def score_neighbor_group(records: list[CleanRecord], config: NeighborConfig) -> None:
    """以相关性、RMSE 和 MAD 评分一个已预处理的同组光谱集合。"""
    if len(records) - 1 < config.minimum_references:
        for record in records:
            record.state = "insufficient_reference"
        return
    spectra = np.stack([record.spectrum for record in records])
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    correlation = (centered @ centered.T) / np.maximum(np.outer(norms, norms), 1e-8)
    neighbor_count = max(3, int(math.sqrt(len(records) - 1)))
    for index, record in enumerate(records):
        order = np.argsort(-correlation[index])
        neighbors = [value for value in order if value != index][:neighbor_count]
        reference = np.median(spectra[neighbors], axis=0)
        record.reference_count = len(records) - 1
        record.neighbor_count = len(neighbors)
        record.neighbor_corr = float(np.median(correlation[index, neighbors]))
        record.rmse = float(np.sqrt(np.mean((spectra[index] - reference) ** 2)))
    corr_limit = compute_mad_limit(
        np.asarray([record.neighbor_corr for record in records]), "low"
    )
    rmse_limit = compute_mad_limit(
        np.asarray([record.rmse for record in records]), "high"
    )
    for record in records:
        if record.neighbor_corr < corr_limit and record.rmse > rmse_limit:
            record.state = "candidate"
            record.reasons = ("low_neighbor_agreement", "high_neighbor_rmse")


def score_folder_neighbor_groups(records: list[CleanRecord], config: NeighborConfig) -> int:
    """按文件夹分别执行已预处理光谱的近邻评分，并返回参与审核的文件夹数。"""
    folders = {}
    for record in records:
        if record.spectrum is not None:
            folders.setdefault((record.group, record.folder), []).append(record)
    for group in folders.values():
        score_neighbor_group(group, config)
    return len(folders)


def preprocess_similarity_records(
    records: list[CleanRecord],
    profile,
    input_config,
    build_config,
) -> None:
    """按训练参数预处理记录，并保留无法比较的原因。"""
    for record in records:
        spectrum = preprocess_comparison_spectrum(
            record.path,
            profile.profile_id,
            input_config,
            build_config,
        )
        if spectrum.normalized is None:
            record.state = "unscorable"
            record.reasons = (spectrum.skip_reason,)
            continue
        record.spectrum = spectrum.normalized
