"""审核阶段间传递的结构化记录。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RawRecord:
    """保存一条原始谱在 Stage1 中的路径、来源和质量指标。"""

    path: Path
    rel_path: str
    genus: str
    folder: str
    prefix: str
    source_rel_path: str = ""
    state: str = "keep"
    reasons: tuple[str, ...] = ()
    points: int = 0
    coverage: float = math.nan
    malformed_lines: int = 0
    longest_flat_points: int = 0
    saturation_points: int = 0
    noise_ratio: float = math.nan
    wavenumbers: np.ndarray | None = None
    intensities: np.ndarray | None = None


@dataclass
class SimilarityRecord:
    """保存近邻相似性阶段使用的预处理谱和判定指标。"""

    raw: RawRecord
    spectrum: np.ndarray | None = None
    state: str = "keep"
    reasons: tuple[str, ...] = ()
    reference_count: int = 0
    neighbor_count: int = 0
    neighbor_corr: float = math.nan
    rmse: float = math.nan


@dataclass(frozen=True)
class StageResult:
    """汇总单个审核阶段的报告位置与候选移动数量。"""

    stage_dir: Path
    record_count: int
    candidate_count: int
    moved_count: int
