"""清洗阶段共用的单谱记录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


__all__ = ("CleanRecord",)


@dataclass
class CleanRecord:
    """保存一条原始谱的路径、文件夹归属和阶段指标。"""

    path: Path
    rel_path: str
    group: str
    folder: str
    state: str = "keep"
    reasons: tuple[str, ...] = ()
    points: int = 0
    coverage: float = float("nan")
    malformed_lines: int = 0
    longest_flat_points: int = 0
    saturation_points: int = 0
    noise_ratio: float = float("nan")
    spectrum: np.ndarray | None = None
    reference_count: int = 0
    neighbor_count: int = 0
    neighbor_corr: float = float("nan")
    rmse: float = float("nan")
