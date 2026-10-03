"""文件夹级离群审核使用的单谱记录。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


__all__ = ("CleanRecord",)


@dataclass
class CleanRecord:
    """保存一条光谱的路径、文件夹归属和近邻审核指标。"""

    path: Path
    rel_path: str
    group: str
    folder: str
    state: str = "keep"
    reasons: tuple[str, ...] = ()
    spectrum: np.ndarray | None = None
    reference_count: int = 0
    neighbor_count: int = 0
    neighbor_corr: float = float("nan")
    rmse: float = float("nan")
    corr_limit: float = float("nan")
    rmse_limit: float = float("nan")
