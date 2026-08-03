"""审核阶段使用的严格原始谱读取。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_raw_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """读取原始两列谱并统计无法解析的非空行，供质量与平移阶段共用。"""
    wavenumbers, intensities = [], []
    malformed_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            values = line.strip().split()
            if not values:
                continue
            if len(values) != 2:
                malformed_lines += 1
                continue
            try:
                wavenumbers.append(float(values[0]))
                intensities.append(float(values[1]))
            except ValueError:
                malformed_lines += 1
    return (
        np.asarray(wavenumbers, dtype=np.float64),
        np.asarray(intensities, dtype=np.float64),
        malformed_lines,
    )
