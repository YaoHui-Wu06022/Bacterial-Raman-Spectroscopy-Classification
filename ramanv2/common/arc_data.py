"""`.arc_data` 两列文本的基础读取。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_arc_data(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """读取两列文本光谱，忽略格式错误的行。"""
    wavenumbers, intensities = [], []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) != 2:
                continue
            try:
                wavenumbers.append(float(fields[0]))
                intensities.append(float(fields[1]))
            except ValueError:
                continue
    return np.asarray(wavenumbers), np.asarray(intensities)


def read_raw_arc_data(path: Path | str) -> tuple[np.ndarray, np.ndarray, int]:
    """读取原始两列谱并统计无法解析的非空行。"""
    wavenumbers, intensities = [], []
    malformed_lines = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                malformed_lines += 1
                continue
            try:
                wavenumbers.append(float(fields[0]))
                intensities.append(float(fields[1]))
            except ValueError:
                malformed_lines += 1
    return (
        np.asarray(wavenumbers, dtype=np.float64),
        np.asarray(intensities, dtype=np.float64),
        malformed_lines,
    )
