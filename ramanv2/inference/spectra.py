"""独立推理的单谱读取、长度校验与模型输入构建。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ramanv2.core.input_spec import InputSpec
from ramanv2.data.input import InputPreprocessor
from ramanv2.data.io import read_arc_data
from ramanv2.spectra.bands import build_valid_mask


def build_inference_preprocessor(
    input_spec: InputSpec,
    device: torch.device | str,
) -> InputPreprocessor:
    """构建独立推理复用的确定性输入预处理器。"""
    return InputPreprocessor(input_spec, device)


def preprocess_spectrum_path(
    spectrum_path: Path | str,
    preprocessor: InputPreprocessor,
    bad_bands: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """读取单条光谱并转换为 `[1, C, L]` 模型输入。"""
    wavenumbers, intensities = read_arc_data(spectrum_path)
    if not wavenumbers.size or not intensities.size:
        raise ValueError(f"光谱文件没有有效数据：{spectrum_path}")
    values = _resolve_input_intensity(
        wavenumbers,
        intensities,
        preprocessor.input_spec.point_count,
        bad_bands,
        spectrum_path,
    )
    return preprocessor.preprocess_intensity(values)


def _resolve_input_intensity(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    point_count: int,
    bad_bands: tuple[tuple[float, float], ...],
    source_path: Path | str,
) -> np.ndarray:
    """优先使用完整输入；必要时按坏段筛选到模型点数。"""
    if int(intensities.size) == int(point_count):
        return np.asarray(intensities, dtype=np.float32)
    valid_mask = build_valid_mask(wavenumbers, bad_bands)
    if valid_mask is not None and int(valid_mask.sum()) == int(point_count):
        return np.asarray(intensities[valid_mask], dtype=np.float32)
    raise ValueError(
        f"输入长度不匹配：{source_path}，实际={intensities.size}，期望={point_count}"
    )
