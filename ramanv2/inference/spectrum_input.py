"""独立推理的输入目录、光谱枚举与单谱输入构建。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ramanv2.common.arc_data import list_arc_data_paths, read_arc_data
from ramanv2.common.naming import build_natural_key
from ramanv2.core.input_spec import InputSpec
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.spectra.bad_bands import build_valid_mask


def resolve_input_dirs(
    input_dir: Path | str,
    one_dir: Path | str | None = None,
) -> list[Path]:
    """列出待预测目录，或解析指定的单个目录。"""
    root_dir = Path(input_dir).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"推理输入目录不存在：{root_dir}")
    if one_dir is None:
        return sorted(path for path in root_dir.iterdir() if path.is_dir())
    target_dir = Path(one_dir)
    if not target_dir.is_absolute():
        target_dir = root_dir / target_dir
    if not target_dir.is_dir():
        raise FileNotFoundError(f"指定推理目录不存在：{target_dir}")
    return [target_dir]


def list_spectrum_paths(input_dir: Path | str) -> list[Path]:
    """按文件名自然顺序列出一个目录中的 `.arc_data` 文件。"""
    return list_arc_data_paths(input_dir, key=lambda path: build_natural_key(path.name))


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
