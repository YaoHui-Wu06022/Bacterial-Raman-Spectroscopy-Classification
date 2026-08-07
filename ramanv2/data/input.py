"""单条光谱到模型输入张量的确定性构建。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional

from ramanv2.core.input_spec import InputSpec
from ramanv2.data.augmentation import AugmentationSpec, augment_raw_spectrum
from ramanv2.spectra.filters import sg_coeff
from ramanv2.spectra.normalize import normalize_spectrum


def build_sg_kernels(
    input_spec: InputSpec,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按输入规格建立平滑与一阶导数所需的 SG 卷积核。"""
    smooth_kernel = _build_sg_kernel(input_spec.smooth_window, 0, device)
    d1_kernel = _build_sg_kernel(input_spec.d1_window, 1, device)
    return smooth_kernel, d1_kernel


def _build_sg_kernel(
    window_length: int,
    derivative: int,
    device: torch.device | str,
) -> torch.Tensor:
    """将 Savitzky-Golay 系数包装为单通道一维卷积核。"""
    coefficients = sg_coeff(window_length, 3, derivative)
    return torch.tensor(coefficients, dtype=torch.float32, device=device).view(1, 1, -1)


def build_model_input(
    raw_intensity: np.ndarray,
    input_spec: InputSpec,
    smooth_kernel: torch.Tensor,
    d1_kernel: torch.Tensor,
    device: torch.device | str,
    augmentation_spec: AugmentationSpec | None = None,
    augmentation_enable: bool = False,
) -> torch.Tensor:
    """将单条原始强度谱转换为 ``[C, L]`` 模型输入张量。"""
    values = np.asarray(raw_intensity, dtype=np.float32)
    if augmentation_enable:
        if augmentation_spec is None:
            raise ValueError("启用增强时必须提供 AugmentationSpec")
        values = augment_raw_spectrum(values, augmentation_spec)
    values = values.copy()

    source = torch.as_tensor(values, dtype=torch.float32, device=device).view(1, 1, -1)
    normalized = normalize_spectrum(values, input_spec.norm_method)
    signal = torch.as_tensor(normalized, dtype=torch.float32, device=device).view(1, 1, -1)

    smooth_signal = None
    if input_spec.smooth_enable:
        smooth_values = torch.nn.functional.conv1d(
            source,
            smooth_kernel,
            padding=input_spec.smooth_window // 2,
        )[0, 0]
        smooth_values = normalize_spectrum(smooth_values, input_spec.norm_method)
        smooth_signal = smooth_values.view(1, 1, -1)

    d1_signal = None
    if input_spec.d1_enable:
        d1_values = torch.nn.functional.conv1d(
            source,
            smooth_kernel,
            padding=input_spec.smooth_window // 2,
        )
        d1_values = torch.nn.functional.conv1d(
            d1_values,
            d1_kernel,
            padding=input_spec.d1_window // 2,
        )[0, 0]
        d1_values = d1_values / input_spec.step_cm
        d1_values = normalize_spectrum(d1_values, input_spec.norm_method)
        d1_signal = d1_values.view(1, 1, -1)

    return _build_input_channels(signal, input_spec, smooth_signal, d1_signal)


def _build_input_channels(
    signal: torch.Tensor,
    input_spec: InputSpec,
    smooth_signal: torch.Tensor | None,
    d1_signal: torch.Tensor | None,
) -> torch.Tensor:
    """按输入规格堆叠原始、平滑和一阶导数通道。"""
    channels = [signal[0, 0]]
    if input_spec.smooth_enable:
        if smooth_signal is None:
            raise ValueError("smooth_enable=True 时必须提供平滑通道")
        channels.append(smooth_signal[0, 0])
    if input_spec.d1_enable:
        if d1_signal is None:
            raise ValueError("d1_enable=True 时必须提供一阶导数通道")
        channels.append(d1_signal[0, 0])
    if len(channels) != input_spec.in_channels:
        raise ValueError("输入规格通道数与构建结果不一致")
    return torch.stack(channels, dim=0)


class InputPreprocessor:
    """为评估与推理复用确定性单谱输入构建。"""

    def __init__(self, input_spec: InputSpec, device: torch.device | str) -> None:
        self.input_spec = input_spec
        self.device = device
        self.smooth_kernel, self.d1_kernel = build_sg_kernels(input_spec, device)

    def preprocess_intensity(self, values: np.ndarray) -> torch.Tensor:
        """处理一条强度数组并返回批量形状 ``[1, C, L]``。"""
        model_input = build_model_input(
            values,
            self.input_spec,
            self.smooth_kernel,
            self.d1_kernel,
            self.device,
        )
        return model_input.unsqueeze(0)
