"""训练、评估和推理共用的运行时解析。"""

from __future__ import annotations

import torch


def resolve_device(
    value: torch.device | str | None = None,
    *,
    use_gpu_enable: bool = True,
) -> torch.device:
    """按显式设备、GPU 开关和 CUDA 可用性解析运行设备。"""
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if use_gpu_enable and torch.cuda.is_available() else "cpu")


__all__ = ["resolve_device"]
