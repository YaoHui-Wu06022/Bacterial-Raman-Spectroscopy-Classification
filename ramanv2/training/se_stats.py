"""验证集 SE 通道缩放统计。"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ramanv2.modeling.layers import SEBlock1D


def build_se_stats_accumulator(model: torch.nn.Module) -> dict[str, dict[str, int | torch.Tensor]]:
    """为启用的 SE 模块建立与模型同设备的 float64 累计容器。"""
    device = next(model.parameters()).device
    accumulators: dict[str, dict[str, int | torch.Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_enable:
            continue
        channels = module.fc[-2].out_features
        accumulators[name] = {
            "sample_count": 0,
            "channel_sum": torch.zeros(channels, dtype=torch.float64, device=device),
            "channel_sq_sum": torch.zeros(channels, dtype=torch.float64, device=device),
            "channel_min": torch.full((channels,), float("inf"), dtype=torch.float64, device=device),
            "channel_max": torch.full((channels,), float("-inf"), dtype=torch.float64, device=device),
        }
    return accumulators


def register_se_scale_hooks(
    model: torch.nn.Module,
    batch_scales: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    """注册前向 hook，在每个 SE 模块处缓存当前 batch 的通道缩放系数。"""
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_enable:
            continue
        hooks.append(module.register_forward_hook(_build_scale_hook(name, batch_scales)))
    return hooks


def _build_scale_hook(
    module_name: str,
    batch_scales: dict[str, torch.Tensor],
) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
    """构造一个从 SE 模块输入计算缩放系数的前向 hook。"""
    def cache_scale(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _: torch.Tensor,
    ) -> None:
        if not inputs:
            return
        if not isinstance(module, SEBlock1D):
            return
        batch_scales[module_name] = module.build_scale(inputs[0]).detach().to(torch.float64)

    return cache_scale


def accumulate_se_stats(
    accumulators: dict[str, dict[str, int | torch.Tensor]],
    batch_scales: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> None:
    """仅将标签有效样本的 SE 缩放系数合并到累计统计。

    累计量始终停留在模型设备上，避免每个 batch 逐模块 `.cpu()` 造成同步。
    """
    if not batch_scales:
        return
    for name, scales in batch_scales.items():
        valid_scales = scales[valid_mask]
        if valid_scales.numel() == 0:
            continue
        stats = accumulators[name]
        stats["sample_count"] = int(stats["sample_count"]) + int(valid_scales.size(0))
        stats["channel_sum"] = stats["channel_sum"] + valid_scales.sum(dim=0)
        stats["channel_sq_sum"] = stats["channel_sq_sum"] + (valid_scales * valid_scales).sum(dim=0)
        stats["channel_min"] = torch.minimum(stats["channel_min"], valid_scales.min(dim=0).values)
        stats["channel_max"] = torch.maximum(stats["channel_max"], valid_scales.max(dim=0).values)


def finalize_se_stats(
    accumulators: dict[str, dict[str, int | torch.Tensor]],
) -> dict[str, dict[str, int | torch.Tensor]]:
    """将累计量转换为可保存的均值、标准差、最小值与最大值。

    只在收尾时统一搬回 CPU，sidecar 因此与训练设备无关。
    """
    results: dict[str, dict[str, int | torch.Tensor]] = {}
    for name, stats in accumulators.items():
        sample_count = int(stats["sample_count"])
        if sample_count <= 0:
            continue
        channel_mean = stats["channel_sum"] / sample_count
        channel_variance = stats["channel_sq_sum"] / sample_count - channel_mean * channel_mean
        results[name] = {
            "sample_count": sample_count,
            "channel_mean": channel_mean.to(torch.float32).cpu(),
            "channel_std": torch.sqrt(torch.clamp(channel_variance, min=0.0)).to(torch.float32).cpu(),
            "channel_min": stats["channel_min"].to(torch.float32).cpu(),
            "channel_max": stats["channel_max"].to(torch.float32).cpu(),
        }
    return results
