"""验证集 SE 通道缩放统计。"""

from __future__ import annotations

from collections.abc import Callable

import torch

from raman_temp.modeling.layers import SEBlock1D


def build_se_stats_accumulator(model: torch.nn.Module) -> dict[str, dict[str, int | torch.Tensor]]:
    """为启用的 SE 模块建立 float64 累计容器。"""
    accumulators: dict[str, dict[str, int | torch.Tensor]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_enable:
            continue
        channels = module.fc[-2].out_features
        accumulators[name] = {
            "sample_count": 0,
            "channel_sum": torch.zeros(channels, dtype=torch.float64),
            "channel_sq_sum": torch.zeros(channels, dtype=torch.float64),
            "channel_min": torch.full((channels,), float("inf"), dtype=torch.float64),
            "channel_max": torch.full((channels,), float("-inf"), dtype=torch.float64),
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
        batch_scales[module_name] = module.build_scale(inputs[0]).detach().cpu().to(torch.float64)

    return cache_scale


def accumulate_se_stats(
    accumulators: dict[str, dict[str, int | torch.Tensor]],
    batch_scales: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> None:
    """仅将标签有效样本的 SE 缩放系数合并到累计统计。"""
    if not batch_scales:
        return
    valid_mask_cpu = valid_mask.detach().cpu()
    if not valid_mask_cpu.any():
        return
    for name, scales in batch_scales.items():
        valid_scales = scales[valid_mask_cpu]
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
    """将累计量转换为可保存的均值、标准差、最小值与最大值。"""
    results: dict[str, dict[str, int | torch.Tensor]] = {}
    for name, stats in accumulators.items():
        sample_count = int(stats["sample_count"])
        if sample_count <= 0:
            continue
        channel_mean = stats["channel_sum"] / sample_count
        channel_variance = stats["channel_sq_sum"] / sample_count - channel_mean * channel_mean
        results[name] = {
            "sample_count": sample_count,
            "channel_mean": channel_mean.to(torch.float32),
            "channel_std": torch.sqrt(torch.clamp(channel_variance, min=0.0)).to(torch.float32),
            "channel_min": stats["channel_min"].to(torch.float32),
            "channel_max": stats["channel_max"].to(torch.float32),
        }
    return results
