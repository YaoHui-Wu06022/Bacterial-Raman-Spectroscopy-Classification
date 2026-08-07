import torch

from raman.model import SEBlock1D


def init_se_stats_accumulator(model):
    """为所有启用的 SEBlock 构造累计统计容器"""
    se_accumulators = {}
    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_use:
            continue
        out_channels = module.fc[-2].out_features
        se_accumulators[name] = {
            "sample_count": 0,
            "channel_sum": torch.zeros(out_channels, dtype=torch.float64),
            "channel_sq_sum": torch.zeros(out_channels, dtype=torch.float64),
            "channel_min": torch.full((out_channels,), float("inf"), dtype=torch.float64),
            "channel_max": torch.full((out_channels,), float("-inf"), dtype=torch.float64),
        }
    return se_accumulators

def attach_se_scale_hooks(model, batch_scales):
    """在验证前向期间缓存每个 SEBlock 的当前 batch scale"""
    hooks = []
    if not batch_scales and not any(isinstance(module, SEBlock1D) and module.se_use for _, module in model.named_modules()):
        return hooks

    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_use:
            continue

        def hook(_module, inputs, _output, block_name=name):
            if not inputs:
                return
            x = inputs[0]
            batch_scales[block_name] = _module._compute_scale(x).detach().cpu().to(torch.float64)

        hooks.append(module.register_forward_hook(hook))

    return hooks

def accumulate_se_stats(se_accumulators, batch_scales, valid_mask):
    """把当前 batch 中有效样本的 scale 合并进累计统计"""
    if not batch_scales:
        return

    valid_mask_cpu = valid_mask.detach().cpu()
    if not valid_mask_cpu.any():
        return

    for name, scale in batch_scales.items():
        valid_scale = scale[valid_mask_cpu]
        if valid_scale.numel() == 0:
            continue
        stats = se_accumulators[name]
        stats["sample_count"] += int(valid_scale.size(0))
        stats["channel_sum"] += valid_scale.sum(dim=0)
        stats["channel_sq_sum"] += (valid_scale * valid_scale).sum(dim=0)
        stats["channel_min"] = torch.minimum(stats["channel_min"], valid_scale.min(dim=0).values)
        stats["channel_max"] = torch.maximum(stats["channel_max"], valid_scale.max(dim=0).values)

def finalize_se_stats(se_accumulators):
    """将累计量还原成最终 SE 统计结果"""
    se_stats = {}
    for name, stats in se_accumulators.items():
        sample_count = int(stats["sample_count"])
        if sample_count <= 0:
            continue
        channel_mean = stats["channel_sum"] / sample_count
        channel_var = stats["channel_sq_sum"] / sample_count - channel_mean * channel_mean
        channel_var = torch.clamp(channel_var, min=0.0)
        channel_std = torch.sqrt(channel_var)
        se_stats[name] = {
            "sample_count": sample_count,
            "channel_mean": channel_mean.to(torch.float32),
            "channel_std": channel_std.to(torch.float32),
            "channel_min": stats["channel_min"].to(torch.float32),
            "channel_max": stats["channel_max"].to(torch.float32),
        }
    return se_stats

