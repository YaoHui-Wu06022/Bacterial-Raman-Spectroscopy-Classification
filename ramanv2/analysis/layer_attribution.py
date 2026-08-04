"""模型中间层的 activation × gradient 归因。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ramanv2.modeling.layers import ResidualBottleneck1D


def compute_layer_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """计算可分析层的归一化 ``mean(abs(activation × gradient))`` 分数。"""
    modules = _collect_attribution_modules(model)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(_build_forward_hook(activations, name)))
        hooks.append(module.register_full_backward_hook(_build_backward_hook(gradients, name)))
    try:
        values = inputs.detach().requires_grad_(True)
        logits = model(values)
        model.zero_grad(set_to_none=True)
        logits.gather(1, labels[:, None]).sum().backward()
        scores = {
            name: float((activations[name] * gradients[name]).abs().mean().item())
            for name in activations
            if name in gradients
        }
    finally:
        for hook in hooks:
            hook.remove()
    total = sum(scores.values())
    return {name: value / (total + 1e-8) for name, value in scores.items()}


def merge_layer_attribution_scores(layer_scores: dict[str, float]) -> dict[str, float]:
    """将细粒度残差块和编码器层归并为 stage 摘要。"""
    groups: dict[str, float] = {}
    for name, value in layer_scores.items():
        group_name = _resolve_layer_group_name(name)
        groups[group_name] = groups.get(group_name, 0.0) + value
    total = sum(groups.values())
    return {name: value / (total + 1e-8) for name, value in groups.items()}


def _collect_attribution_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """收集残差块、编码器层与输入投影等可解释模块。"""
    modules: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, (ResidualBottleneck1D, nn.TransformerEncoderLayer, nn.LSTM)):
            modules[name] = module
        elif name in ("conv1", "input_proj"):
            modules[name] = module
    if modules:
        return modules
    return {
        name: module
        for name, module in model.named_modules()
        if name in ("layer1", "layer2", "layer3", "layer4", "transformer", "lstm")
    }


def _resolve_layer_group_name(name: str) -> str:
    """将模块路径映射为日志使用的 stage 名称。"""
    if name.startswith("transformer"):
        return "transformer"
    if name.startswith("lstm"):
        return "lstm"
    return name.split(".", 1)[0]


def _build_forward_hook(storage: dict[str, torch.Tensor], name: str):
    """构建保存层输出的临时 hook。"""
    def save_output(_module, _inputs, output) -> None:
        storage[name] = output[0] if isinstance(output, tuple) else output

    return save_output


def _build_backward_hook(storage: dict[str, torch.Tensor], name: str):
    """构建保存层输出梯度的临时 hook。"""
    def save_gradient(_module, _grad_inputs, grad_outputs) -> None:
        storage[name] = grad_outputs[0]

    return save_gradient
