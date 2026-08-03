"""模型中间层的 activation × gradient 归因。"""

from __future__ import annotations

import torch


ANALYSIS_LAYER_NAMES = ("input_proj", "layer1", "layer2", "layer3", "layer4", "transformer", "lstm")


def compute_layer_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """计算可分析层的归一化 ``mean(abs(activation × gradient))`` 分数。"""
    modules = {
        name: module
        for name, module in model.named_modules()
        if name in ANALYSIS_LAYER_NAMES
    }
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
