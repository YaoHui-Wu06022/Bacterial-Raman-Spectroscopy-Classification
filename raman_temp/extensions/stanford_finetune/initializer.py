"""Stanford 权重加载、分类头重置与逐模块冻结。"""

from __future__ import annotations

from pathlib import Path

import torch


class TransferInitializer:
    """在目标模型创建后加载骨干，并维持配置指定的训练模块。"""

    def __init__(self, source_model_path: Path | str, trainable_modules: tuple[str, ...]) -> None:
        self.source_model_path = Path(source_model_path)
        self.trainable_modules = tuple(trainable_modules)
        self.reports: dict[str, dict] = {}

    def initialize(self, model: torch.nn.Module, task) -> None:
        """严格加载非分类头权重，重置 head 并设置可训练参数。"""
        source_state = _load_model_state(self.source_model_path)
        target_state = model.state_dict()
        loaded_state = {}
        incompatible = []
        unexpected = []
        skipped_head = []
        for key, value in source_state.items():
            if key.startswith("head."):
                skipped_head.append(key)
                continue
            target_value = target_state.get(key)
            if target_value is None:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                incompatible.append(key)
                continue
            loaded_state[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(loaded_state, strict=False)
        missing_required = [key for key in missing_keys if not key.startswith("head.")]
        if missing_required or incompatible or unexpected or unexpected_keys:
            raise ValueError(
                "来源骨干不能完整加载："
                f"missing={missing_required}, incompatible={incompatible}, "
                f"unexpected={unexpected}, load_unexpected={list(unexpected_keys)}"
            )
        trainable_roots = _resolve_trainable_roots(model, self.trainable_modules)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.split(".", 1)[0] in trainable_roots
        self.reports[task.model_tag] = {
            "source_model_path": str(self.source_model_path),
            "target_num_classes": int(task.num_classes),
            "loaded_parameter_keys": len(loaded_state),
            "skipped_source_head_keys": skipped_head,
            "head_reset": True,
            "trainable_modules": sorted(trainable_roots),
            "frozen_modules": sorted(
                name for name, _module in model.named_children() if name not in trainable_roots
            ),
        }

    def apply_training_mode(self, model: torch.nn.Module) -> None:
        """让冻结模块维持 eval，避免其统计量和随机层状态继续更新。"""
        trainable_roots = _resolve_trainable_roots(model, self.trainable_modules)
        for name, module in model.named_children():
            module.train(name in trainable_roots)


def _load_model_state(model_path: Path) -> dict:
    """读取纯 state_dict 或 checkpoint 中的模型参数字典。"""
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    if not isinstance(payload, dict):
        raise TypeError(f"来源权重不是 state_dict：{model_path}")
    return payload


def _resolve_trainable_roots(model: torch.nn.Module, selected: tuple[str, ...]) -> set[str]:
    """验证逐模块配置，并始终让新分类头参与训练。"""
    available = {name for name, _module in model.named_children()}
    invalid = sorted(set(selected) - available - {"head"})
    if invalid:
        raise ValueError(f"未知可训练模块：{invalid}；可选值：{sorted(available)}")
    return {"head", *selected}
