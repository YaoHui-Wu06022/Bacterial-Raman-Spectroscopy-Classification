"""训练期验证、指标汇总与 SE 统计编排。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ramanv2.common.metrics import compute_classification_metrics
from ramanv2.core.hierarchy import mask_logits_by_parent, select_level_targets
from ramanv2.training.se_stats import (
    accumulate_se_stats,
    build_se_stats_accumulator,
    finalize_se_stats,
    register_se_scale_hooks,
)


def evaluate_validation_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    level_index: int,
    label_map_tensor: torch.Tensor | None = None,
    parent_level_index: int | None = None,
    parent_to_children: Mapping[int | str, Sequence[int]] | None = None,
    amp_enable: bool = False,
) -> tuple[float, float, dict[str, float], dict[str, dict[str, int | torch.Tensor]]]:
    """验证单层模型，统一处理局部标签、父类约束、指标与 SE 统计。"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    num_classes: int | None = None
    se_accumulators = build_se_stats_accumulator(model)
    batch_scales: dict[str, torch.Tensor] = {}
    se_hooks = register_se_scale_hooks(model, batch_scales)

    try:
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_scales.clear()
                with torch.autocast(device_type=device.type, enabled=amp_enable):
                    logits = model(inputs)
                level_targets = select_level_targets(labels, level_index)
                level_targets = _map_local_targets(level_targets, label_map_tensor)
                logits, valid_parent_mask = _apply_parent_mask(
                    logits,
                    labels,
                    parent_level_index,
                    parent_to_children,
                )
                valid_mask = (level_targets >= 0) & valid_parent_mask
                accumulate_se_stats(se_accumulators, batch_scales, valid_mask)
                if not valid_mask.any():
                    continue

                valid_logits = logits[valid_mask]
                valid_targets = level_targets[valid_mask]
                if num_classes is None:
                    num_classes = valid_logits.size(1)
                loss = criterion(valid_logits.float(), valid_targets)
                batch_size = valid_targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                predictions.append(valid_logits.argmax(1).detach().cpu().numpy())
                targets.append(valid_targets.detach().cpu().numpy())
    finally:
        for hook in se_hooks:
            hook.remove()

    se_stats = finalize_se_stats(se_accumulators)
    if num_classes is None or not targets:
        metrics = {"accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0}
        return 0.0, 0.0, metrics, se_stats
    metrics = compute_classification_metrics(
        np.concatenate(targets, axis=0),
        np.concatenate(predictions, axis=0),
        range(num_classes),
    )
    return total_loss / max(total_samples, 1), metrics["accuracy"], metrics, se_stats


def _map_local_targets(
    targets: torch.Tensor,
    label_map_tensor: torch.Tensor | None,
) -> torch.Tensor:
    """将全局标签映射为父类子模型使用的局部标签，并保留无效标签。"""
    if label_map_tensor is None:
        return targets
    invalid_mask = targets < 0
    local_targets = label_map_tensor[targets.clamp_min(0)]
    local_targets[invalid_mask] = -1
    return local_targets


def _apply_parent_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    parent_level_index: int | None,
    parent_to_children: Mapping[int | str, Sequence[int]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """父类约束启用时按完整多层标签遮罩子类 logits。"""
    if parent_level_index is None or parent_to_children is None:
        valid_mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, valid_mask
    if labels.ndim != 2:
        raise ValueError("parent_level_index 需要完整的多层标签输入")
    return mask_logits_by_parent(logits, labels[:, parent_level_index], parent_to_children)
