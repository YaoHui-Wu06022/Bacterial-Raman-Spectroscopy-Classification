"""训练 DataLoader、优化器和学习率调度器构建。"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ramanv2.training.train_spec import LoaderSpec, OptimizerSpec


STEM_LEARNING_RATE_SCALE = 0.6
HEAD_LEARNING_RATE_SCALE = 1.1


def _seed_worker(_worker_id: int) -> None:
    """按 PyTorch 派生的 worker 种子同步播种 NumPy 与 Python 随机数。

    增强逻辑使用全局 ``np.random``；若不显式播种，各 worker 的增强序列
    既无法复现，fork 启动时还会继承父进程的同一状态而彼此重复。
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_loader(
    dataset: Any,
    loader_spec: LoaderSpec,
    device: torch.device,
) -> DataLoader:
    """按冻结规格构建训练或验证 DataLoader。"""
    num_workers = max(loader_spec.num_workers, 0)
    options: dict[str, Any] = {
        "batch_size": loader_spec.batch_size,
        "shuffle": loader_spec.shuffle_enable,
        "num_workers": num_workers,
    }
    if device.type == "cuda" and loader_spec.pin_memory_enable:
        options["pin_memory"] = True
    if num_workers > 0:
        options["persistent_workers"] = loader_spec.persistent_workers_enable
        options["prefetch_factor"] = loader_spec.prefetch_factor
        options["worker_init_fn"] = _seed_worker
    return DataLoader(dataset, **options)


def build_optimizer(
    model: torch.nn.Module,
    optimizer_spec: OptimizerSpec,
) -> torch.optim.AdamW:
    """按 stem、主体和分类头构建三组 AdamW 学习率。"""
    stem_parameters: list[torch.nn.Parameter] = []
    backbone_parameters: list[torch.nn.Parameter] = []
    head_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("stem_branches"):
            stem_parameters.append(parameter)
        elif name.startswith("head"):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    if stem_parameters:
        parameter_groups.append(
            {
                "params": stem_parameters,
                "lr": optimizer_spec.learning_rate * STEM_LEARNING_RATE_SCALE,
            }
        )
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": optimizer_spec.learning_rate}
        )
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": optimizer_spec.learning_rate * HEAD_LEARNING_RATE_SCALE,
            }
        )
    if not parameter_groups:
        raise ValueError("当前模型没有可训练参数")
    # fused AdamW 只在 CUDA 参数上可用；CPU 路径保持默认实现。
    use_fused_enable = parameter_groups[0]["params"][0].is_cuda
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=optimizer_spec.weight_decay,
        fused=use_fused_enable,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_spec: OptimizerSpec,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """构建与训练时程匹配的余弦退火调度器。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_spec.scheduler_t_max,
        eta_min=optimizer_spec.scheduler_eta_min,
    )
