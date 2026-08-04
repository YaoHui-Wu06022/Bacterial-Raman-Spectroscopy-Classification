"""训练状态保存、恢复与清理。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


@dataclass
class TrainingState:
    """训练循环持续维护的最佳指标和产物路径。"""

    best_score: float = -1e9
    best_epoch: int = -1
    patience_counter: int = 0
    ema_class_ce: torch.Tensor | None = None
    model_path: Path | None = None
    se_stats_path: Path | None = None
    checkpoint_path: Path | None = None


def build_checkpoint_path(model_path: Path | str) -> Path:
    """为模型权重路径生成同目录的续训 checkpoint 路径。"""
    target_path = Path(model_path)
    suffix = "_model.pt"
    if target_path.name.endswith(suffix):
        return target_path.with_name(target_path.name[: -len(suffix)] + "_checkpoint.pt")
    return target_path.with_name(target_path.name + ".checkpoint.pt")


def save_training_checkpoint(
    checkpoint_path: Path | str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_score: float,
    best_epoch: int,
    patience_counter: int,
    ema_class_ce: torch.Tensor,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """保存可恢复训练状态；传入 scaler 时一并保存其缩放状态。"""
    checkpoint = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "ema_class_ce": ema_class_ce.detach().cpu(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "grad_scaler_state": None if scaler is None else scaler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def restore_training_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    log_message: Callable[[str], None],
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float, int, int, torch.Tensor | None]:
    """恢复训练状态并返回下一轮 epoch 与早停相关状态。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("grad_scaler_state") is not None:
        scaler.load_state_dict(checkpoint["grad_scaler_state"])
    _restore_random_state(checkpoint)

    ema_class_ce = checkpoint.get("ema_class_ce")
    if ema_class_ce is not None:
        ema_class_ce = ema_class_ce.to(device=device, dtype=torch.float32)
    epoch = int(checkpoint.get("epoch", 0))
    best_epoch = int(checkpoint.get("best_epoch", -1))
    log_message(
        f"[Resume] loaded checkpoint: {checkpoint_path}, "
        f"last_epoch={epoch}, best_epoch={best_epoch}"
    )
    return (
        epoch + 1,
        float(checkpoint.get("best_score", -1e9)),
        best_epoch,
        int(checkpoint.get("patience_counter", 0)),
        ema_class_ce,
    )


def _restore_random_state(checkpoint: dict[str, object]) -> None:
    """恢复 checkpoint 中保存的 PyTorch、NumPy 和 Python 随机数状态。"""
    torch_state = checkpoint.get("torch_rng_state")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_state = checkpoint.get("cuda_rng_state")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    numpy_state = checkpoint.get("numpy_rng_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    python_state = checkpoint.get("python_rng_state")
    if python_state is not None:
        random.setstate(python_state)


def remove_training_checkpoint(
    checkpoint_path: Path | str | None,
    log_message: Callable[[str], None],
) -> None:
    """训练成功结束后删除已不再需要的续训 checkpoint。"""
    if checkpoint_path is None:
        return
    target_path = Path(checkpoint_path)
    if not target_path.exists():
        return
    target_path.unlink()
    log_message(f"[Checkpoint] removed finished checkpoint: {target_path}")
