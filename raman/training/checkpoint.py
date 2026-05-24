import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from raman.eval.experiment import resolve_model_sidecar_path


@dataclass
class TrainingState:
    best_score: float = -1e9
    best_epoch: int = -1
    patience_counter: int = 0
    ema_class_ce: torch.Tensor | None = None
    model_path: str | None = None
    se_stats_path: str | None = None
    checkpoint_path: str | None = None


def build_model_artifact_paths(output_dir, level_name, model_tag):
    """构造按层子目录组织的模型与 sidecar 路径"""
    if os.path.basename(os.path.normpath(output_dir)).startswith("run_"):
        model_dir = output_dir
    else:
        model_dir = os.path.join(output_dir, level_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_tag}_model.pt")
    se_stats_path = resolve_model_sidecar_path(model_path)
    return model_path, se_stats_path

def build_checkpoint_path(model_path):
    """续训 checkpoint 与模型权重放在同一层目录"""
    suffix = "_model.pt"
    if model_path.endswith(suffix):
        return model_path[: -len(suffix)] + "_checkpoint.pt"
    return model_path + ".checkpoint.pt"

def save_training_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_score,
    best_epoch,
    patience_counter,
    ema_class_ce,
):
    """保存可恢复训练状态，不替代最佳模型权重"""
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
    }
    torch.save(checkpoint, checkpoint_path)

def restore_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    device,
    model_log,
):
    """恢复续训状态，并返回下一轮 epoch 与 early stop 状态"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    epoch = int(checkpoint.get("epoch", 0))
    best_score = float(checkpoint.get("best_score", -1e9))
    best_epoch = int(checkpoint.get("best_epoch", -1))
    patience_counter = int(checkpoint.get("patience_counter", 0))
    ema_class_ce = checkpoint.get("ema_class_ce")
    if ema_class_ce is not None:
        ema_class_ce = ema_class_ce.to(device=device, dtype=torch.float32)

    model_log(
        f"[Resume] loaded checkpoint: {checkpoint_path}, "
        f"last_epoch={epoch}, best_epoch={best_epoch}"
    )
    return epoch + 1, best_score, best_epoch, patience_counter, ema_class_ce

def remove_training_checkpoint(checkpoint_path, model_log):
    """训练正常结束后删除续训 checkpoint"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return
    os.remove(checkpoint_path)
    model_log(f"[Checkpoint] removed finished checkpoint: {checkpoint_path}")

def build_relpath(output_dir, path):
    """将模型绝对路径转成相对实验目录的路径"""
    return os.path.relpath(path, output_dir)

