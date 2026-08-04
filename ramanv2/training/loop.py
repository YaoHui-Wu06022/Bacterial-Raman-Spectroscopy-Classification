"""训练循环使用的 DataLoader 与优化器构建。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ramanv2.training.spec import ExecutionSpec, TrainingSpec
from ramanv2.training.checkpoint import (
    TrainingState,
    remove_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from ramanv2.training.diagnostics import write_nonfinite_diagnostic
from ramanv2.training.losses import (
    FocalLoss,
    SupConLoss,
    build_class_weights,
    compute_align_loss,
    compute_linear_weight,
)
from ramanv2.training.optimizer import build_optimizer, build_scheduler
from ramanv2.training.split import TrainTask
from ramanv2.training.validation import evaluate_validation_loader


@dataclass(frozen=True)
class TrainArtifacts:
    """由 workflow 创建并交给训练循环消费的单模型产物。"""

    model_path: Path
    se_stats_path: Path
    checkpoint_path: Path
    diagnostic_path: Path


@dataclass(frozen=True)
class TrainResult:
    """单个模型训练完成后的可追溯结果。"""

    model_path: Path
    se_stats_path: Path
    checkpoint_path: Path
    best_epoch: int
    best_score: float


def run_train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_task: TrainTask,
    train_spec: TrainingSpec,
    runtime_spec: ExecutionSpec,
    train_artifacts: TrainArtifacts,
    log_message: Callable[[str], None],
    apply_training_mode: Callable[[torch.nn.Module], None] | None = None,
) -> TrainResult:
    """执行单个层级或父类子模型的完整 epoch 训练与验证。"""
    device = _resolve_device(runtime_spec)
    label_map_tensor = _build_label_map_tensor(train_task, device)
    model = model.to(device)
    optimizer = build_optimizer(model, train_spec.optimizer)
    scheduler = build_scheduler(optimizer, train_spec.optimizer)
    criterion, train_state = _build_training_loss(train_task, train_spec, device)
    amp_enable = runtime_spec.amp_enable and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enable else None
    start_epoch = _restore_train_state(
        model,
        optimizer,
        scheduler,
        scaler,
        runtime_spec,
        device,
        train_artifacts,
        train_state,
        log_message,
    )
    if start_epoch > train_spec.epochs:
        return TrainResult(
            train_artifacts.model_path,
            train_artifacts.se_stats_path,
            train_artifacts.checkpoint_path,
            train_state.best_epoch,
            train_state.best_score,
        )

    diagnostic_written_enable = False
    try:
        for epoch in range(start_epoch, train_spec.epochs + 1):
            _apply_training_mode(model, apply_training_mode)
            _update_ema_class_weights(criterion, train_state, train_spec, epoch)
            align_weight, supcon_weight = _compute_auxiliary_weights(train_spec, epoch)
            epoch_result, diagnostic_written_enable = _run_train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                train_state,
                train_task,
                train_spec,
                runtime_spec,
                device,
                label_map_tensor,
                epoch,
                align_weight,
                supcon_weight,
                train_artifacts.diagnostic_path,
                diagnostic_written_enable,
            )
            _raise_if_model_parameters_are_nonfinite(model, epoch)
            val_loss, val_accuracy, val_metrics, se_stats = evaluate_validation_loader(
                model,
                val_loader,
                device,
                train_task.level_index,
                label_map_tensor=label_map_tensor,
                amp_enable=amp_enable,
            )
            scheduler.step()
            score = (
                train_spec.loss.early_stop_f1_weight
                * val_metrics["macro_f1"]
                + train_spec.loss.early_stop_accuracy_weight
                * val_accuracy
            )
            _log_epoch_result(
                log_message,
                epoch,
                epoch_result,
                val_loss,
                val_accuracy,
                val_metrics,
                optimizer,
                train_spec,
                score,
            )
            _update_best_model(
                model,
                se_stats,
                train_state,
                train_artifacts,
                epoch,
                score,
                log_message,
            )
            if epoch % runtime_spec.checkpoint_interval == 0:
                save_training_checkpoint(
                    train_artifacts.checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    train_state.best_score,
                    train_state.best_epoch,
                    train_state.patience_counter,
                    train_state.ema_class_ce,
                    scaler,
                )
            log_message(f"[{train_task.model_tag}] ------------------------------------------------")
            if train_state.patience_counter >= train_spec.patience:
                log_message("EarlyStopping Triggered by weighted score!")
                break
    except Exception:
        raise
    else:
        remove_training_checkpoint(train_artifacts.checkpoint_path, log_message)
        log_message(f"=== Best model epoch: {train_state.best_epoch} ===")
    return TrainResult(
        train_artifacts.model_path,
        train_artifacts.se_stats_path,
        train_artifacts.checkpoint_path,
        train_state.best_epoch,
        train_state.best_score,
    )


def _build_training_loss(
    train_task: TrainTask,
    train_spec: TrainingSpec,
    device: torch.device,
) -> tuple[FocalLoss, TrainingState]:
    """根据当前层标签建立 Focal 类别权重与 EMA 状态。"""
    weights = torch.tensor(
        build_class_weights(train_task.weight_labels, train_task.num_classes),
        dtype=torch.float32,
        device=device,
    )
    criterion = FocalLoss(train_spec.loss.focal_gamma, weights, -1)
    train_state = TrainingState(
        ema_class_ce=torch.ones(train_task.num_classes, device=device)
    )
    return criterion, train_state


def _restore_train_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    runtime_spec: ExecutionSpec,
    device: torch.device,
    train_artifacts: TrainArtifacts,
    train_state: TrainingState,
    log_message: Callable[[str], None],
) -> int:
    """根据 checkpoint 恢复训练状态，并返回下一轮 epoch。"""
    if not runtime_spec.resume_enable or not train_artifacts.checkpoint_path.exists():
        return 1
    (
        start_epoch,
        train_state.best_score,
        train_state.best_epoch,
        train_state.patience_counter,
        ema_class_ce,
    ) = restore_training_checkpoint(
        train_artifacts.checkpoint_path,
        model,
        optimizer,
        scheduler,
        device,
        log_message,
        scaler,
    )
    if ema_class_ce is not None:
        train_state.ema_class_ce = ema_class_ce
    _raise_if_model_parameters_are_nonfinite(model, start_epoch - 1)
    return start_epoch


def _apply_training_mode(
    model: torch.nn.Module,
    apply_training_mode: Callable[[torch.nn.Module], None] | None,
) -> None:
    """切换模型训练态，并应用可选的冻结模块策略。"""
    model.train()
    if apply_training_mode is not None:
        apply_training_mode(model)


def _update_ema_class_weights(
    criterion: FocalLoss,
    train_state: TrainingState,
    train_spec: TrainingSpec,
    epoch: int,
) -> None:
    """在指定 epoch 后按 EMA 分类难度更新 Focal 类别权重。"""
    spec = train_spec.loss
    if not spec.ema_enable or epoch < spec.ema_start_epoch:
        return
    if not torch.isfinite(train_state.ema_class_ce).all().item():
        train_state.ema_class_ce = torch.ones_like(train_state.ema_class_ce)
    difficulty = train_state.ema_class_ce / (train_state.ema_class_ce.mean() + 1e-12)
    factor = 1.0 + spec.ema_difficulty_weight * (difficulty - 1.0)
    weights = criterion.weight * factor
    criterion.weight = weights / (weights.mean() + 1e-12)


def _compute_auxiliary_weights(train_spec: TrainingSpec, epoch: int) -> tuple[float, float]:
    """计算 Align 与 SupCon 启用后在当前 epoch 实际参与的权重。"""
    spec = train_spec.loss
    align_weight = (
        compute_linear_weight(
            epoch,
            spec.align_start_epoch,
            spec.align_end_epoch,
            0.0,
            spec.align_weight,
        )
        if spec.align_enable
        else 0.0
    )
    supcon_weight = (
        compute_linear_weight(
            epoch,
            spec.supcon_start_epoch,
            spec.supcon_end_epoch,
            0.0,
            spec.supcon_weight,
        )
        if spec.supcon_enable
        else 0.0
    )
    epochs = train_spec.epochs
    decay_start = int(spec.auxiliary_decay_start_ratio * epochs)
    if epoch <= decay_start:
        return align_weight, supcon_weight
    decay = max(1.0 - (epoch - decay_start) / max(1, epochs - decay_start), 0.2)
    return align_weight * decay, supcon_weight * decay


def _run_train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    criterion: FocalLoss,
    train_state: TrainingState,
    train_task: TrainTask,
    train_spec: TrainingSpec,
    runtime_spec: ExecutionSpec,
    device: torch.device,
    label_map_tensor: torch.Tensor | None,
    epoch: int,
    align_weight: float,
    supcon_weight: float,
    diagnostic_path: Path,
    diagnostic_written_enable: bool,
) -> tuple[dict[str, float], bool]:
    """训练一个 epoch，并在首个数值异常处写入有限摘要。"""
    running_loss = 0.0
    running_align_loss = 0.0
    running_supcon_loss = 0.0
    correct_count = 0
    sample_count = 0
    skipped_batches = 0
    amp_enable = runtime_spec.amp_enable and device.type == "cuda"
    supcon_criterion = SupConLoss(train_spec.loss.supcon_tau).to(device)
    loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{train_spec.epochs}")
    for batch_index, (inputs, labels, sample_paths) in enumerate(loader_iter):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(inputs).all().item():
            skipped_batches += 1
            diagnostic_written_enable = write_nonfinite_diagnostic(
                diagnostic_written_enable,
                diagnostic_path,
                "input",
                epoch,
                batch_index,
                inputs,
                labels,
                sample_paths,
            )
            continue
        with torch.autocast(device_type=device.type, enabled=amp_enable):
            logits, embeddings = model(inputs, return_embedding_enable=True)
        if not torch.isfinite(logits).all().item() or not torch.isfinite(embeddings).all().item():
            skipped_batches += 1
            diagnostic_written_enable = write_nonfinite_diagnostic(
                diagnostic_written_enable,
                diagnostic_path,
                "forward",
                epoch,
                batch_index,
                inputs,
                labels,
                sample_paths,
                tensors={"logits": logits, "embedding": embeddings},
            )
            continue
        loss_values = _compute_batch_losses(
            logits.float(),
            embeddings.float(),
            labels,
            criterion,
            supcon_criterion,
            train_task,
            label_map_tensor,
            align_weight,
            supcon_weight,
        )
        if not torch.isfinite(loss_values["total"]).all().item():
            skipped_batches += 1
            diagnostic_written_enable = write_nonfinite_diagnostic(
                diagnostic_written_enable,
                diagnostic_path,
                "loss",
                epoch,
                batch_index,
                inputs,
                labels,
                sample_paths,
                tensors={"logits": logits, "embedding": embeddings},
                losses=loss_values,
            )
            continue
        if not _apply_gradient_step(
            loss_values["total"],
            model,
            optimizer,
            scaler,
            train_spec.optimizer.grad_clip_norm,
        ):
            skipped_batches += 1
            diagnostic_written_enable = write_nonfinite_diagnostic(
                diagnostic_written_enable,
                diagnostic_path,
                "gradient",
                epoch,
                batch_index,
                inputs,
                labels,
                sample_paths,
                tensors={"logits": logits, "embedding": embeddings},
                losses=loss_values,
            )
            continue
        _update_ema_class_loss(train_state, loss_values, train_task, train_spec)
        running_loss += float(loss_values["classification"].detach())
        running_align_loss += float(loss_values["align"].detach())
        running_supcon_loss += float(loss_values["supcon"].detach())
        valid_mask = loss_values["valid_mask"]
        if valid_mask.any():
            valid_logits = loss_values["logits"]
            valid_labels = loss_values["targets"]
            correct_count += int((valid_logits.argmax(1) == valid_labels).sum())
            sample_count += int(valid_mask.sum())
        loader_iter.set_postfix(
            cls=f"{running_loss / max(batch_index + 1 - skipped_batches, 1):.4f}",
            acc=f"{100 * correct_count / max(sample_count, 1):.2f}%",
        )

    effective_batches = max(len(train_loader) - skipped_batches, 1)
    return {
        "classification_loss": running_loss / effective_batches,
        "align_loss": align_weight * running_align_loss / effective_batches,
        "supcon_loss": supcon_weight * running_supcon_loss / effective_batches,
        "accuracy": correct_count / max(sample_count, 1),
        "skipped_batches": float(skipped_batches),
    }, diagnostic_written_enable


def _compute_batch_losses(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    criterion: FocalLoss,
    supcon_criterion: SupConLoss,
    train_task: TrainTask,
    label_map_tensor: torch.Tensor | None,
    align_weight: float,
    supcon_weight: float,
) -> dict[str, torch.Tensor]:
    """计算主分类损失及启用时参与的 Align、SupCon 损失。"""
    level_targets = _select_training_labels(labels, train_task)
    local_targets = _map_local_targets(level_targets, label_map_tensor)
    valid_mask = local_targets >= 0
    if valid_mask.any():
        valid_logits = logits[valid_mask]
        valid_targets = local_targets[valid_mask]
        classification_loss = criterion(valid_logits, valid_targets).mean()
    else:
        valid_logits = logits[:0]
        valid_targets = local_targets[:0]
        classification_loss = _zero_loss(embeddings)
    align_loss = (
        compute_align_loss(embeddings, level_targets)
        if align_weight > 0
        else _zero_loss(embeddings)
    )
    supcon_labels = level_targets
    supcon_valid_mask = supcon_labels >= 0
    supcon_loss = (
        supcon_criterion(embeddings[supcon_valid_mask], supcon_labels[supcon_valid_mask])
        if supcon_weight > 0 and supcon_valid_mask.sum() > 1
        else _zero_loss(embeddings)
    )
    total_loss = classification_loss + align_weight * align_loss + supcon_weight * supcon_loss
    return {
        "classification": classification_loss,
        "align": align_loss,
        "supcon": supcon_loss,
        "total": total_loss,
        "valid_mask": valid_mask,
        "logits": valid_logits,
        "targets": valid_targets,
    }


def _select_training_labels(
    labels: torch.Tensor,
    train_task: TrainTask,
) -> torch.Tensor:
    """从单层或多层标签中建立当前 batch 的层级标签视图。"""
    if labels.ndim != 2:
        return labels
    return labels[:, train_task.level_index]


def _map_local_targets(
    targets: torch.Tensor,
    label_map_tensor: torch.Tensor | None,
) -> torch.Tensor:
    """将全局类别索引映射为父类子模型使用的局部标签索引。"""
    if label_map_tensor is None:
        return targets
    invalid_mask = targets < 0
    local_targets = label_map_tensor[targets.clamp_min(0)]
    local_targets[invalid_mask] = -1
    return local_targets


def _apply_gradient_step(
    total_loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float,
) -> bool:
    """反向传播、可选梯度裁剪和优化器更新；异常梯度返回 ``False``。"""
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    if grad_clip_norm > 0:
        try:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError:
            if scaler is not None:
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return False
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return True


def _update_ema_class_loss(
    train_state: TrainingState,
    loss_values: dict[str, torch.Tensor],
    train_task: TrainTask,
    train_spec: TrainingSpec,
) -> None:
    """用未加权交叉熵更新各类别的 EMA 难度。"""
    if not train_spec.loss.ema_enable or not loss_values["valid_mask"].any():
        return
    with torch.no_grad():
        losses = torch.nn.functional.cross_entropy(
            loss_values["logits"],
            loss_values["targets"],
            reduction="none",
        )
        finite_mask = torch.isfinite(losses)
        for class_index in range(train_task.num_classes):
            class_mask = (loss_values["targets"] == class_index) & finite_mask
            if not class_mask.any():
                continue
            class_loss = losses[class_mask].mean()
            train_state.ema_class_ce[class_index] = (
                train_spec.loss.ema_alpha
                * train_state.ema_class_ce[class_index]
                + (1.0 - train_spec.loss.ema_alpha) * class_loss
            )


def _update_best_model(
    model: torch.nn.Module,
    se_stats: dict[str, dict[str, int | torch.Tensor]],
    train_state: TrainingState,
    train_artifacts: TrainArtifacts,
    epoch: int,
    score: float,
    log_message: Callable[[str], None],
) -> None:
    """仅在验证评分改善时更新模型权重和对应的 SE 统计。"""
    if score > train_state.best_score:
        train_state.best_score = score
        train_state.best_epoch = epoch
        train_state.patience_counter = 0
        torch.save(model.state_dict(), train_artifacts.model_path)
        if se_stats:
            torch.save(se_stats, train_artifacts.se_stats_path)
        log_message("  --> Best model updated! (EarlyStop score improved)")
    else:
        train_state.patience_counter += 1


def _log_epoch_result(
    log_message: Callable[[str], None],
    epoch: int,
    epoch_result: dict[str, float],
    val_loss: float,
    val_accuracy: float,
    val_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
    train_spec: TrainingSpec,
    score: float,
) -> None:
    """写入当前 epoch 的损失、指标、学习率和早停评分摘要。"""
    log_message(
        f"[Epoch {epoch}] "
        f"TrainLoss(cls)={epoch_result['classification_loss']:.4f}, "
        f"AlignLossW={epoch_result['align_loss']:.4f}, "
        f"SupConLossW={epoch_result['supcon_loss']:.4f}, "
        f"ValLoss={val_loss:.4f}\n"
        f"TrainAcc={epoch_result['accuracy'] * 100:.2f}%, "
        f"ValAcc={val_accuracy * 100:.2f}%, "
        f"ValMacroF1={val_metrics['macro_f1'] * 100:.2f}%, "
        f"ValMacroRecall={val_metrics['macro_recall'] * 100:.2f}%, "
        f"LR={optimizer.param_groups[0]['lr']:.2e}, "
    )
    log_message(
        f"EarlyStop score = {score:.4f} "
        f"(w_f1={train_spec.loss.early_stop_f1_weight}, "
        f"w_acc={train_spec.loss.early_stop_accuracy_weight}), "
        f"Skipped={int(epoch_result['skipped_batches'])}"
    )


def _raise_if_model_parameters_are_nonfinite(model: torch.nn.Module, epoch: int) -> None:
    """在恢复后和每个 epoch 后检查模型参数是否均为有限值。"""
    if all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        return
    raise FloatingPointError(f"model parameters are non-finite after epoch={epoch}")


def _zero_loss(values: torch.Tensor) -> torch.Tensor:
    """创建与当前计算图设备一致的零损失张量。"""
    return torch.zeros((), device=values.device, dtype=values.dtype)


def _resolve_device(runtime_spec: ExecutionSpec) -> torch.device:
    """按运行规格选择当前训练设备。"""
    use_cuda_enable = runtime_spec.use_gpu_enable and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda_enable else "cpu")


def _build_label_map_tensor(
    train_task: TrainTask,
    device: torch.device,
) -> torch.Tensor | None:
    """将父类子模型的局部标签映射移动到训练设备。"""
    if train_task.label_map is None:
        return None
    return torch.tensor(train_task.label_map, dtype=torch.long, device=device)
