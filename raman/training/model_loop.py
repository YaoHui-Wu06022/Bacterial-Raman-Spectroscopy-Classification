from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

from raman.eval.common import mask_logits_by_parent
from raman.model import RamanClassifier1D
from raman.training.checkpoint import (
    TrainingState,
    build_checkpoint_path,
    build_model_artifact_paths,
    remove_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from raman.training.losses import (
    AlignLoss,
    FocalLoss,
    SupConLoss,
    build_class_weights,
    get_linear_weight,
)
from raman.training.session import create_model_logger
from raman.training.validation import evaluate_validation_loader


def _build_loader_kwargs(config, device, train=True):
    """按当前配置构造 DataLoader 参数"""
    num_workers = int(
        config.train_loader_num_workers if train else config.val_loader_num_workers
    )
    num_workers = max(num_workers, 0)
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": bool(train),
        "num_workers": num_workers,
    }
    if device.type == "cuda" and bool(config.loader_pin_memory):
        kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.loader_persistent_workers)
        kwargs["prefetch_factor"] = int(config.loader_prefetch_factor)
    return kwargs

def _compute_severity_weights(prob, targets):
    """
    按当前层类别数自适应计算样本级严重程度权重

    设计原则：
    - 二分类时不再额外做 severity 重加权，避免和 Focal 重复放大
    - 三分类时只对“差一点分对”的样本做轻微降权
    - 四类及以上时再明显区分 rank-2 / rank-3 / 高置信度错判
    """
    num_classes = prob.size(1)
    severity_w = torch.ones(prob.size(0), dtype=prob.dtype, device=prob.device)
    # 二分类直接返回
    if num_classes <= 2:
        return severity_w

    topk = min(3, num_classes)
    topk_val, topk_idx = prob.topk(topk, dim=1)
    is_top1 = topk_idx[:, 0] == targets

    rank = torch.full_like(topk_idx[:, 0], fill_value=topk + 1)
    for k in range(topk):
        rank[topk_idx[:, k] == targets] = k + 1
    # 三分类
    if num_classes == 3:
        if topk >= 2:
            severity_w[rank == 2] = 0.95
        severity_w[rank >= 3] = 1.10

        high_conf_wrong = (~is_top1) & (topk_val[:, 0] > 0.88)
        severity_w[high_conf_wrong & (rank == 2)] = 1.05
        severity_w[high_conf_wrong & (rank >= 3)] = 1.25
        return severity_w

    if topk >= 2:
        severity_w[rank == 2] = 0.90
    if topk >= 3:
        severity_w[rank == 3] = 1.00
    severity_w[rank >= 4] = 1.10

    high_conf_wrong = (~is_top1) & (topk_val[:, 0] > 0.85)
    severity_w[high_conf_wrong & (rank == 2)] = 1.10
    severity_w[high_conf_wrong & (rank >= 3)] = 1.35
    return severity_w


def _loss_value_for_log(loss):
    """安全地把 loss 张量转换成日志用的浮点数。"""
    if not isinstance(loss, torch.Tensor):
        return float(loss)
    value = loss.detach()
    if value.numel() != 1:
        value = value.mean()
    return float(value.cpu())


def _model_parameters_are_finite(model):
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def _disabled_aux_loss(zero_loss):
    """构造被关闭的辅助损失函数"""
    def loss_fn(feat, _hier_labels):
        return zero_loss(feat)

    return loss_fn


@dataclass
class ModelTrainContext:
    """单模型训练所需的共享上下文，避免训练入口传递过长参数列表"""

    config: object
    log: object
    runtime_dirs: dict
    device: object
    train_dataset: object
    val_dataset: object
    full_dataset: object
    head_names: list
    use_align_loss: bool
    use_supcon_loss: bool
    align_loss_weight: float
    supcon_loss_weight: float
    decay_start_ratio: float
    zero_loss: object

def train_model(
    ctx,
    model_tag,
    level_name,
    level_idx,
    train_indices,
    val_indices,
    num_classes,
    parent_level_idx=None,
    parent_to_children=None,
    label_map_np=None,
    use_parent_mask=False,
):
    """训练一个全局层模型或一个 parent 内子模型"""
    config = ctx.config
    log = ctx.log
    runtime_dirs = ctx.runtime_dirs
    device = ctx.device
    train_dataset = ctx.train_dataset
    val_dataset = ctx.val_dataset
    full_dataset = ctx.full_dataset
    head_names = ctx.head_names
    USE_ALIGN_LOSS = ctx.use_align_loss
    USE_SUPCON_LOSS = ctx.use_supcon_loss
    ALIGN_LOSS_WEIGHT = ctx.align_loss_weight
    SUPCON_LOSS_WEIGHT = ctx.supcon_loss_weight
    decay_start_ratio = ctx.decay_start_ratio
    zero_loss = ctx.zero_loss

    model_log_path, model_log, model_log_file = create_model_logger(
        runtime_dirs["logs"],
        model_tag,
        log,
    )
    if len(train_indices) == 0:
        model_log(f"[Skip] {model_tag}: no train samples")
        model_log_file.close()
        return None

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices) if len(val_indices) > 0 else None

    train_loader = DataLoader(
        train_subset,
        **_build_loader_kwargs(config, device, train=True),
    )

    val_loader = None
    if val_subset is not None and len(val_subset) > 0:
        val_loader = DataLoader(
            val_subset,
            **_build_loader_kwargs(config, device, train=False),
        )
    if val_loader is None:
        raise ValueError(
            f"{model_tag} 没有可用的验证样本，当前实现要求每个参与训练的 leaf "
            "在切分后都能为验证集提供样本"
        )

    model = RamanClassifier1D(
        num_classes=num_classes,
        config=config
    ).to(device)

    # 类别权重（只针对当前训练层）
    labels_for_weights = full_dataset.level_labels[train_indices, level_idx]
    if label_map_np is not None:
        mapped = np.full_like(labels_for_weights, -1)
        valid = labels_for_weights >= 0
        mapped[valid] = label_map_np[labels_for_weights[valid]]
        labels_for_weights = mapped

    base_class_weights = torch.tensor(
        build_class_weights(labels_for_weights, num_classes),
        dtype=torch.float32,
        device=device,
    )

    # 主分类损失使用 Focal Loss，EMA 启用后会替换类别权重
    criterion = FocalLoss(
        gamma=config.gamma,
        weight=base_class_weights,  # EMA 动态权重启用后会替换为 ema_class_weights
        ignore_index=-1,
    )

    # EMA 只统计类别难度，权重延迟启用
    train_state = TrainingState(ema_class_ce=torch.ones(num_classes, device=device))
    ema_alpha = 0.9
    lambda_diff = 0.3
    ema_start_epoch = 10

    supcon_level = train_dataset._resolve_level_name(level_name)

    if USE_ALIGN_LOSS:
        def align_loss_fn(feat, hier_labels):
            level_labels = hier_labels.get(level_name)
            if level_labels is None:
                return zero_loss(feat)
            return AlignLoss(
                feat,
                level_labels,
            )
    else:
        align_loss_fn = _disabled_aux_loss(zero_loss)

    if USE_SUPCON_LOSS:
        supcon_criterion = SupConLoss(tau=config.supcon_tau).to(device)

        def supcon_loss_fn(feat, hier_labels):
            if supcon_level not in hier_labels:
                return zero_loss(feat)
            labels = hier_labels[supcon_level]
            valid = labels >= 0
            if valid.sum() <= 1:
                return zero_loss(feat)
            return supcon_criterion(feat[valid], labels[valid])
    else:
        supcon_loss_fn = _disabled_aux_loss(zero_loss)

    # 输入 stem 用较小学习率，分类头用较大学习率
    group_conv = []
    group_head = []
    group_backbone = []

    for name, p in model.named_parameters():
        if name.startswith("stem_branches"):
            group_conv.append(p)
        elif name.startswith("head"):
            group_head.append(p)
        else:
            group_backbone.append(p)

    optimizer = optim.AdamW([
        {"params": group_conv, "lr": config.learning_rate*0.6},
        {"params": group_backbone, "lr": config.learning_rate},
        {"params": group_head, "lr": config.learning_rate*1.1},
    ], weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_Tmax, eta_min=config.scheduler_eta_min
    )

    best_model_path, best_se_stats_path = build_model_artifact_paths(
        config.output_dir,
        level_name,
        model_tag,
    )
    checkpoint_path = build_checkpoint_path(best_model_path)
    train_state.model_path = best_model_path
    train_state.se_stats_path = best_se_stats_path
    train_state.checkpoint_path = checkpoint_path
    # parent 子模型需要把全局类别索引映射到局部输出索引
    label_map_tensor = None
    if label_map_np is not None:
        label_map_tensor = torch.tensor(label_map_np, dtype=torch.long, device=device)

    model_log(f"[{model_tag}] ==================== MODEL START ====================")
    model_log(f"[{model_tag}] log_path = {model_log_path}")
    model_log(f"[{model_tag}] checkpoint_path = {checkpoint_path}")
    try:
        start_epoch = 1
        if getattr(config, "resume_training", True) and os.path.exists(checkpoint_path):
            (
                start_epoch,
                train_state.best_score,
                train_state.best_epoch,
                train_state.patience_counter,
                restored_ema_class_ce,
            ) = restore_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                device,
                model_log,
            )
            if restored_ema_class_ce is not None:
                train_state.ema_class_ce = restored_ema_class_ce
            if not _model_parameters_are_finite(model):
                raise ValueError(
                    f"Restored checkpoint contains non-finite model parameters: {checkpoint_path}. "
                    "Delete this checkpoint or run with resume_training=False."
                )

        if start_epoch > config.epochs:
            model_log(
                f"[Resume] checkpoint 已完成到 epoch={start_epoch - 1}，"
                f"当前 epochs={config.epochs}，跳过训练"
            )
            return {
                "best_model_path": best_model_path,
                "best_se_stats_path": best_se_stats_path,
                "checkpoint_path": checkpoint_path,
                "model": None,
                "level_name": level_name,
                "model_log_path": model_log_path,
            }

        for epoch in range(start_epoch, config.epochs + 1):
            model.train()
            # 到达起始轮后，用 EMA 类别 CE 调整静态类别权重
            if config.use_ema and epoch >= ema_start_epoch:
                if not torch.isfinite(train_state.ema_class_ce).all().item():
                    model_log(
                        "[Warn] ema_class_ce contains non-finite values; reset to ones"
                    )
                    train_state.ema_class_ce = torch.ones_like(train_state.ema_class_ce)
                raw_diff = train_state.ema_class_ce / (train_state.ema_class_ce.mean() + 1e-12)
                diff_factor = 1.0 + lambda_diff * (raw_diff - 1.0)
                ema_class_weights = base_class_weights * diff_factor
                ema_class_weights = ema_class_weights / (ema_class_weights.mean() + 1e-12)
                criterion.weight = ema_class_weights

            running_loss, running_correct, running_total = 0, 0, 0
            running_align_loss = 0.0
            running_supcon_loss = 0.0
            skipped_batches = 0

            # 辅助损失先线性启用，再在后期共同衰减
            align_w = get_linear_weight(epoch,
                                       start=config.align_start,
                                       end=config.align_end,
                                       w_min=0.0,
                                       w_max=ALIGN_LOSS_WEIGHT
            )
            supcon_w = get_linear_weight(epoch,
                                        start=config.supcon_start,
                                        end=config.supcon_end,
                                        w_min=0.0,
                                        w_max=SUPCON_LOSS_WEIGHT,
            )
            decay_start = int(decay_start_ratio * config.epochs)
            decay_ratio = 1.0
            if epoch > decay_start:
                decay_ratio = 1.0 - (epoch - decay_start) / max(1, config.epochs - decay_start)
                decay_ratio = max(decay_ratio, 0.2)
            align_w = align_w * decay_ratio
            supcon_w = supcon_w * decay_ratio
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")

            for batch_idx, (x, y, _) in enumerate(loader_iter):

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(x).all().item():
                    skipped_batches += 1
                    model_log(
                        f"[Warn] skipped non-finite input at epoch={epoch}, batch={batch_idx}"
                    )
                    continue

                logits, feat = model(x, return_feat=True)

                if y.ndim == 2:
                    hier_labels = {name: y[:, idx] for idx, name in enumerate(head_names)}
                    y_level = y[:, level_idx]
                    parent_labels = y[:, parent_level_idx] if parent_level_idx is not None else None
                else:
                    hier_labels = {level_name: y}
                    y_level = y
                    parent_labels = None

                if label_map_tensor is not None:
                    invalid = y_level < 0
                    y_level = label_map_tensor[y_level.clamp_min(0)]
                    y_level[invalid] = -1

                # 父类遮罩只允许样本在真实父类的子类集合内竞争
                if use_parent_mask and parent_labels is not None:
                    logits_masked, valid_parent = mask_logits_by_parent(
                        logits, parent_labels, parent_to_children
                    )
                else:
                    logits_masked = logits
                    valid_parent = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

                valid = (y_level >= 0) & valid_parent

                if valid.any():
                    logits_valid = logits_masked[valid]
                    y_valid = y_level[valid]
                    loss_cls_each = criterion(logits_valid, y_valid)

                    if config.use_severity_weight:
                        with torch.no_grad():
                            prob = torch.softmax(logits_valid, dim=1)
                            severity_w = _compute_severity_weights(prob, y_valid)

                        loss_cls = (loss_cls_each * severity_w).mean()
                    else:
                        loss_cls = loss_cls_each.mean()
                else:
                    loss_cls = torch.tensor(0.0, device=device)

                loss_align = align_loss_fn(feat, hier_labels) if align_w > 0 else zero_loss(feat)
                loss_supcon = supcon_loss_fn(feat, hier_labels) if supcon_w > 0 else zero_loss(feat)

                loss_total = loss_cls
                if align_w > 0:
                    loss_total = loss_total + align_w * loss_align
                if supcon_w > 0:
                    loss_total = loss_total + supcon_w * loss_supcon

                if not torch.isfinite(loss_total).all().item():
                    skipped_batches += 1
                    model_log(
                        f"[Warn] skipped non-finite loss at epoch={epoch}, batch={batch_idx}: "
                        f"cls={_loss_value_for_log(loss_cls):.6g}, "
                        f"align={_loss_value_for_log(loss_align):.6g}, "
                        f"supcon={_loss_value_for_log(loss_supcon):.6g}, "
                        f"align_w={align_w:.6g}, supcon_w={supcon_w:.6g}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss_total.backward()

                grad_clip_norm = float(getattr(config, "grad_clip_norm", 0.0) or 0.0)
                if grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )
                    if not torch.isfinite(grad_norm).all().item():
                        skipped_batches += 1
                        model_log(
                            f"[Warn] skipped non-finite gradients at epoch={epoch}, "
                            f"batch={batch_idx}, grad_norm={_loss_value_for_log(grad_norm):.6g}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                optimizer.step()

                running_align_loss += loss_align.item()
                running_supcon_loss += loss_supcon.item()

                running_loss += loss_cls.item()
                if valid.any():
                    running_correct += (logits_valid.argmax(1) == y_valid).sum().item()
                    running_total += valid.sum().item()

                postfix = {
                    "cls": f"{running_loss / max(batch_idx + 1 - skipped_batches, 1):.4f}",
                    "acc": f"{100 * running_correct / max(running_total, 1):.2f}%"
                }
                loader_iter.set_postfix(postfix)

                # 用未加权 CE 更新每个类别的 EMA 难度
                if config.use_ema and valid.any():
                    with torch.no_grad():
                        ce_each = F.cross_entropy(logits_valid, y_valid, reduction="none")
                        finite_ce = torch.isfinite(ce_each)
                        for c in range(num_classes):
                            class_mask_c = (y_valid == c) & finite_ce
                            if class_mask_c.any():
                                mean_ce_c = ce_each[class_mask_c].mean()
                                if torch.isfinite(mean_ce_c).all().item():
                                    train_state.ema_class_ce[c] = (
                                        ema_alpha * train_state.ema_class_ce[c]
                                        + (1.0 - ema_alpha) * mean_ce_c
                                    )

            effective_batches = max(len(train_loader) - skipped_batches, 1)
            train_loss = running_loss / effective_batches
            train_align_loss = align_w * running_align_loss / effective_batches
            train_supcon_loss = supcon_w * running_supcon_loss / effective_batches
            train_acc = running_correct / max(running_total, 1)
            val_loss, val_acc, val_metrics, se_stats = evaluate_validation_loader(
                model,
                val_loader,
                device,
                head_index=level_idx,
                head_name=level_name,
                label_map_tensor=None if use_parent_mask else label_map_tensor,
                parent_index=parent_level_idx if use_parent_mask else None,
                parent_to_children=parent_to_children if use_parent_mask else None,
            )
            macro_f1 = val_metrics["macro_f1"]
            macro_recall = val_metrics["macro_recall"]
            scheduler.step()
            if epoch % 10 == 0:
                model_log(f"  base_class_weights = {base_class_weights.detach().cpu().numpy()}")
                if config.use_ema:
                    model_log(f"  ema_class_ce       = {train_state.ema_class_ce.detach().cpu().numpy()}")
                    model_log(f"  ema_class_weights  = {criterion.weight.detach().cpu().numpy()}")
            model_log(
                f"[Epoch {epoch}] TrainLoss(cls)={train_loss:.4f}, "
                f"AlignLossW={train_align_loss:.4f}, "
                f"SupConLossW={train_supcon_loss:.4f}, "
                f"ValLoss={val_loss:.4f}\n"
                f"TrainAcc={train_acc * 100:.2f}%, "
                f"ValAcc={val_acc * 100:.2f}%, "
                f"ValMacroF1={macro_f1 * 100:.2f}%, "
                f"ValMacroRecall={macro_recall * 100:.2f}%, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}, "
            )
            # Early Stop 只看宏 F1 与 accuracy 的加权分数
            score = (
                config.early_stop_w_f1 * macro_f1
                + config.early_stop_w_acc * val_acc
            )
            model_log(
                f"EarlyStop score = "
                f"{score:.4f} (w_f1={config.early_stop_w_f1}, "
                f"w_acc={config.early_stop_w_acc})"
            )
            # 最优模型和对应 SE 统计一起保存
            if score >= train_state.best_score:
                train_state.best_score = score
                train_state.best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                if se_stats:
                    torch.save(se_stats, best_se_stats_path)
                model_log("  --> Best model updated! (EarlyStop score improved)")
                train_state.patience_counter = 0
            else:
                train_state.patience_counter += 1

            checkpoint_interval = max(1, int(getattr(config, "checkpoint_interval", 10)))
            should_save_checkpoint = epoch % checkpoint_interval == 0
            if should_save_checkpoint:
                save_training_checkpoint(
                    checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    train_state.best_score,
                    train_state.best_epoch,
                    train_state.patience_counter,
                    train_state.ema_class_ce,
                )
            model_log(f"[{model_tag}] ------------------------------------------------")

            if train_state.patience_counter >= config.patience:
                model_log("EarlyStopping Triggered by weighted score!")
                break

        remove_training_checkpoint(checkpoint_path, model_log)
        model_log(f"=== Best model epoch: {train_state.best_epoch} ===")
    finally:
        model_log_file.close()


    # 显式释放当前模型，方便连续训练多个层级或 parent
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_model_path": train_state.model_path,
        "best_se_stats_path": train_state.se_stats_path,
        "checkpoint_path": train_state.checkpoint_path,
        "model": None,
        "level_name": level_name,
        "model_log_path": model_log_path,
    }

