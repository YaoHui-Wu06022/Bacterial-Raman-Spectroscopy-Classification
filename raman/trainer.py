"""Raman 层级分类训练脚本。"""
from copy import deepcopy
import json
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F

from raman.config import config as default_config
from raman.data import RamanDataset
from raman.model import ResNeXt1D_Transformer
from raman.training.eval import (
    evaluate_file_level,
    evaluate_file_level_local,
    mask_logits_by_parent,
)
from raman.training.losses import (
    FocalLoss,
    SupConLoss,
    build_class_weights,
    get_linear_weight,
    hierarchical_center_loss,
)
from raman.training.session import (
    create_model_logger,
    prepare_training_runtime,
    save_hierarchy_meta,
    set_seed,
)
from raman.training.split import (
    apply_train_filter,
    build_label_map_np,
    log_split_summary,
    resolve_level_order,
    resolve_levels_to_train,
    resolve_train_scope,
    resolve_train_split,
)

@dataclass
class TrainOverrides:
    """训练入口覆盖项，供根目录脚本和 Colab 统一复用。"""

    current_train_level: str | None = None
    train_only_parent_name: str | None = None
    train_only_parent: int | None = None
    override_decay_start_ratio: float | None = None
    override_align_loss_weight: float | None = None
    override_supcon_tau: float | None = None
    override_supcon_loss_weight: float | None = None
    override_timestamp: str | None = None
    override_output_dir: str | None = None


def apply_train_overrides(config, overrides=None):
    """把入口层的手动覆盖项写回配置对象。"""
    overrides = overrides or TrainOverrides()

    if overrides.current_train_level is not None:
        config.current_train_level = overrides.current_train_level
    if overrides.train_only_parent_name is not None:
        config.train_only_parent_name = overrides.train_only_parent_name
    if overrides.train_only_parent is not None:
        config.train_only_parent = overrides.train_only_parent

    if overrides.override_align_loss_weight is not None:
        config.align_loss_weight = float(overrides.override_align_loss_weight)
    if overrides.override_supcon_tau is not None:
        config.supcon_tau = float(overrides.override_supcon_tau)
    if overrides.override_supcon_loss_weight is not None:
        config.supcon_loss_weight = float(overrides.override_supcon_loss_weight)
    if overrides.override_decay_start_ratio is not None:
        config.decay_start_ratio = float(overrides.override_decay_start_ratio)

    if overrides.override_timestamp is not None:
        config.timestamp = str(overrides.override_timestamp)
    if overrides.override_output_dir is not None:
        config.output_dir = str(overrides.override_output_dir)
    return config


def _load_existing_hierarchy_meta(exp_dir):
    """读取已有实验目录中的层级训练元数据。"""
    meta_path = os.path.join(exp_dir, "hierarchy_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _resolve_saved_model_path(exp_dir, model_path):
    """把层级元数据中的模型路径解析成绝对路径。"""
    if not model_path:
        return None
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(exp_dir, model_path)


def _is_parent_entry_covered(exp_dir, entry):
    """判断某个 parent 条目是否足以支持级联。"""
    child_ids = list(entry.get("child_ids", []))
    model_path = entry.get("model_path")
    if model_path:
        full_path = _resolve_saved_model_path(exp_dir, model_path)
        return full_path is not None and os.path.exists(full_path)
    return len(child_ids) <= 1


def _build_loader_kwargs(config, device, train=True):
    """按当前配置构造 DataLoader 参数。"""
    num_workers = int(
        config.train_loader_num_workers if train else config.eval_loader_num_workers
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


def _has_level_coverage(exp_dir, full_dataset, level_name):
    """判断实验目录里是否已有某一上级层的完整训练结果。"""
    meta = _load_existing_hierarchy_meta(exp_dir)
    if meta is not None:
        level_models = meta.get("level_models", {})
        model_name = level_models.get(level_name)
        model_path = _resolve_saved_model_path(exp_dir, model_name)
        if model_path is not None and os.path.exists(model_path):
            return True

        parent_entries = meta.get("parent_models", {}).get(level_name, {})
        expected_mapping = full_dataset.parent_to_children.get(level_name, {})
        if expected_mapping:
            all_covered = True
            for parent_idx in expected_mapping.keys():
                entry = parent_entries.get(str(parent_idx))
                if entry is None:
                    entry = parent_entries.get(int(parent_idx))
                if entry is None or not _is_parent_entry_covered(exp_dir, entry):
                    all_covered = False
                    break
            if all_covered:
                return True
        elif parent_entries:
            if any(_is_parent_entry_covered(exp_dir, entry) for entry in parent_entries.values()):
                return True

    default_level_model = os.path.join(exp_dir, f"{level_name}_model.pt")
    if os.path.exists(default_level_model):
        return True

    prefix = f"{level_name}_parent_"
    suffix = "_model.pt"
    for name in os.listdir(exp_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            return True
    return False


def _log_missing_upper_level_hint(full_dataset, current_train_level, train_per_parent, exp_dir, log):
    """缺少上一级结果时给出非阻断提示。"""
    if not train_per_parent:
        return

    upper_level_name = full_dataset.get_parent_level(current_train_level)
    if upper_level_name is None:
        return

    if _has_level_coverage(exp_dir, full_dataset, upper_level_name):
        return

    log(
        f"[Hint] train_per_parent=True，但当前实验目录缺少上一级 "
        f"{upper_level_name} 的完整模型记录。"
    )
    log(
        f"[Hint] 若后续需要在同一 EXP_DIR 中继续向下训练、级联预测或测试评估，"
        f"建议先训练 current_train_level={upper_level_name}。"
    )


def _compute_severity_weights(prob, targets):
    """
    按当前层类别数自适应计算样本级严重程度权重。

    设计原则：
    - 二分类时不再额外做 severity 重加权，避免和 Focal 重复放大
    - 三分类时只对“差一点分对”的样本做轻微降权
    - 四类及以上时再明显区分 rank-2 / rank-3 / 高置信度错判
    """
    num_classes = prob.size(1)
    severity_w = torch.ones(prob.size(0), dtype=prob.dtype, device=prob.device)
    if num_classes <= 2:
        return severity_w

    topk = min(3, num_classes)
    topk_val, topk_idx = prob.topk(topk, dim=1)
    is_top1 = topk_idx[:, 0] == targets

    rank = torch.full_like(topk_idx[:, 0], fill_value=topk + 1)
    for k in range(topk):
        rank[topk_idx[:, k] == targets] = k + 1

    if num_classes == 3:
        if topk >= 2:
            severity_w[rank == 2] = 0.90

        high_conf_wrong = (~is_top1) & (topk_val[:, 0] > 0.85)
        severity_w[high_conf_wrong & (rank == 2)] = 1.10
        severity_w[high_conf_wrong & (rank >= 3)] = 1.45
        return severity_w

    if topk >= 2:
        severity_w[rank == 2] = 0.85
    if topk >= 3:
        severity_w[rank == 3] = 0.95

    high_conf_wrong = (~is_top1) & (topk_val[:, 0] > 0.80)
    severity_w[high_conf_wrong & (rank == 2)] = 1.20
    severity_w[high_conf_wrong & (rank >= 3)] = 1.80
    return severity_w



def run_training(config_obj=None, overrides=None):
    """执行一次完整训练流程。"""
    config = deepcopy(config_obj or default_config)
    config = apply_train_overrides(config, overrides)
    current_train_level = getattr(config, "current_train_level", None)
    if current_train_level is None:
        raise ValueError("训练入口必须显式提供 current_train_level。")
    TRAIN_PER_PARENT = config.train_per_parent
    USE_ALIGN_LOSS = getattr(config, "use_align_loss", True)
    USE_SUPCON_LOSS = getattr(config, "use_supcon_loss", True)
    ALIGN_LOSS_WEIGHT = getattr(config, "align_loss_weight", 0.05)
    SUPCON_LOSS_WEIGHT = getattr(config, "supcon_loss_weight", 0.03)

    # 读取对齐损失和 SupCon 的衰减起点比例，并写回配置，便于完整记录实验参数
    decay_start_ratio = float(config.decay_start_ratio)
    decay_start_ratio = min(max(decay_start_ratio, 0.0), 1.0)
    config.decay_start_ratio = decay_start_ratio

    runtime_dirs, log, _, log_file, config_log_file = prepare_training_runtime(config)

    # 设备与随机种子
    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"Using device: {device} (config.use_gpu={config.use_gpu}, cuda_available={torch.cuda.is_available()})")
    set_seed(config.seed, deterministic=config.deterministic)
    log(f"Seed set to {config.seed} (deterministic={config.deterministic})")
    log(f"Decay start ratio set to {decay_start_ratio:.3f}")

    def zero_loss(feat):
        return torch.tensor(0.0, device=feat.device)

    # 构建完整数据集对象，用于读取全局层级和类别信息
    full_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config)

    head_names = full_dataset.head_names
    head_name_to_idx = full_dataset.head_name_to_idx
    # 解析当前训练层级
    current_train_level, _ = resolve_level_order(full_dataset, current_train_level)
    if current_train_level not in head_name_to_idx:
        raise ValueError(f"Unknown current_train_level: {current_train_level}")
    _log_missing_upper_level_hint(
        full_dataset,
        current_train_level,
        TRAIN_PER_PARENT,
        config.output_dir,
        log,
    )
    train_idx, test_idx = resolve_train_split(full_dataset, config)
    only_parent = resolve_train_scope(
        full_dataset,
        config,
        current_train_level,
        head_name_to_idx,
    )
    train_idx, test_idx = apply_train_filter(
        full_dataset,
        train_idx,
        test_idx,
        config,
        head_name_to_idx,
    )

    log_split_summary(full_dataset, train_idx, test_idx, current_train_level, head_name_to_idx)

    # 构建训练和验证数据集，并保持层级映射一致
    train_dataset = RamanDataset(
        config.dataset_root,
        augment=True,
        config=config
    )
    test_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config
    )

    # 决定要训练哪些层级
    levels_to_train = resolve_levels_to_train(current_train_level)

    level_models = {}
    parent_models = {}
    def train_single_model(
        model_tag,
        level_name,
        level_idx,
        train_indices,
        test_indices,
        num_classes,
        parent_level_idx=None,
        parent_to_children=None,
        label_map_np=None,
        use_parent_mask=False,
    ):
        # 单层（或父类子模型）训练入口
        model_log_path, model_log, model_log_file = create_model_logger(
            runtime_dirs["logs"],
            model_tag,
            config.timestamp,
            log,
        )
        if len(train_indices) == 0:
            model_log(f"[Skip] {model_tag}: no train samples")
            model_log_file.close()
            return None

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices) if len(test_indices) > 0 else None

        # --------------------------------------------------------
        # 数据加载器
        # --------------------------------------------------------
        train_loader = DataLoader(
            train_subset,
            **_build_loader_kwargs(config, device, train=True),
        )

        test_loader = None
        if test_subset is not None and len(test_subset) > 0:
            test_loader = DataLoader(
                test_subset,
                **_build_loader_kwargs(config, device, train=False),
            )

        # 单层模型
        model = ResNeXt1D_Transformer(
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

        weights = build_class_weights(labels_for_weights, num_classes)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # 主分类损失使用 Focal Loss，并忽略无效标签
        criterion = FocalLoss(
            gamma=config.gamma,
            weight=class_weights,
            ignore_index=-1,
            label_smoothing=config.label_smoothing
        )

        # 动态权重相关
        ema_class_ce = torch.ones(num_classes, device=device)
        ema_momentum = 0.9
        lambda_diff = 0.3
        drw_start_epoch = 10

        # 对齐损失仅约束当前训练层级
        hier_level_weights = {level_name: 1.0}

        supcon_level = train_dataset._resolve_level_name(level_name)

        if USE_ALIGN_LOSS:
            def align_loss_fn(feat, hier_labels):
                return hierarchical_center_loss(
                    feat,
                    hier_labels,
                    hier_level_weights
                )
        else:
            def align_loss_fn(feat, hier_labels):
                return zero_loss(feat)

        if USE_SUPCON_LOSS:
            supcon_criterion = SupConLoss(temperature=config.supcon_tau).to(device)

            def supcon_loss_fn(feat, hier_labels):
                if supcon_level not in hier_labels:
                    return zero_loss(feat)
                labels = hier_labels[supcon_level]
                valid = labels >= 0
                if valid.sum() <= 1:
                    return zero_loss(feat)
                return supcon_criterion(feat[valid], labels[valid])
        else:
            def supcon_loss_fn(feat, hier_labels):
                return zero_loss(feat)

        # 按模块分组学习率：输入 stem 用更小学习率
        group_conv = []
        group_head = []
        group_backbone = []

        def is_stem_param(param_name):
            return param_name.startswith(("conv1", "bn1", "stem_branches"))

        for name, p in model.named_parameters():
            if is_stem_param(name):
                group_conv.append(p)
            elif name.startswith("head") or name.startswith("heads") or name.startswith("center_head"):
                group_head.append(p)
            else:
                group_backbone.append(p)

        optimizer = optim.Adam([
            {"params": group_conv, "lr": config.learning_rate*0.6},
            {"params": group_backbone, "lr": config.learning_rate},
            {"params": group_head, "lr": config.learning_rate*1.2},
        ], weight_decay=5e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler_Tmax, eta_min=config.scheduler_eta_min
        )

        best_score = -1e9
        best_epoch = -1
        patience_counter = 0


        best_model_path = os.path.join(
            config.output_dir,
            f"{model_tag}_model.pt"
        )
        # 将全局类别索引映射成当前子模型使用的局部类别索引
        label_map_tensor = None
        if label_map_np is not None:
            label_map_tensor = torch.tensor(label_map_np, dtype=torch.long, device=device)

        model_log(f"[{model_tag}] ==================== MODEL START ====================")
        model_log(f"[{model_tag}] log_path = {model_log_path}")
        try:
            # 训练一个 epoch
            for epoch in range(1, config.epochs + 1):
                model.train()
                # 按 epoch 更新动态类别权重
                if config.use_drw and epoch >= drw_start_epoch:
                    raw_diff = ema_class_ce / (ema_class_ce.mean() + 1e-12)
                    diff_factor = 1.0 + lambda_diff * (raw_diff - 1.0)
                    dynamic_weights = class_weights * diff_factor
                    dynamic_weights = dynamic_weights / (dynamic_weights.mean() + 1e-12)
                    criterion.weight = dynamic_weights

                running_loss, running_correct, running_total = 0, 0, 0
                running_align_loss = 0.0
                running_supcon_loss = 0.0

                # 对齐损失权重线性启用
                align_w = get_linear_weight(epoch,
                                           start=config.align_start,
                                           end=config.align_end,
                                           w_min=0.0,
                                           w_max=ALIGN_LOSS_WEIGHT
                )
                # SupCon 损失权重线性启用
                supcon_w = get_linear_weight(epoch,
                                            start=config.supcon_start,
                                            end=config.supcon_end,
                                            w_min=0.0,
                                            w_max=SUPCON_LOSS_WEIGHT,
                )
                # 对齐损失和 SupCon 损失在训练后期一起衰减
                decay_start = int(decay_start_ratio * config.epochs)
                decay_ratio = 1.0
                if epoch > decay_start:
                    decay_ratio = 1.0 - (epoch - decay_start) / max(1, config.epochs - decay_start)
                    decay_ratio = max(decay_ratio, 0.2)
                align_w = align_w * decay_ratio
                supcon_w = supcon_w * decay_ratio
                # 训练一个 epoch
                loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")

                for _, (x, y, _) in enumerate(loader_iter):

                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()

                    # ---------- 前向传播 ----------
                    logits, feat = model(x, return_feat=True)

                    # ---------- 标签准备 ----------
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

                    # ---------- 遮罩：只在真实父类的子类中训练 ----------
                    if use_parent_mask and parent_labels is not None:
                        logits_masked, valid_parent = mask_logits_by_parent(
                            logits, parent_labels, parent_to_children
                        )
                    else:
                        logits_masked = logits
                        valid_parent = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

                    valid = (y_level >= 0) & valid_parent

                    # ---------- 主损失 ----------
                    if valid.any():
                        logits_valid = logits_masked[valid]
                        y_valid = y_level[valid]
                        loss_each = criterion(logits_valid, y_valid)

                        if config.use_severity_weight:
                            with torch.no_grad():
                                prob = torch.softmax(logits_valid, dim=1)
                                severity_w = _compute_severity_weights(prob, y_valid)

                            loss_primary = (loss_each * severity_w).mean()
                        else:
                            loss_primary = loss_each.mean()
                    else:
                        loss_primary = torch.tensor(0.0, device=device)

                    # ---------- 可选损失 ----------
                    loss_align = align_loss_fn(feat, hier_labels)
                    loss_supcon = supcon_loss_fn(feat, hier_labels)

                    loss_total = loss_primary + align_w * loss_align + supcon_w * loss_supcon

                    loss_total.backward()
                    optimizer.step()

                    running_align_loss += loss_align.item()
                    running_supcon_loss += loss_supcon.item()

                    running_loss += loss_primary.item()
                    if valid.any():
                        running_correct += (logits_valid.argmax(1) == y_valid).sum().item()
                        running_total += valid.sum().item()

                    postfix = {
                        "cls": f"{running_loss / len(train_loader):.4f}",
                        "acc": f"{100 * running_correct / max(running_total, 1):.2f}%"
                    }
                    loader_iter.set_postfix(postfix)

                    # ---------- 用 EMA 统计各类别当前训练难度 ----------
                    if config.use_drw and valid.any():
                        with torch.no_grad():
                            ce_each = F.cross_entropy(logits_valid, y_valid, reduction="none")
                            for g in range(num_classes):
                                mask = (y_valid == g)
                                if mask.any():
                                    mean_ce = ce_each[mask].mean()
                                    ema_class_ce[g] = (
                                        ema_momentum * ema_class_ce[g]
                                        + (1.0 - ema_momentum) * mean_ce
                                    )

                train_loss = running_loss / len(train_loader)
                train_align_loss = align_w * running_align_loss / max(len(train_loader), 1)
                train_supcon_loss = supcon_w * running_supcon_loss / max(len(train_loader), 1)
                train_acc = running_correct / max(running_total, 1)
                # 验证阶段
                if test_loader is None:
                    test_loss = train_loss
                    test_acc = train_acc
                    test_metrics = {
                        "macro_f1": train_acc,
                        "balanced_acc": train_acc
                    }
                else:
                    if use_parent_mask:
                        test_loss, test_acc, test_metrics = evaluate_file_level(
                            model,
                            test_loader,
                            device,
                            head_index=level_idx,
                            parent_index=parent_level_idx,
                            parent_to_children=parent_to_children
                        )
                    else:
                        test_loss, test_acc, test_metrics = evaluate_file_level_local(
                            model,
                            test_loader,
                            device,
                            head_index=level_idx,
                            label_map_tensor=label_map_tensor
                        )

                macro_f1 = test_metrics["macro_f1"]
                balanced_acc = test_metrics["balanced_acc"]
                # 更新学习率
                scheduler.step()
                if epoch % 10 == 0:
                    model_log(f"  base_w     = {class_weights.detach().cpu().numpy()}")
                    if config.use_drw:
                        model_log(f"  ema_class_ce= {ema_class_ce.detach().cpu().numpy()}")
                        model_log(f"  final_w    = {criterion.weight.detach().cpu().numpy()}")
                model_log(
                    f"[Epoch {epoch}] TrainLoss(cls)={train_loss:.4f}, "
                    f"AlignLossW={train_align_loss:.4f}, "
                    f"SupConLossW={train_supcon_loss:.4f}, "
                    f"TestLoss={test_loss:.4f}\n"
                    f"TrainAcc={train_acc * 100:.2f}%, "
                    f"TestAcc={test_acc * 100:.2f}%, "
                    f"TestMacroF1={macro_f1 * 100:.2f}%, "
                    f"TestBalAcc={balanced_acc * 100:.2f}%, "
                    f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                )
                # Early Stop 使用的综合评分
                score = (
                    config.early_stop_w_f1 * macro_f1
                    + config.early_stop_w_acc * test_acc
                )
                model_log(
                    f"EarlyStop score = "
                    f"{score:.4f} (w_f1={config.early_stop_w_f1}, "
                    f"w_acc={config.early_stop_w_acc})"
                )
                # 保存当前最优模型
                if score >= best_score:
                    best_score = score
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)
                    model_log("  --> Best model updated! (MacroF1 improved)")
                    patience_counter = 0
                else:
                    patience_counter += 1

                model_log(f"[{model_tag}] ------------------------------------------------")

                if patience_counter >= config.patience:
                    model_log("EarlyStopping Triggered by weighted score!")
                    break

            model_log(f"=== Best model epoch: {best_epoch} ===")
        finally:
            model_log_file.close()


        # 释放显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "best_model_path": best_model_path,
            "model": None,
            "level_name": level_name,
            "model_log_path": model_log_path,
        }
    # 逐层训练（全局层或父类子模型）
    for level_name in levels_to_train:
        log(f"[{level_name}] ==================== LEVEL START ====================")
        level_idx = head_name_to_idx[level_name]
        parent_name = full_dataset.get_parent_level(level_name)
        parent_level_idx = head_name_to_idx[parent_name] if parent_name else None
        parent_to_children = full_dataset.parent_to_children.get(level_name, {})
        # 顶层或非按父类训练：训练全局单模型
        if (parent_name is None) or (not TRAIN_PER_PARENT):
            result = train_single_model(
                model_tag=level_name,
                level_name=level_name,
                level_idx=level_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                num_classes=full_dataset.num_classes_by_level[level_name],
                parent_level_idx=parent_level_idx,
                parent_to_children=parent_to_children,
                label_map_np=None,
                use_parent_mask=(parent_level_idx is not None),
            )
            if result is None:
                continue
            level_models[level_name] = os.path.basename(result["best_model_path"])
        else:
            # 父类内子类独立模型
            parent_models[level_name] = {}
            target_parent_idx = int(only_parent) if only_parent is not None else None

            for parent_idx, child_ids in parent_to_children.items():
                if target_parent_idx is not None and int(parent_idx) != target_parent_idx:
                    continue
                child_ids = list(child_ids)
                if not child_ids:
                    continue

                child_names = [
                    full_dataset.class_names_by_level[level_idx][cid]
                    for cid in child_ids
                ]
                # 只有一个子类时不需要训练
                if len(child_ids) <= 1:
                    log(f"parent={parent_idx} only one child, skip training")
                    parent_models[level_name][parent_idx] = {
                        "model_path": None,
                        "child_ids": child_ids,
                        "child_names": child_names
                    }
                    continue

                labels_train = full_dataset.level_labels[train_idx]
                labels_test = full_dataset.level_labels[test_idx]

                train_mask = (labels_train[:, parent_level_idx] == parent_idx) & (
                    labels_train[:, level_idx] >= 0
                )
                test_mask = (labels_test[:, parent_level_idx] == parent_idx) & (
                    labels_test[:, level_idx] >= 0
                )

                train_indices = train_idx[train_mask]
                test_indices = test_idx[test_mask]

                log(
                    f"parent={parent_idx} train={len(train_indices)} "
                    f"test={len(test_indices)} child={child_ids}"
                )

                label_map_np = build_label_map_np(
                    child_ids,
                    full_dataset.num_classes_by_level[level_name]
                )

                result = train_single_model(
                    model_tag=f"{level_name}_parent_{parent_idx}",
                    level_name=level_name,
                    level_idx=level_idx,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    num_classes=len(child_ids),
                    parent_level_idx=None,
                    parent_to_children=None,
                    label_map_np=label_map_np,
                    use_parent_mask=False,
                )

                if result is None:
                    continue

                parent_models[level_name][parent_idx] = {
                    "model_path": os.path.basename(result["best_model_path"]),
                    "child_ids": child_ids,
                    "child_names": child_names
                }

    save_hierarchy_meta(
        config,
        full_dataset,
        head_names,
        current_train_level,
        level_models,
        parent_models,
    )

    config_log_file.close()
    log_file.close()

    return {
        "output_dir": config.output_dir,
        "current_train_level": current_train_level,
        "levels_to_train": levels_to_train,
    }


def main():
    """提示用户改用根目录训练入口。"""
    raise SystemExit("请使用根目录 train.py，并在入口里显式设置 CURRENT_TRAIN_LEVEL。")


if __name__ == "__main__":
    main()
