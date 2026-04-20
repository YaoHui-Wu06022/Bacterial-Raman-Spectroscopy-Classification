"""Raman 层级分类训练脚本"""
from copy import deepcopy
import json
import os
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F

from raman.config import config as default_config
from raman.data import RamanDataset
from raman.eval.experiment import resolve_model_sidecar_path
from raman.eval.common import (
    compute_classification_metrics,
    mask_logits_by_parent,
    select_level_targets,
    select_logits,
)
from raman.model import RamanClassifier1D, SEBlock1D
from raman.training.losses import (
    AlignLoss,
    FocalLoss,
    SupConLoss,
    build_class_weights,
    get_linear_weight,
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
    """训练入口覆盖项，供根目录脚本和 Colab 统一复用"""

    current_train_level: str | None = None
    train_only_parent_name: str | None = None
    train_only_parent: int | None = None
    override_align_loss_weight: float | None = None
    override_supcon_tau: float | None = None
    override_supcon_loss_weight: float | None = None
    override_output_dir: str | None = None


def apply_train_overrides(config, overrides=None):
    """把入口层的手动覆盖项写回配置对象"""
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
    if overrides.override_output_dir is not None:
        config.output_dir = str(overrides.override_output_dir)
    return config


def _load_existing_hierarchy_meta(exp_dir):
    """读取已有实验目录中的层级训练元数据"""
    meta_path = os.path.join(exp_dir, "hierarchy_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _resolve_saved_model_path(exp_dir, model_path):
    """把层级元数据中的模型路径解析成绝对路径"""
    if not model_path:
        return None
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(exp_dir, model_path)


def _build_model_artifact_paths(output_dir, level_name, model_tag):
    """构造按层子目录组织的模型与 sidecar 路径"""
    level_dir = os.path.join(output_dir, level_name)
    os.makedirs(level_dir, exist_ok=True)
    model_path = os.path.join(level_dir, f"{model_tag}_model.pt")
    se_stats_path = resolve_model_sidecar_path(model_path)
    return model_path, se_stats_path


def _build_checkpoint_path(model_path):
    """续训 checkpoint 与模型权重放在同一层目录"""
    suffix = "_model.pt"
    if model_path.endswith(suffix):
        return model_path[: -len(suffix)] + "_checkpoint.pt"
    return model_path + ".checkpoint.pt"


def _save_training_checkpoint(
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


def _restore_training_checkpoint(
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


def _build_relpath(output_dir, path):
    """将模型绝对路径转成相对实验目录的路径"""
    return os.path.relpath(path, output_dir)


def _init_se_stats_accumulator(model):
    """为所有启用的 SEBlock 构造累计统计容器"""
    se_accumulators = {}
    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_use:
            continue
        out_channels = module.fc[-2].out_features
        se_accumulators[name] = {
            "sample_count": 0,
            "channel_sum": torch.zeros(out_channels, dtype=torch.float64),
            "channel_sq_sum": torch.zeros(out_channels, dtype=torch.float64),
            "channel_min": torch.full((out_channels,), float("inf"), dtype=torch.float64),
            "channel_max": torch.full((out_channels,), float("-inf"), dtype=torch.float64),
        }
    return se_accumulators


def _attach_se_scale_hooks(model, batch_scales):
    """在验证前向期间缓存每个 SEBlock 的当前 batch scale"""
    hooks = []
    if not batch_scales and not any(isinstance(module, SEBlock1D) and module.se_use for _, module in model.named_modules()):
        return hooks

    for name, module in model.named_modules():
        if not isinstance(module, SEBlock1D) or not module.se_use:
            continue

        def hook(_module, inputs, _output, block_name=name):
            if not inputs:
                return
            x = inputs[0]
            batch_scales[block_name] = _module._compute_scale(x).detach().cpu().to(torch.float64)

        hooks.append(module.register_forward_hook(hook))

    return hooks


def _accumulate_se_stats(se_accumulators, batch_scales, valid_mask):
    """把当前 batch 中有效样本的 scale 合并进累计统计"""
    if not batch_scales:
        return

    valid_mask_cpu = valid_mask.detach().cpu()
    if not valid_mask_cpu.any():
        return

    for name, scale in batch_scales.items():
        valid_scale = scale[valid_mask_cpu]
        if valid_scale.numel() == 0:
            continue
        stats = se_accumulators[name]
        stats["sample_count"] += int(valid_scale.size(0))
        stats["channel_sum"] += valid_scale.sum(dim=0)
        stats["channel_sq_sum"] += (valid_scale * valid_scale).sum(dim=0)
        stats["channel_min"] = torch.minimum(stats["channel_min"], valid_scale.min(dim=0).values)
        stats["channel_max"] = torch.maximum(stats["channel_max"], valid_scale.max(dim=0).values)


def _finalize_se_stats(se_accumulators):
    """将累计量还原成最终 SE 统计结果"""
    se_stats = {}
    for name, stats in se_accumulators.items():
        sample_count = int(stats["sample_count"])
        if sample_count <= 0:
            continue
        channel_mean = stats["channel_sum"] / sample_count
        channel_var = stats["channel_sq_sum"] / sample_count - channel_mean * channel_mean
        channel_var = torch.clamp(channel_var, min=0.0)
        channel_std = torch.sqrt(channel_var)
        se_stats[name] = {
            "sample_count": sample_count,
            "channel_mean": channel_mean.to(torch.float32),
            "channel_std": channel_std.to(torch.float32),
            "channel_min": stats["channel_min"].to(torch.float32),
            "channel_max": stats["channel_max"].to(torch.float32),
        }
    return se_stats


def _is_parent_entry_covered(exp_dir, entry):
    """判断某个 parent 条目是否足以支持级联"""
    child_ids = list(entry.get("child_ids", []))
    model_path = entry.get("model_path")
    if model_path:
        full_path = _resolve_saved_model_path(exp_dir, model_path)
        return full_path is not None and os.path.exists(full_path)
    return len(child_ids) <= 1


def _build_loader_kwargs(config, device, train=True):
    """按当前配置构造 DataLoader 参数"""
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


def _evaluate_validation_loader(
    model,
    loader,
    device,
    head_index,
    head_name=None,
    label_map_tensor=None,
    parent_index=None,
    parent_to_children=None,
):
    """训练期验证入口：统一处理层级标签、局部映射与父类遮罩"""
    model.eval()
    criterion_eval = torch.nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    all_targets = []
    all_preds = []
    num_classes = None
    se_accumulators = _init_se_stats_accumulator(model)
    batch_scales = {}
    se_hooks = _attach_se_scale_hooks(model, batch_scales)

    try:
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device)

                batch_scales.clear()
                logits = select_logits(model(x), head_name=head_name)
                y_level = select_level_targets(y, head_index)

                if label_map_tensor is not None:
                    invalid = y_level < 0
                    y_level = label_map_tensor[y_level.clamp_min(0)]
                    y_level[invalid] = -1

                if parent_index is not None and parent_to_children is not None:
                    if y.ndim != 2:
                        raise ValueError("parent_index 需要完整的多层标签输入")
                    parent_labels = y[:, parent_index]
                    logits, valid_parent = mask_logits_by_parent(
                        logits,
                        parent_labels,
                        parent_to_children,
                    )
                else:
                    valid_parent = torch.ones_like(y_level, dtype=torch.bool)

                valid = (y_level >= 0) & valid_parent
                _accumulate_se_stats(se_accumulators, batch_scales, valid)
                if not valid.any():
                    continue

                logits = logits[valid]
                y_valid = y_level[valid]

                if num_classes is None:
                    num_classes = logits.size(1)

                loss = criterion_eval(logits, y_valid)
                batch_size = y_valid.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size

                all_preds.append(logits.argmax(1).detach().cpu().numpy())
                all_targets.append(y_valid.detach().cpu().numpy())
    finally:
        for hook in se_hooks:
            hook.remove()

    if num_classes is None or not all_targets:
        metrics = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_recall": 0.0,
        }
        return 0.0, 0.0, metrics, _finalize_se_stats(se_accumulators)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=range(num_classes),
    )
    return total_loss / max(total, 1), metrics["accuracy"], metrics, _finalize_se_stats(se_accumulators)


def _has_level_coverage(exp_dir, full_dataset, level_name):
    """判断实验目录里是否已有某一上级层的完整训练结果"""
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

    default_level_model = os.path.join(exp_dir, level_name, f"{level_name}_model.pt")
    if os.path.exists(default_level_model):
        return True

    prefix = f"{level_name}_parent_"
    suffix = "_model.pt"
    level_dir = os.path.join(exp_dir, level_name)
    if not os.path.isdir(level_dir):
        return False
    for name in os.listdir(level_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            return True
    return False


def _log_missing_upper_level_hint(full_dataset, current_train_level, train_per_parent, exp_dir, log):
    """缺少上一级结果时给出非阻断提示"""
    if not train_per_parent:
        return

    upper_level_name = full_dataset.get_parent_level(current_train_level)
    if upper_level_name is None:
        return

    if _has_level_coverage(exp_dir, full_dataset, upper_level_name):
        return

    log(
        f"[Hint] train_per_parent=True，但当前实验目录缺少上一级 "
        f"{upper_level_name} 的完整模型记录"
    )
    log(
        f"[Hint] 若后续需要在同一 EXP_DIR 中继续向下训练、级联预测或测试评估，"
        f"建议先训练 current_train_level={upper_level_name}"
    )


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



def run_training(config_obj=None, overrides=None):
    """执行一次完整训练流程"""
    config = deepcopy(config_obj or default_config)
    config = apply_train_overrides(config, overrides)
    current_train_level = getattr(config, "current_train_level", None)
    if current_train_level is None:
        raise ValueError("训练入口必须显式提供 current_train_level")
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
    business_head_names = full_dataset.level_names
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
        if test_loader is None:
            raise ValueError(
                f"{model_tag} 没有可用的验证样本，当前实现要求每个参与训练的 leaf "
                "在切分后都能为验证集提供样本"
            )

        # 单层模型
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

        # 主分类损失使用 Focal Loss，并忽略无效标签
        criterion = FocalLoss(
            gamma=config.gamma,
            weight=base_class_weights,  # EMA 动态权重启用后会替换为 ema_class_weights
            ignore_index=-1,
        )

        # EMA 动态类别权重相关
        ema_class_ce = torch.ones(num_classes, device=device)
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
            def align_loss_fn(feat, hier_labels):
                return zero_loss(feat)

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
            def supcon_loss_fn(feat, hier_labels):
                return zero_loss(feat)

        # 按模块分组学习率：输入 stem 用更小学习率
        group_conv = []
        group_head = []
        group_backbone = []

        def is_stem_param(param_name):
            return param_name.startswith("stem_branches")

        for name, p in model.named_parameters():
            if is_stem_param(name):
                group_conv.append(p)
            elif name.startswith("head"):
                group_head.append(p)
            else:
                group_backbone.append(p)

        optimizer = optim.Adam([
            {"params": group_conv, "lr": config.learning_rate*0.6},
            {"params": group_backbone, "lr": config.learning_rate},
            {"params": group_head, "lr": config.learning_rate*1.2},
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler_Tmax, eta_min=config.scheduler_eta_min
        )

        best_score = -1e9
        best_epoch = -1
        patience_counter = 0


        best_model_path, best_se_stats_path = _build_model_artifact_paths(
            config.output_dir,
            level_name,
            model_tag,
        )
        checkpoint_path = _build_checkpoint_path(best_model_path)
        # 将全局类别索引映射成当前子模型使用的局部类别索引
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
                    best_score,
                    best_epoch,
                    patience_counter,
                    restored_ema_class_ce,
                ) = _restore_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    model_log,
                )
                if restored_ema_class_ce is not None:
                    ema_class_ce = restored_ema_class_ce

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

            # 训练一个 epoch
            for epoch in range(start_epoch, config.epochs + 1):
                model.train()
                # 延迟启用基于 EMA 类别难度的动态类别权重
                if config.use_ema and epoch >= ema_start_epoch:
                    raw_diff = ema_class_ce / (ema_class_ce.mean() + 1e-12)
                    diff_factor = 1.0 + lambda_diff * (raw_diff - 1.0)
                    ema_class_weights = base_class_weights * diff_factor
                    ema_class_weights = ema_class_weights / (ema_class_weights.mean() + 1e-12)
                    criterion.weight = ema_class_weights

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

                    # ---------- 可选损失 ----------
                    loss_align = align_loss_fn(feat, hier_labels)
                    loss_supcon = supcon_loss_fn(feat, hier_labels)

                    loss_total = loss_cls + align_w * loss_align + supcon_w * loss_supcon

                    loss_total.backward()
                    optimizer.step()

                    running_align_loss += loss_align.item()
                    running_supcon_loss += loss_supcon.item()

                    running_loss += loss_cls.item()
                    if valid.any():
                        running_correct += (logits_valid.argmax(1) == y_valid).sum().item()
                        running_total += valid.sum().item()

                    postfix = {
                        "cls": f"{running_loss / len(train_loader):.4f}",
                        "acc": f"{100 * running_correct / max(running_total, 1):.2f}%"
                    }
                    loader_iter.set_postfix(postfix)

                    # ---------- 用 EMA 统计各类别当前训练难度 ----------
                    if config.use_ema and valid.any():
                        with torch.no_grad():
                            ce_each = F.cross_entropy(logits_valid, y_valid, reduction="none")
                            for c in range(num_classes):
                                class_mask_c = (y_valid == c)
                                if class_mask_c.any():
                                    mean_ce_c = ce_each[class_mask_c].mean()
                                    ema_class_ce[c] = ema_alpha * ema_class_ce[c] + (1.0 - ema_alpha) * mean_ce_c

                train_loss = running_loss / len(train_loader)
                train_align_loss = align_w * running_align_loss / max(len(train_loader), 1)
                train_supcon_loss = supcon_w * running_supcon_loss / max(len(train_loader), 1)
                train_acc = running_correct / max(running_total, 1)
                # 验证阶段
                test_loss, test_acc, test_metrics, se_stats = _evaluate_validation_loader(
                    model,
                    test_loader,
                    device,
                    head_index=level_idx,
                    head_name=level_name,
                    label_map_tensor=None if use_parent_mask else label_map_tensor,
                    parent_index=parent_level_idx if use_parent_mask else None,
                    parent_to_children=parent_to_children if use_parent_mask else None,
                )
                macro_f1 = test_metrics["macro_f1"]
                macro_recall = test_metrics["macro_recall"]
                # 更新学习率
                scheduler.step()
                if epoch % 10 == 0:
                    model_log(f"  base_class_weights = {base_class_weights.detach().cpu().numpy()}")
                    if config.use_ema:
                        model_log(f"  ema_class_ce       = {ema_class_ce.detach().cpu().numpy()}")
                        model_log(f"  ema_class_weights  = {criterion.weight.detach().cpu().numpy()}")
                model_log(
                    f"[Epoch {epoch}] TrainLoss(cls)={train_loss:.4f}, "
                    f"AlignLossW={train_align_loss:.4f}, "
                    f"SupConLossW={train_supcon_loss:.4f}, "
                    f"TestLoss={test_loss:.4f}\n"
                    f"TrainAcc={train_acc * 100:.2f}%, "
                    f"TestAcc={test_acc * 100:.2f}%, "
                    f"TestMacroF1={macro_f1 * 100:.2f}%, "
                    f"TestMacroRecall={macro_recall * 100:.2f}%, "
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
                    if se_stats:
                        torch.save(se_stats, best_se_stats_path)
                    model_log("  --> Best model updated! (EarlyStop score improved)")
                    patience_counter = 0
                else:
                    patience_counter += 1

                checkpoint_interval = max(1, int(getattr(config, "checkpoint_interval", 10)))
                should_save_checkpoint = epoch % checkpoint_interval == 0
                if should_save_checkpoint:
                    _save_training_checkpoint(
                        checkpoint_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        best_score,
                        best_epoch,
                        patience_counter,
                        ema_class_ce,
                    )
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
            "best_se_stats_path": best_se_stats_path,
            "checkpoint_path": checkpoint_path,
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
                use_parent_mask=False,
            )
            if result is None:
                continue
            level_models[level_name] = _build_relpath(config.output_dir, result["best_model_path"])
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
                    "model_path": _build_relpath(config.output_dir, result["best_model_path"]),
                    "child_ids": child_ids,
                    "child_names": child_names
                }

    save_hierarchy_meta(
        config,
        full_dataset,
        business_head_names,
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
    """提示用户改用根目录训练入口"""
    raise SystemExit("请使用根目录 train.py，并在入口里显式设置 CURRENT_TRAIN_LEVEL")


if __name__ == "__main__":
    main()
