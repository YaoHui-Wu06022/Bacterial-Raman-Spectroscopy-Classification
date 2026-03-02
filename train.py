"""Raman 层级分类训练脚本。"""
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import json

from raman.config import config
from raman.model import ResNeXt1D_Transformer
from raman.dataset import RamanDataset
from raman.config_io import dump_config_to_yaml
from raman.train_utils import(
    prepare_output_dirs,
    evaluate_file_level,
    get_linear_weight,
    FocalLoss,
    hierarchical_center_loss,
    SupConLoss,
    split_by_lowest_level_ratio,
    save_split_files,
    mask_logits_by_parent,
    classification_metrics
)

# ============================================================
# 手动覆盖（仅改这里，不改 config 文件）
# - 适合在 Colab 里快速单独训练某个层级/父类
# ============================================================
TRAIN_ONLY_LEVEL = "level_1"          # 例如 "level_2"
TRAIN_ONLY_PARENT_NAME = None    # 例如 "dachang"
TRAIN_ONLY_PARENT = None        # 例如 2（可选，优先级高于名称）
# 可选：手动覆盖衰减起点比例（None 表示走 config 自动判断）
OVERRIDE_DECAY_START_RATIO = None

# 耐药level_3用
# config.supcon_start = 25
# config.supcon_end = 45

# 可选：覆盖损失参数（单独训练时可能不同）
OVERRIDE_ALIGN_LOSS_WEIGHT = None # 耐药level3:0
OVERRIDE_SUPCON_TAU = None
OVERRIDE_SUPCON_LOSS_WEIGHT = None
# 可选：指定 SupCon 使用的层级（None 表示跟随当前训练层级）
SUPCON_LEVEL_OVERRIDE = "level_1"

# 可选：固定输出目录/时间戳，避免切换 config 导致输出分散
OVERRIDE_TIMESTAMP = None
OVERRIDE_OUTPUT_DIR = None
# ============================================================
# 应用覆盖到 config
# ============================================================
if TRAIN_ONLY_PARENT_NAME is not None:
    config.train_only_parent_name = TRAIN_ONLY_PARENT_NAME
if TRAIN_ONLY_PARENT is not None:
    config.train_only_parent = TRAIN_ONLY_PARENT
if TRAIN_ONLY_LEVEL is not None:
    config.train_only_level = TRAIN_ONLY_LEVEL

if OVERRIDE_ALIGN_LOSS_WEIGHT is not None:
    config.align_loss_weight = float(OVERRIDE_ALIGN_LOSS_WEIGHT)
if OVERRIDE_SUPCON_TAU is not None:
    config.supcon_tau = float(OVERRIDE_SUPCON_TAU)
if OVERRIDE_SUPCON_LOSS_WEIGHT is not None:
    config.supcon_loss_weight = float(OVERRIDE_SUPCON_LOSS_WEIGHT)
if SUPCON_LEVEL_OVERRIDE is not None:
    config.supcon_level = SUPCON_LEVEL_OVERRIDE
if OVERRIDE_DECAY_START_RATIO is not None:
    config.decay_start_ratio = float(OVERRIDE_DECAY_START_RATIO)

if OVERRIDE_TIMESTAMP is not None:
    config.timestamp = str(OVERRIDE_TIMESTAMP)
if OVERRIDE_OUTPUT_DIR is not None:
    config.output_dir = str(OVERRIDE_OUTPUT_DIR)

TRAIN_LEVEL = config.train_level or "leaf"
TRAIN_PER_PARENT = config.train_per_parent

# Loss config（全局设置）
USE_ALIGN_LOSS = getattr(config, "use_align_loss", True)
USE_SUPCON_LOSS = getattr(config, "use_supcon_loss", True)
ALIGN_LOSS_WEIGHT = getattr(config, "align_loss_weight", 0.05)
SUPCON_LOSS_WEIGHT = getattr(config, "supcon_loss_weight", 0.03)
SUPCON_LEVEL = getattr(config, "supcon_level", TRAIN_LEVEL)

def set_seed(seed, deterministic=True):
    # 统一随机种子，保证可复现
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_level_order(dataset, target_level):
    # 解析训练层级顺序，并确保目标层级合法
    target_level = dataset._resolve_level_name(target_level)
    if target_level not in dataset.head_names:
        target_level = "leaf"
    level_order = []
    for name in dataset.head_names:
        level_order.append(name)
        if name == target_level:
            break
    return target_level, level_order


def _normalize_filter_values(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return [val]


def _resolve_parent_idx_by_name(dataset, parent_level_idx, parent_name):
    if parent_name is None:
        return None
    name_to_idx = dataset.label_maps_by_level[parent_level_idx]
    return name_to_idx.get(parent_name)

def build_class_weights(level_labels, num_classes):
    # 根据当前层级标签统计类别权重（对数平滑）
    valid = level_labels >= 0
    if not valid.any():
        return np.ones(num_classes, dtype=np.float32)
    counts = np.bincount(level_labels[valid], minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.log(counts + 1.5)
    weights = weights / weights.mean()
    return weights.astype(np.float32)

def build_label_map_np(child_ids, num_classes):
    # 将全局类别 id 映射为当前子模型的局部 id
    mapping = np.full(num_classes, -1, dtype=np.int64)
    for local_idx, global_idx in enumerate(child_ids):
        mapping[int(global_idx)] = int(local_idx)
    return mapping

def evaluate_file_level_local(
    model,
    loader,
    device,
    head_index,
    label_map_tensor
):
    # 不使用父类遮罩时的单层评估（适配局部 label_map）
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds = []
    all_targets = []
    num_classes = None

    criterion_eval = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            if y.ndim == 2:
                y = y[:, head_index]

            if label_map_tensor is not None:
                invalid = y < 0
                y = label_map_tensor[y.clamp_min(0)]
                y[invalid] = -1

            if num_classes is None:
                num_classes = logits.size(1)

            valid = y >= 0
            if not valid.any():
                continue

            logits = logits[valid]
            y = y[valid]

            loss = criterion_eval(logits, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total += bs

            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    if num_classes is None:
        return 0.0, 0.0, {"macro_f1": 0.0, "balanced_acc": 0.0}

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    metrics = classification_metrics(y_true, y_pred, num_classes)

    return total_loss / max(total, 1), total_correct / max(total, 1), metrics

def main():
    # 允许外部手动指定 output_dir；未设置才自动生成
    if config.timestamp is None:
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_dir is None:
        config.output_dir = os.path.join(
            f"output_{TRAIN_LEVEL}",
            config.timestamp
        )

    # 解析对齐/SupCon 衰减起点比例，并写回 config 以便完整记录到 config.yaml
    if hasattr(config, "resolve_decay_start_ratio"):
        decay_start_ratio = float(config.resolve_decay_start_ratio())
    else:
        decay_start_ratio = float(getattr(config, "decay_start_ratio", 0.7))
    decay_start_ratio = min(max(decay_start_ratio, 0.0), 1.0)
    config.decay_start_ratio = decay_start_ratio

    # 准备输出目录（日志/模型等）
    dirs = prepare_output_dirs(config=config)

    dump_config_to_yaml(
        config,
        os.path.join(config.output_dir, "config.yaml")
    )
    log_file = open(os.path.join(dirs["logs"], "log.txt"), "w", buffering=1)
    config_log_file = open(os.path.join(dirs["logs"], "config.txt"), "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    def config_log(msg):
        config_log_file.write(msg + "\n")

    # 记录配置，便于复现
    config_log("===== Run Meta =====")
    config_log(f"Experiment timestamp: {config.timestamp}")
    config_log(f"Output dir: {config.output_dir}")
    config_log("=====================\n")
    config_log("===== Full Config Dump =====")

    for k in sorted(dir(config)):
        if k.startswith("_"):
            continue

        try:
            v = getattr(config, k)
        except Exception:
            continue

        if callable(v):
            continue

        config_log(f"{k}: {v}")

    # 设备与随机种子
    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"Using device: {device} (config.use_gpu={config.use_gpu}, cuda_available={torch.cuda.is_available()})")
    set_seed(config.seed, deterministic=config.deterministic)
    log(f"Seed set to {config.seed} (deterministic={config.deterministic})")
    log(f"Decay start ratio set to {decay_start_ratio:.3f}")

    def zero_loss(feat):
        return torch.tensor(0.0, device=feat.device)

    # 构建完整 Dataset（用于全局层级/类别信息）
    full_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config)

    head_names = full_dataset.head_names
    head_name_to_idx = full_dataset.head_name_to_idx
    # 解析训练层级与需要训练的层级顺序
    train_level, level_order = resolve_level_order(full_dataset, TRAIN_LEVEL)
    if train_level not in head_name_to_idx:
        raise ValueError(f"Unknown train_level: {train_level}")
    # 按指定层级进行样本级划分
    split_level = config.split_level or "leaf"

    # ============================================================
    # 样本级划分
    # - 按 split_level 分组划分
    # ============================================================
    train_idx, test_idx = split_by_lowest_level_ratio(
        full_dataset,
        lowest_level=split_level,
        train_ratio=config.train_split,
        seed=config.seed
    )

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    # 保存全量切分（跨机器复现），已有则不覆盖
    from raman.train_utils import load_split_files
    existing_split = load_split_files(full_dataset, config.output_dir)
    if existing_split is None:
        save_split_files(full_dataset, train_idx, test_idx, config.output_dir)
    else:
        train_idx, test_idx = existing_split

    # ============================================================
    # 只训练指定层级/父类：简化配置（只需 level + 父类名称）
    # ============================================================
    only_level = getattr(config, "train_only_level", None)
    if only_level is not None:
        only_level = full_dataset._resolve_level_name(only_level)
    else:
        only_level = train_level

    only_parent = getattr(config, "train_only_parent", None)
    only_parent_name = getattr(config, "train_only_parent_name", None)
    if only_parent_name is not None and only_parent is None:
        parent_level = full_dataset.get_parent_level(only_level)
        if parent_level is None:
            raise ValueError(
                f"train_only_parent_name requires a parent level for {only_level}"
            )
        parent_level_idx = head_name_to_idx[parent_level]
        resolved = _resolve_parent_idx_by_name(
            full_dataset, parent_level_idx, only_parent_name
        )
        if resolved is None:
            raise ValueError(
                f"train_only_parent_name not found in {parent_level}: {only_parent_name}"
            )
        only_parent = int(resolved)
        config.train_only_parent = only_parent
        # 自动设置筛选条件（保证先切分、后筛选）
        if getattr(config, "train_filter_level", None) is None and getattr(config, "train_filter_value", None) is None:
            config.train_filter_level = parent_level
            config.train_filter_value = only_parent_name

    # ============================================================
    # 训练集筛选（先全量切分，再筛选；不影响 evalute/analysis）
    # ============================================================
    filter_level = getattr(config, "train_filter_level", None)
    filter_value = getattr(config, "train_filter_value", None)
    if filter_level and filter_value is not None:
        filter_level = full_dataset._resolve_level_name(filter_level)
        if filter_level not in head_name_to_idx:
            raise ValueError(
                f"Unknown train_filter_level: {filter_level}. Available: {head_names}"
            )
        filter_level_idx = head_name_to_idx[filter_level]
        values = _normalize_filter_values(filter_value)
        desired_ids = set()
        for v in values:
            if isinstance(v, int):
                desired_ids.add(int(v))
            else:
                idx = full_dataset.label_maps_by_level[filter_level_idx].get(str(v))
                if idx is None:
                    print(f"[Warn] train_filter_value not found in {filter_level}: {v}")
                    continue
                desired_ids.add(int(idx))

        if not desired_ids:
            raise ValueError("No valid train_filter_value found; check config.")

        labels_filter = full_dataset.level_labels[:, filter_level_idx]
        mask = np.isin(labels_filter, list(desired_ids))
        train_idx = train_idx[mask[train_idx]]
        test_idx = test_idx[mask[test_idx]]
        print(
            f"[Filter] level={filter_level}, values={values} -> "
            f"Train {len(train_idx)}, Test {len(test_idx)}"
        )

    # 统计样本分布时跟随当前训练层级（单独训练时用 only_level）
    stats_level = only_level or train_level
    stats_level_idx = head_name_to_idx[stats_level]
    labels_train_level = full_dataset.level_labels[:, stats_level_idx]

    print(f"[Sample-level Split] Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    print(
        f"Train {stats_level} counts:",
        np.bincount(labels_train_level[train_idx][labels_train_level[train_idx] >= 0],
                   minlength=full_dataset.num_classes_by_level[stats_level])
    )
    print(
        f"Test  {stats_level} counts:",
        np.bincount(labels_train_level[test_idx][labels_train_level[test_idx] >= 0],
                   minlength=full_dataset.num_classes_by_level[stats_level])
    )

    # 构建训练/测试 Dataset（保持同一映射）
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
    if TRAIN_PER_PARENT:
        levels_to_train = level_order
    else:
        levels_to_train = [train_level]

    if only_level:
        levels_to_train = [n for n in levels_to_train if n == only_level]
        if not levels_to_train:
            raise ValueError(f"train_only_level not found: {only_level}")

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
        if len(train_indices) == 0:
            log(f"[Skip] {model_tag}: no train samples")
            return None

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices) if len(test_indices) > 0 else None

        # --------------------------------------------------------
        # 数据加载器
        # --------------------------------------------------------
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = None
        if test_subset is not None and len(test_subset) > 0:
            test_loader = DataLoader(
                test_subset,
                batch_size=config.batch_size,
                shuffle=False
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

        # FocalLoss（忽略无效标签）
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

        supcon_level = train_dataset._resolve_level_name(SUPCON_LEVEL)

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

        # 分组学习率：stem 用小一点 lr，后面层用正常 lr
        group_conv = []
        group_head = []
        group_backbone = []

        for name, p in model.named_parameters():
            if name.startswith("conv1") or name.startswith("bn1"):
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
        # label_map：全局 -> 局部类别
        label_map_tensor = None
        if label_map_np is not None:
            label_map_tensor = torch.tensor(label_map_np, dtype=torch.long, device=device)

        log(f"[{model_tag}] ==================== MODEL START ====================")
        # 训练一个 epoch
        for epoch in range(1, config.epochs + 1):
            model.train()
            # ============================================================
            # Epoch-level: 更新动态 class weight（DRW-aware）
            # ============================================================
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
            # SupCon 权重线性启用
            supcon_w = get_linear_weight(epoch,
                                        start=config.supcon_start,
                                        end=config.supcon_end,
                                        w_min=0.0,
                                        w_max=SUPCON_LOSS_WEIGHT,
            )
            # 对齐 / SupCon 后期衰减
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
                            prob = torch.softmax(logits_valid, dim=1)  # [B, C]
                            topk = min(3, prob.size(1))
                            topk_val, topk_idx = prob.topk(topk, dim=1)  # top-3 prediction

                            # 是否 top1 正确
                            is_top1 = (topk_idx[:, 0] == y_valid)

                            # --------------------------------------------------------
                            # 正确类别在 top-k 中的 rank
                            # - rank = 1 / 2 / 3 ：进入 top-k
                            # - rank = 4         ：未进入 top-k（默认）
                            # --------------------------------------------------------
                            rank = torch.full_like(topk_idx[:, 0], fill_value=4)

                            for k in range(topk):
                                rank[topk_idx[:, k] == y_valid] = k + 1

                            # 默认权重 = 1（错误且不在 top-k）
                            severity_w = torch.ones_like(rank, dtype=prob.dtype)

                            # -------- rank-aware soft penalty --------
                            num_classes = prob.size(1)
                            if num_classes >= 3:
                                if topk >= 2:
                                    severity_w[rank == 2] = 0.8
                                if topk >= 3:
                                    severity_w[rank == 3] = 0.9
                            """
                            if topk >= 2:
                                severity_w[rank == 2] = 0.8
                            if topk >= 3:
                                severity_w[rank == 3] = 0.9
                            """
                            # -------- 错得很离谱：高置信度 top1 错 --------
                            top1_conf = topk_val[:, 0]
                            high_conf_wrong = (~is_top1) & (top1_conf > 0.8)
                            severity_w[high_conf_wrong] = 2.0

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

                # ---------- 动态难度统计（EMA） ----------
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
                log(f"  base_w     = {class_weights.detach().cpu().numpy()}")
                if config.use_drw:
                    log(f"  ema_class_ce= {ema_class_ce.detach().cpu().numpy()}")
                    log(f"  final_w    = {criterion.weight.detach().cpu().numpy()}")
            log(
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
            # EarlyStop 评分
            score = (
                config.early_stop_w_f1 * macro_f1
                + config.early_stop_w_acc * test_acc
            )
            log(
                f"EarlyStop score = "
                f"{score:.4f} (w_f1={config.early_stop_w_f1}, "
                f"w_acc={config.early_stop_w_acc})"
            )
            # 保存最优模型
            if score >= best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                log("  --> Best model updated! (MacroF1 improved)")
                patience_counter = 0
            else:
                patience_counter += 1

            log(f"[{model_tag}] ------------------------------------------------")

            if patience_counter >= config.patience:
                log("EarlyStopping Triggered by weighted score!")
                break

        log(f"=== Best model epoch: {best_epoch} ===")


        # 释放显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "best_model_path": best_model_path,
            "model": None,
            "level_name": level_name,
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

    # 保存 class_names.json（全层级）
    class_names_path = os.path.join(config.output_dir, "class_names.json")
    class_names_by_level = {
        name: full_dataset.class_names_by_level[i]
        for i, name in enumerate(head_names)
    }
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(
            class_names_by_level,
            f,
            indent=2,
            ensure_ascii=False
        )

    # 保存层级元数据，便于级联推理
    parent_to_children_json = {
        level: {str(k): v for k, v in mapping.items()}
        for level, mapping in full_dataset.parent_to_children.items()
    }
    level_models_json = {
        k: v for k, v in level_models.items()
    }
    parent_models_json = {
        level: {str(k): v for k, v in mapping.items()}
        for level, mapping in parent_models.items()
    }

    # 合并历史 meta（避免单训练覆盖已有模型记录）
    hier_meta_path = os.path.join(config.output_dir, "hierarchy_meta.json")
    if os.path.exists(hier_meta_path):
        try:
            with open(hier_meta_path, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
        except Exception:
            old_meta = {}
        old_level_models = old_meta.get("level_models", {}) if isinstance(old_meta, dict) else {}
        old_parent_models = old_meta.get("parent_models", {}) if isinstance(old_meta, dict) else {}

        # merge level_models
        merged_level_models = dict(old_level_models)
        merged_level_models.update(level_models_json)
        level_models_json = merged_level_models

        # merge parent_models (level -> {parent_idx: entry})
        merged_parent_models = {}
        for level, mapping in old_parent_models.items():
            if isinstance(mapping, dict):
                merged_parent_models[level] = dict(mapping)
        for level, mapping in parent_models_json.items():
            merged_parent_models.setdefault(level, {})
            merged_parent_models[level].update(mapping)
        parent_models_json = merged_parent_models

    with open(hier_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "head_names": head_names,
                "level_names": full_dataset.level_names,
                "class_names_by_level": class_names_by_level,
                "parent_to_children": parent_to_children_json,
                "parent_level_name": full_dataset.parent_level_name,
                "train_level": train_level,
                "level_models": level_models_json,
                "parent_models": parent_models_json,
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    config_log_file.close()
    log_file.close()

if __name__ == "__main__":
    main()
