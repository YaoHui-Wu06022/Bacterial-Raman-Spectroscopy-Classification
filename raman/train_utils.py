import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Sampler, Subset
from collections import defaultdict, Counter
import json

# 准备输出目录
def prepare_output_dirs(config):
    base = config.output_dir
    dirs = {
        "base": base,
        "logs": os.path.join(base, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _norm_relpath(path):
    return os.path.normpath(path).replace("\\", "/")


def save_split_files(dataset, train_idx, test_idx, out_dir):
    """Save split file lists as paths relative to dataset.root_dir."""
    os.makedirs(out_dir, exist_ok=True)
    root = dataset.root_dir
    samples = dataset.samples
    train_files = [
        _norm_relpath(os.path.relpath(samples[i], root)) for i in train_idx
    ]
    test_files = [
        _norm_relpath(os.path.relpath(samples[i], root)) for i in test_idx
    ]
    with open(os.path.join(out_dir, "train_files.json"), "w", encoding="utf-8") as f:
        json.dump(train_files, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "test_files.json"), "w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2, ensure_ascii=False)


def load_split_files(dataset, split_dir):
    """Load split file lists and map to dataset indices. Return (train_idx, test_idx) or None."""
    train_path = os.path.join(split_dir, "train_files.json")
    test_path = os.path.join(split_dir, "test_files.json")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    with open(train_path, "r", encoding="utf-8") as f:
        train_files = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_files = json.load(f)

    root = dataset.root_dir
    samples = dataset.samples
    rel_to_idx = {}
    for i, p in enumerate(samples):
        rel = _norm_relpath(os.path.relpath(p, root))
        rel_to_idx[rel] = i

    def map_list(items, kind):
        idxs = []
        missing = []
        for rel in items:
            rel_n = _norm_relpath(rel)
            if rel_n not in rel_to_idx:
                missing.append(rel)
            else:
                idxs.append(rel_to_idx[rel_n])
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} {kind} files from split list; "
                f"first: {missing[0]}"
            )
        return np.array(sorted(idxs))

    train_idx = map_list(train_files, "train")
    test_idx = map_list(test_files, "test")
    return train_idx, test_idx

def classification_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t < 0 or p < 0:
            continue
        cm[t, p] += 1

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    support = tp + fn
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(support, 1)
    denom = precision + recall
    f1 = np.zeros_like(denom, dtype=float)
    np.divide(
        2 * precision * recall,
        denom,
        out=f1,
        where=denom > 0
    )
    valid = support > 0
    if valid.any():
        macro_f1 = float(np.mean(f1[valid]))
        balanced_acc = float(np.mean(recall[valid]))
    else:
        macro_f1 = 0.0
        balanced_acc = 0.0

    return {"macro_f1": macro_f1, "balanced_acc": balanced_acc}

# 准确度评价(file level)
# 统一选择输出 logits
def _select_logits(pred, head_name=None):
    if isinstance(pred, dict):
        if head_name is None:
            head_name = list(pred.keys())[-1]
        return pred[head_name]
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred

def mask_logits_by_parent(logits, parent_labels, parent_to_children):
    """
    使用父类标签对 logits 做遮罩，仅保留该父类的子类
    返回:
        - masked_logits
        - valid_mask（哪些样本存在可用子类）
    """
    if parent_labels is None or parent_to_children is None:
        valid = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, valid

    device = logits.device
    batch = logits.size(0)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    valid = torch.zeros(batch, dtype=torch.bool, device=device)

    for i, p in enumerate(parent_labels.tolist()):
        if p < 0:
            continue
        child_idx = parent_to_children.get(p)
        if child_idx is None:
            child_idx = parent_to_children.get(str(p))
        if not child_idx:
            continue

        # 索引越界会触发 CUDA device-side assert，这里提前检查
        num_classes = logits.size(1)
        child_list = list(child_idx)
        invalid = [c for c in child_list if c < 0 or c >= num_classes]
        if invalid:
            raise ValueError(
                f"parent_to_children 索引越界: parent={p}, invalid={invalid}, num_classes={num_classes}"
            )

        mask[i, child_list] = True
        valid[i] = True

    masked_logits = logits.masked_fill(~mask, float("-inf"))
    if (~valid).any():
        masked_logits[~valid] = 0.0

    return masked_logits, valid

# 文件级评估
def evaluate_file_level(
    model,
    loader,
    device,
    head_index=None,
    head_name=None,
    parent_index=None,
    parent_to_children=None
):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds = []
    all_targets = []
    num_classes = None

    criterion_eval = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            logits = _select_logits(pred, head_name=head_name)

            y_full = y
            if y.ndim == 2:
                if head_index is None:
                    head_index = y.size(1) - 1
                y = y[:, head_index]

            if num_classes is None:
                num_classes = logits.size(1)

            if parent_index is not None and parent_to_children is not None:
                if y_full.ndim != 2:
                    raise ValueError("parent_index 需要二维标签输入")
                parent_labels = y_full[:, parent_index]
                logits, valid_parent = mask_logits_by_parent(
                    logits, parent_labels, parent_to_children
                )
            else:
                valid_parent = torch.ones_like(y, dtype=torch.bool)

            valid = (y >= 0) & valid_parent
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

def split_by_lowest_level_ratio(
    dataset,
    lowest_level: str = "leaf",
    train_ratio: float = 0.8,
    seed: int = 42,
    min_train_samples: int = 1,
):
    """
    按指定层级进行分组切分
    返回:
    - train_indices
    - test_indices
    """
    rng = np.random.RandomState(seed)

    # level_key -> [sample indices]
    group_to_indices = defaultdict(list)

    for i in range(len(dataset)):
        if "/" in str(lowest_level):
            key = dataset.get_split_key(i, lowest_level)
        else:
            key = dataset.get_level_key(i, lowest_level)
        if key is None:
            key = dataset.get_level_key(i, "leaf")
        group_to_indices[key].append(i)

    train_idx = []
    test_idx = []

    for _, indices in group_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            # 该组只有 1 个样本，直接放入训练集
            train_idx.append(indices[0])
            continue

        n_train = int(len(indices) * train_ratio)
        n_train = max(min_train_samples, n_train)
        n_train = min(n_train, len(indices) - 1)

        train_idx.extend(indices[:n_train])
        test_idx.extend(indices[n_train:])

    return train_idx, test_idx

def get_linear_weight(epoch, start, end, w_min, w_max):
    """
    Linearly increase loss weight from w_min to w_max
    between epoch [start, end]
    """
    if epoch < start:
        return w_min
    elif epoch > end:
        return w_max
    else:
        ratio = (epoch - start) / (end - start)
        return w_min + ratio * (w_max - w_min)

# FocalLoss
# 对于当前任务logits 是否把样本分到正确类
class FocalLoss(nn.Module):
    def __init__(self, gamma, weight=None, ignore_index=-1, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index
        )
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            if not valid.any():
                return torch.tensor(0.0, device=logits.device)
            ce_loss = ce_loss[valid]
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss

def hierarchical_center_loss(
    feat,
    hier_labels: dict,
    level_weights: dict,
):
    loss = 0.0
    count = 0

    for level, weight in level_weights.items():
        if level not in hier_labels:
            continue

        labels = hier_labels[level]
        valid = labels >= 0
        if not valid.any():
            continue

        labels = labels[valid]
        feat_valid = feat[valid]

        # 各层级分别计算中心损失
        for c in labels.unique():
            fc = feat_valid[labels == c]
            # 样本太少不计算中心
            if fc.size(0) <= 1:
                continue

            center = fc.mean(dim=0, keepdim=True)
            diff = fc - center
            radial = (diff * diff).sum(dim=1)
            loss += weight * radial.mean()
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=feat.device)

    return loss / count

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (single-view version)
    设计目的：
    - 直接在 embedding 空间上约束几何结构
    - 同类样本相对更近
    - 不同类样本相对更远
    - 不要求类内单中心（允许多模态 / 多子簇）
    输入：
    - feat   : Tensor [B, D]，模型输出的 embedding
    - labels : Tensor [B]，监督标签（类别 id）

    说明：
    - 使用 cosine similarity（先做 L2 normalize）
    - temperature 控制对比“硬度”
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = float(temperature)

    def forward(self, feat, labels):
        """
        feat   : [B, D]
        labels : [B]
        """
        device = feat.device
        B = feat.size(0)

        # batch 太小，或没有正样本时，直接返回 0
        if B <= 1:
            return torch.tensor(0.0, device=device)

        # ------------------------------------------------------------
        # 1. 嵌入向量在进行对比比较之前会 L2 normalize（只比较角度，不约束幅值）
        # ------------------------------------------------------------
        z = F.normalize(feat, p=2, dim=1)  # [B, D]

        # ------------------------------------------------------------
        # 2. 计算两两相似度矩阵 (cosine / tau)
        # ------------------------------------------------------------
        logits = torch.matmul(z, z.t()) / self.tau  # [B, B]

        # ------------------------------------------------------------
        # 3. 构造 mask
        #    - 去掉 i == j
        #    - 正样本：labels 相同
        # ------------------------------------------------------------
        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        labels = labels.view(-1, 1)  # [B, 1]
        pos_mask = (labels == labels.t()) & logits_mask  # [B, B]

        # 如果 batch 内没有任何正样本（极端情况），直接跳过
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        # ------------------------------------------------------------
        # 4. 数值稳定：每行减去最大值
        # ------------------------------------------------------------
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # ------------------------------------------------------------
        # 5. 计算 log-prob
        # ------------------------------------------------------------
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # ------------------------------------------------------------
        # 6. 对每个 anchor，平均所有正样本
        # ------------------------------------------------------------
        pos_count = pos_mask.sum(dim=1)  # [B]
        valid = pos_count > 0

        mean_log_prob_pos = (
            (log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-12)
        )

        # ------------------------------------------------------------
        # 7. 取负号，做最小化
        # ------------------------------------------------------------
        loss = -mean_log_prob_pos[valid].mean()

        return loss

class AutoHierarchicalBatchSampler(Sampler):
    """
    自动层级采样器
    - batch 优先覆盖 top_level
    - 每个 leaf 至少采样 min_samples_per_leaf
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        top_level: str = "level_1",
        leaf_level: str = "leaf",
        min_samples_per_leaf: int = 2,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.min_k = int(min_samples_per_leaf)
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        # --------------------------------------------------
        # 处理 Subset
        # --------------------------------------------------
        if isinstance(dataset, Subset):
            self.base_dataset = dataset.dataset
            self.subset_indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            self.subset_indices = None

        # 读取层级映射
        self.hier_names = self.base_dataset.hier_names
        self.label_maps = {
            name: self.base_dataset.label_maps_by_level[i]
            for i, name in enumerate(self.base_dataset.head_names)
        }

        self.top_level = self._resolve_level_name(top_level)
        self.leaf_level = self._resolve_level_name(leaf_level)

        # top -> leaf -> indices
        self.top_to_leaf = self._build_index()

        self.num_samples = len(self.dataset)
        self._last_batch = None

    # --------------------------------------------------
    # index 相关
    # --------------------------------------------------
    def _resolve_level_name(self, level_name):
        if hasattr(self.base_dataset, "_resolve_level_name"):
            return self.base_dataset._resolve_level_name(level_name)
        return level_name

    def _to_base_index(self, local_idx):
        if self.subset_indices is None:
            return local_idx
        return self.subset_indices[local_idx]

    def _build_index(self):
        """
        构建索引
        top_id -> leaf_id -> [local_indices]
        """
        index = defaultdict(lambda: defaultdict(list))

        for local_idx in range(len(self.dataset)):
            base_idx = self._to_base_index(local_idx)
            hier = self.hier_names[base_idx]

            top_name = hier.get(self.top_level)
            leaf_name = hier.get(self.leaf_level)

            if top_name is None or leaf_name is None:
                continue

            top_id = self.label_maps[self.top_level][top_name]
            leaf_id = self.label_maps[self.leaf_level][leaf_name]

            index[top_id][leaf_id].append(local_idx)

        return index

    # --------------------------------------------------
    # Sampler 核心
    # --------------------------------------------------
    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for t in self.top_to_leaf.values():
            for s in t.values():
                self.rng.shuffle(s)

        batches = []

        while len(batches) * self.batch_size < self.num_samples:
            batch = self._sample_one_batch()
            if len(batch) < self.batch_size:
                break
            batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        for b in batches:
            self._last_batch = b
            yield b

    # --------------------------------------------------
    # 采样一个 batch
    # --------------------------------------------------
    def _sample_one_batch(self):
        batch = []

        top_ids_all = list(self.top_to_leaf.keys())
        self.rng.shuffle(top_ids_all)

        # ---------
        # 选择若干 top-level
        # ---------
        T = min(len(top_ids_all), max(2, self.batch_size // 8))
        top_ids = top_ids_all[:T]

        per_top_budget = self.batch_size // max(1, T)

        for t_id in top_ids:
            if len(batch) >= self.batch_size:
                break

            leaf_dict = self.top_to_leaf[t_id]
            leaf_ids = list(leaf_dict.keys())
            self.rng.shuffle(leaf_ids)

            # ---------
            # 选择若干 leaf
            # ---------
            S = min(
                len(leaf_ids),
                max(1, per_top_budget // max(1, self.min_k))
            )
            leaf_ids = leaf_ids[:S]

            for leaf_id in leaf_ids:
                if len(batch) >= self.batch_size:
                    break

                indices = leaf_dict[leaf_id]
                if not indices:
                    continue

                remaining = self.batch_size - len(batch)

                # ---------
                # 每个 leaf 选取样本数
                # ---------
                k = min(
                    len(indices),
                    max(self.min_k, remaining // max(1, (T * S)))
                )

                picked = self.rng.sample(indices, k)
                batch.extend(picked)

        # 保证 leaf 至少有 2 个样本的比例
        if len(batch) < self.batch_size:
            remain = self.batch_size - len(batch)

            all_indices = []
            candidates = []

            for t in self.top_to_leaf.values():
                for idxs in t.values():
                    all_indices.extend(idxs)
                    if len(idxs) >= 2:
                        candidates.extend(idxs)

            if len(candidates) >= remain:
                batch.extend(self.rng.sample(candidates, remain))
            else:
                batch.extend(self.rng.sample(all_indices, remain))

        return batch

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------
    def debug_print_batch(self, batch_indices, prefix=""):
        if batch_indices is None:
            print(f"{prefix}[debug] batch is None")
            return

        print(f"\n{prefix}Batch structure debug:")

        for key in [self.top_level, self.leaf_level]:
            labels = []
            for local_idx in batch_indices:
                base_idx = self._to_base_index(local_idx)
                name = self.hier_names[base_idx][key]
                label = self.label_maps[key][name]
                labels.append(label)

            counter = Counter(labels)
            print(
                f"  {key:<8}: "
                f"unique={len(counter):<3} | "
                f"counts={dict(counter)}"
            )

        print(f"  samples : {len(batch_indices)}")

class HierarchicalBatchEpochStats:
    """
    统计 sampler 在一个 epoch 内的 batch 结构
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.reset()

    def reset(self):
        self.num_batches = 0
        self.top_counts = []
        self.leaf_counts = []
        self.leaf_ge2_ratios = []

    def update(self):
        """
        更新一个 batch 的统计量
        """
        batch = self.sampler._last_batch
        if batch is None:
            return

        top_labels = []
        leaf_labels = []

        top = self.sampler.top_level
        leaf = self.sampler.leaf_level

        for local_idx in batch:
            base_idx = self.sampler._to_base_index(local_idx)
            hier = self.sampler.hier_names[base_idx]

            top_labels.append(
                self.sampler.label_maps[top][hier[top]]
            )
            leaf_labels.append(
                self.sampler.label_maps[leaf][hier[leaf]]
            )

        top_counter = Counter(top_labels)
        leaf_counter = Counter(leaf_labels)

        self.top_counts.append(len(top_counter))
        self.leaf_counts.append(len(leaf_counter))

        # leaf >= 2 的比例
        ge2 = sum(v >= 2 for v in leaf_counter.values())
        ratio = ge2 / max(1, len(leaf_counter))
        self.leaf_ge2_ratios.append(ratio)

        self.num_batches += 1

    def summary(self):
        """
        输出 epoch 统计
        """
        if self.num_batches == 0:
            return {}

        return {
            f"avg_{self.sampler.top_level}_per_batch":
                sum(self.top_counts) / self.num_batches,
            f"avg_{self.sampler.leaf_level}_per_batch":
                sum(self.leaf_counts) / self.num_batches,
            f"avg_{self.sampler.leaf_level}_ge2_ratio":
                sum(self.leaf_ge2_ratios) / self.num_batches,
        }

