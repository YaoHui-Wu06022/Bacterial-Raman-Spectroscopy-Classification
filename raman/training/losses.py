import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_class_weights(level_labels, num_classes):
    """
    根据当前训练层的标签分布构造类别权重。
    这里使用对数平滑，避免极少数类权重过大导致训练不稳定。
    """
    valid = level_labels >= 0
    if not valid.any():
        return np.ones(num_classes, dtype=np.float32)

    counts = np.bincount(level_labels[valid], minlength=num_classes)
    counts = np.maximum(counts, 1)
    base_class_weights = 1.0 / np.log(counts + 1.5)
    base_class_weights = base_class_weights / base_class_weights.mean()
    return base_class_weights.astype(np.float32)


def get_linear_weight(epoch, start, end, w_min, w_max):
    """
    在指定 epoch 区间内对损失权重做线性拉升。
    """
    if epoch < start:
        return w_min
    if epoch > end:
        return w_max

    ratio = (epoch - start) / (end - start)
    return w_min + ratio * (w_max - w_min)


class FocalLoss(nn.Module):
    """
    Focal Loss。

    在交叉熵基础上降低易分类样本的权重，突出难样本。
    """

    def __init__(self, gamma, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=None,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            if not valid.any():
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            targets = targets[valid]
            ce_loss = ce_loss[valid]

        pt = torch.exp(-ce_loss)
        focal_factor = (1 - pt) ** self.gamma

        if self.weight is not None:
            alpha_t = self.weight[targets]
            loss = alpha_t * focal_factor * ce_loss
        else:
            loss = focal_factor * ce_loss

        return loss


def AlignLoss(feat, labels):
    """
    仅按当前训练层标签计算 batch 内类内紧凑损失。
    """
    valid = labels >= 0
    if not valid.any():
        return torch.tensor(0.0, device=feat.device)

    labels = labels[valid]
    feat_valid = feat[valid]

    loss = 0.0
    count = 0
    for class_id in labels.unique():
        feat_class = feat_valid[labels == class_id]
        if feat_class.size(0) <= 1:
            continue

        center = feat_class.mean(dim=0, keepdim=True)
        diff = feat_class - center
        radial = (diff * diff).sum(dim=1)
        loss += radial.mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=feat.device)

    return loss / count


class SupConLoss(nn.Module):
    """
    单视角版本的监督式对比损失。

    目标是让同类样本在 embedding 空间更接近，不同类样本更分离，
    但不强制每个类别只有一个中心。
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = float(temperature)

    def forward(self, feat, labels):
        """
        参数：
        - `feat`: `[B, D]` 的 embedding
        - `labels`: `[B]` 的监督标签
        """
        device = feat.device
        batch_size = feat.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        z = F.normalize(feat, p=2, dim=1)
        logits = torch.matmul(z, z.t()) / self.tau

        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & logits_mask
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        mean_log_prob_pos = (
            (log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-12)
        )

        return -mean_log_prob_pos[valid].mean()
