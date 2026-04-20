import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_class_weights(level_labels, num_classes):
    """
    根据当前训练层的标签分布构造类别权重
    这里使用对数平滑，避免极少数类权重过大导致训练不稳定
    """
    valid_mask = level_labels >= 0
    if not valid_mask.any():
        return np.ones(num_classes, dtype=np.float32)

    counts = np.bincount(level_labels[valid_mask], minlength=num_classes)
    counts = np.maximum(counts, 1)
    base_class_weights = 1.0 / np.log(counts + 1.5)
    base_class_weights = base_class_weights / base_class_weights.mean()
    return base_class_weights.astype(np.float32)


def get_linear_weight(epoch, start, end, w_min, w_max):
    """
    在指定 epoch 区间内对损失权重做线性拉升
    """
    if epoch < start:
        return w_min
    if epoch > end:
        return w_max

    ratio = (epoch - start) / (end - start)
    return w_min + ratio * (w_max - w_min)


class FocalLoss(nn.Module):
    """
    Focal Loss
    在交叉熵基础上降低易分类样本的权重，突出难样本
    """

    def __init__(self, gamma, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=None,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not valid_mask.any():
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            targets = targets[valid_mask]
            ce_loss = ce_loss[valid_mask]

        pt = torch.exp(-ce_loss)
        focal_factor = (1 - pt) ** self.gamma

        if self.weight is not None:
            sample_weight = self.weight[targets]
            loss = sample_weight * focal_factor * ce_loss
        else:
            loss = focal_factor * ce_loss

        return loss

def AlignLoss(feat, y):
    """
    仅按当前训练层标签计算 batch 内类内紧凑损失
    """
    valid_mask = y >= 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=feat.device)

    y_valid = y[valid_mask]
    feat_valid = feat[valid_mask]

    loss_sum = 0.0
    valid_group_count = 0
    for c in y_valid.unique():
        feat_c = feat_valid[y_valid == c]
        if feat_c.size(0) <= 1:
            continue

        center_c = feat_c.mean(dim=0, keepdim=True)
        diff_c = feat_c - center_c
        radial_c = (diff_c * diff_c).sum(dim=1)
        loss_sum += radial_c.mean()
        valid_group_count += 1

    if valid_group_count == 0:
        return torch.tensor(0.0, device=feat.device)

    return loss_sum / valid_group_count


class SupConLoss(nn.Module):
    """
    单视角版本的监督式对比损失

    目标是让同类样本在 embedding 空间更接近，不同类样本更分离，
    但不强制每个类别只有一个中心
    """

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = float(tau)

    def forward(self, feat, y):
        """
        参数：
        - `feat`: `[B, D]` 的 embedding
        - `y`: `[B]` 的监督标签
        """
        device = feat.device
        batch_size = feat.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        z = F.normalize(feat, p=2, dim=1)
        sim = torch.matmul(z, z.t()) / self.tau

        off_diag_mask = torch.ones_like(sim, dtype=torch.bool)
        off_diag_mask.fill_diagonal_(False)

        y = y.view(-1, 1)
        pos_mask = (y == y.t()) & off_diag_mask # 标出正样本位置
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)
        # 减去最大值是为了数值稳定性
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * off_diag_mask.float()
        
        # 计算有效样本数
        pos_count = pos_mask.sum(dim=1)
        # 有效类
        valid_anchor = pos_count > 0

        log_q = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        valid_anchor = pos_count > 0
        mean_log_q_pos = (
            (log_q * pos_mask.float()).sum(dim=1) / (pos_count + 1e-12)
        )

        return -mean_log_q_pos[valid_anchor].mean()
