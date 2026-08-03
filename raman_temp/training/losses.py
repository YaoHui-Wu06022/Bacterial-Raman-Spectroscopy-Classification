"""训练损失与类别权重的纯张量计算。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn
import torch.nn.functional


def build_class_weights(level_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """按当前训练层标签分布构建对数平滑的类别权重。"""
    valid_mask = level_labels >= 0
    if not valid_mask.any():
        return np.ones(num_classes, dtype=np.float32)

    counts = np.bincount(level_labels[valid_mask], minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.log(counts + 1.5)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_linear_weight(
    epoch: int,
    start: int,
    end: int,
    minimum: float,
    maximum: float,
) -> float:
    """在指定 epoch 区间内线性调整辅助损失权重。"""
    if epoch < start:
        return minimum
    if epoch > end:
        return maximum

    ratio = (epoch - start) / (end - start)
    return minimum + ratio * (maximum - minimum)


class FocalLoss(torch.nn.Module):
    """逐样本 Focal 分类损失，调用方负责对返回值进行聚合。"""

    def __init__(
        self,
        gamma: float,
        weight: torch.Tensor | None = None,
        ignore_index: int | None = -1,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """以 float32 计算分类损失，避免混合精度下高置信样本过早舍入。"""
        logits = logits.float()
        ce_loss = torch.nn.functional.cross_entropy(
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

        focal_factor = (1 - torch.exp(-ce_loss)) ** self.gamma
        if self.weight is not None:
            return self.weight[targets] * focal_factor * ce_loss
        return focal_factor * ce_loss


def compute_align_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算 batch 内同类 embedding 围绕类中心的平均离散度。"""
    valid_mask = labels >= 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=features.device)

    valid_labels = labels[valid_mask]
    valid_features = features[valid_mask]
    loss_sum = 0.0
    group_count = 0
    for label in valid_labels.unique():
        group_features = valid_features[valid_labels == label]
        if group_features.size(0) <= 1:
            continue
        centered = group_features - group_features.mean(dim=0, keepdim=True)
        loss_sum += (centered * centered).sum(dim=1).mean()
        group_count += 1

    if group_count == 0:
        return torch.tensor(0.0, device=features.device)
    return loss_sum / group_count


class SupConLoss(torch.nn.Module):
    """单视角监督式对比损失。"""

    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.tau = float(tau)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """以同类样本为正样本，计算有效 anchor 的平均对比损失。"""
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)

        normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        similarity = torch.matmul(normalized, normalized.t()) / self.tau
        off_diagonal_mask = torch.ones_like(similarity, dtype=torch.bool)
        off_diagonal_mask.fill_diagonal_(False)

        column_labels = labels.view(-1, 1)
        positive_mask = (column_labels == column_labels.t()) & off_diagonal_mask
        if not positive_mask.any():
            return torch.tensor(0.0, device=features.device)

        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        exp_similarity = torch.exp(similarity) * off_diagonal_mask.float()
        positive_count = positive_mask.sum(dim=1)
        valid_anchor_mask = positive_count > 0
        log_probability = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-12)
        positive_log_probability = (
            (log_probability * positive_mask.float()).sum(dim=1) / (positive_count + 1e-12)
        )
        return -positive_log_probability[valid_anchor_mask].mean()
