"""分类头定义。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CosineClassifier(nn.Module):
    """使用特征和类别向量方向相似度的分类头。"""

    def __init__(self, in_features: int, out_features: int, scale: float = 30.0):
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = functional.normalize(values, p=2, dim=1)
        weights = functional.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(features, weights.t())
