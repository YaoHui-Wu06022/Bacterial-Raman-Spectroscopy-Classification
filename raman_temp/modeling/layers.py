"""一维拉曼分类网络的可复用基础层。"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn


def build_activation(inplace: bool = True, negative_slope: float = 0.01) -> nn.LeakyReLU:
    """构建主干统一使用的 LeakyReLU 激活层。"""
    return nn.LeakyReLU(negative_slope=float(negative_slope), inplace=inplace)


def build_conv_block(in_channels: int, out_channels: int, kernel_size: int, make_activation=None, *, stride: int = 1, padding: int | None = None, groups: int = 1, inplace: bool = True) -> nn.Sequential:
    """构建 Conv1d、BatchNorm 和可选激活组成的基础块。"""
    if padding is None:
        padding = kernel_size // 2
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm1d(out_channels),
    ]
    if make_activation is not None:
        layers.append(make_activation(inplace=inplace))
    return nn.Sequential(*layers)


class SEBlock1D(nn.Module):
    """一维 SE 通道重标定模块。"""

    def __init__(self, channels: int, reduction: int, se_enable: bool, make_activation):
        super().__init__()
        self.se_enable = bool(se_enable)
        hidden_channels = max(int(channels // reduction), 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            make_activation(inplace=False),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def build_scale(self, values: torch.Tensor) -> torch.Tensor:
        """根据当前 batch 的通道响应生成缩放系数。"""
        batch_size, channels, _ = values.size()
        return self.fc(self.pool(values).view(batch_size, channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.se_enable:
            return values
        batch_size, channels, length = values.size()
        scale = self.build_scale(values).unsqueeze(-1).expand(batch_size, channels, length)
        return values * scale


def resolve_mid_channels(out_channels: int, block_type: str, cardinality: int | None = None, base_width: int | None = None, bottleneck_ratio: int | None = None) -> int:
    """按 ResNet 或 ResNeXt 配置计算瓶颈层中间通道数。"""
    normalized = str(block_type).lower()
    if normalized == "resnext":
        return max(int(out_channels * (base_width / 64.0)) * cardinality, int(cardinality))
    if normalized == "resnet":
        return max(int(out_channels // bottleneck_ratio), 1)
    raise ValueError(f"Unknown cnn_block_type: {block_type}")


class ResidualBottleneck1D(nn.Module):
    """支持 ResNet 与 ResNeXt 的一维残差瓶颈块。"""

    def __init__(self, in_channels: int, out_channels: int, block_type: str = "resnext", cardinality: int | None = None, base_width: int | None = None, bottleneck_ratio: int = 4, reduction: int | None = None, se_enable: bool = True, activation_negative_slope: float = 0.01):
        super().__init__()
        normalized = str(block_type).lower()
        make_activation = partial(build_activation, negative_slope=activation_negative_slope)
        groups = 1 if normalized == "resnet" else int(cardinality)
        mid_channels = resolve_mid_channels(out_channels, normalized, cardinality, base_width, bottleneck_ratio)
        self.conv_reduce = build_conv_block(in_channels, mid_channels, 1, make_activation)
        self.conv_mid = build_conv_block(mid_channels, mid_channels, 3, make_activation, groups=groups)
        self.conv_expand = build_conv_block(mid_channels, out_channels, 1)
        self.se = SEBlock1D(out_channels, reduction, se_enable, make_activation)
        self.shortcut = build_conv_block(in_channels, out_channels, 1, padding=0) if in_channels != out_channels else nn.Identity()
        self.out_act = make_activation(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(values)
        values = self.conv_reduce(values)
        values = self.conv_mid(values)
        values = self.se(self.conv_expand(values))
        return self.out_act(values + residual)


class PositionalEncoding1D(nn.Module):
    """给一维序列特征加入固定正余弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.pe[:, :values.size(1), :]
