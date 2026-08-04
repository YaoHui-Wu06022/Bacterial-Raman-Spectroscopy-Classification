"""一维拉曼分类模型组装。"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from ramanv2.modeling.heads import CosineClassifier
from ramanv2.modeling.layers import PositionalEncoding1D, ResidualBottleneck1D, build_activation, build_conv_block
from ramanv2.modeling.spec import ModelSpec


class RamanClassifier1D(nn.Module):
    """按主干、序列编码、池化与分类头组装的一维拉曼分类器。"""

    def __init__(self, num_classes: int, model_spec: ModelSpec):
        super().__init__()
        if isinstance(num_classes, (dict, list, tuple)):
            raise ValueError("num_classes must be a single int.")
        self.num_classes = int(num_classes)
        self.parse_core_config(model_spec)
        self.build_backbone()
        self.build_sequence_encoder()
        self.build_pooling()
        self.build_classifier_head()
    def parse_core_config(self, model_spec: ModelSpec) -> None:
        """把输入配置转换为模型构建所需的稳定内部字段。"""
        self.cnn_backbone_enable = model_spec.backbone_type == "cnn"
        self.transformer_enable = model_spec.encoder_type == "transformer"
        self.lstm_enable = model_spec.encoder_type == "lstm"
        self.in_channels = model_spec.in_channels
        self.proj_dim = model_spec.transformer_dim
        self.pooling_type = model_spec.pooling_type
        self.cosine_head_enable = model_spec.cosine_head_enable
        self._stem_kernel_sizes = model_spec.stem_kernel_sizes
        self._make_backbone_activation = partial(build_activation, negative_slope=model_spec.activation_negative_slope)
        self._block_kwargs = {
            "block_type": model_spec.cnn_block_type,
            "cardinality": model_spec.cardinality,
            "base_width": model_spec.base_width,
            "bottleneck_ratio": model_spec.resnet_bottleneck_ratio,
            "reduction": model_spec.reduction,
            "se_enable": model_spec.se_enable,
            "activation_negative_slope": model_spec.activation_negative_slope,
        }
        self._encoder_spec = model_spec

    def build_backbone(self) -> None:
        """按规格创建 CNN 主干或完整谱轴投影路径。"""
        if self.cnn_backbone_enable:
            self.build_cnn_backbone()
            return
        self.input_proj = nn.Sequential(nn.Conv1d(self.in_channels, self.proj_dim, 1, bias=False), nn.BatchNorm1d(self.proj_dim), nn.GELU())

    def split_channels(self, total_channels: int, branch_count: int) -> list[int]:
        """将 stem 通道尽量均匀分配给多个卷积分支。"""
        base, remainder = divmod(total_channels, branch_count)
        return [base + (index < remainder) for index in range(branch_count)]

    def build_cnn_backbone(self) -> None:
        """创建多尺度 stem、四个残差 stage 和统一投影层。"""
        kernel_sizes = self._stem_kernel_sizes
        self.stem_branches = nn.ModuleList(
            [
                build_conv_block(
                    self.in_channels,
                    channels,
                    kernel_size,
                    self._make_backbone_activation,
                )
                for kernel_size, channels in zip(
                    kernel_sizes,
                    self.split_channels(64, len(kernel_sizes)),
                )
            ]
        )
        channels = 64
        self.layer1, channels = self.build_stage(channels, 64, 2, False)
        self.layer2, channels = self.build_stage(channels, 128, 2, True)
        self.layer3, channels = self.build_stage(channels, 256, 2, True)
        self.layer4, channels = self.build_stage(channels, 512, 2, False)
        self.proj = nn.Conv1d(channels, self.proj_dim, 1, bias=False)

    def build_stage(self, in_channels: int, out_channels: int, block_count: int, pool_first_enable: bool) -> tuple[nn.Sequential, int]:
        """构建一个残差 stage，并显式返回其输出通道数。"""
        modules = [nn.AvgPool1d(2)] if pool_first_enable else []
        layer, output_channels = self.build_layer(in_channels, out_channels, block_count)
        modules.append(layer)
        return nn.Sequential(*modules), output_channels

    def build_layer(self, in_channels: int, out_channels: int, block_count: int) -> tuple[nn.Sequential, int]:
        """堆叠指定数量的同输出通道残差瓶颈块。"""
        layers = [ResidualBottleneck1D(in_channels, out_channels, **self._block_kwargs)]
        layers.extend(
            ResidualBottleneck1D(out_channels, out_channels, **self._block_kwargs)
            for _ in range(1, block_count)
        )
        return nn.Sequential(*layers), out_channels

    def build_sequence_encoder(self) -> None:
        """创建 Transformer、LSTM 或空序列编码器。"""
        self.seq_dim = self.proj_dim
        if self.transformer_enable:
            self.pos_encoder = PositionalEncoding1D(self.proj_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self._encoder_spec.transformer_nhead,
                dim_feedforward=self._encoder_spec.transformer_ffn_dim,
                dropout=self._encoder_spec.transformer_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=self._encoder_spec.transformer_layers)
            self.lstm = None
        else:
            self.pos_encoder = None
            self.transformer = None
            if self.lstm_enable:
                hidden = self._encoder_spec.lstm_hidden
                layers = self._encoder_spec.lstm_layers
                bidirectional_enable = self._encoder_spec.lstm_bidirectional_enable
                self.lstm = nn.LSTM(
                    self.proj_dim,
                    hidden,
                    layers,
                    dropout=self._encoder_spec.lstm_dropout if layers > 1 else 0.0,
                    bidirectional=bidirectional_enable,
                    batch_first=True,
                )
                self.seq_dim = hidden * (2 if bidirectional_enable else 1)
            else:
                self.lstm = None

    def build_pooling(self) -> None:
        """创建注意力或统计池化，并设置最终特征维度。"""
        if self.pooling_type == "attn":
            self.att_pool = nn.Sequential(
                nn.Linear(self.seq_dim, self.seq_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.seq_dim // 2, 1),
            )
            self.feat_dim = self.seq_dim
        elif self.pooling_type == "stat":
            self.att_pool = None
            self.feat_dim = self.seq_dim * 2
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def build_classifier_head(self) -> None:
        """创建余弦或线性分类头。"""
        self.head = (
            CosineClassifier(
                self.feat_dim,
                self.num_classes,
                self._encoder_spec.cosine_scale,
            )
            if self.cosine_head_enable
            else nn.Linear(self.feat_dim, self.num_classes)
        )
        del self._stem_kernel_sizes, self._block_kwargs, self._encoder_spec, self._make_backbone_activation

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        """将输入谱转换为主干序列特征。"""
        if self.cnn_backbone_enable:
            values = torch.cat([branch(values) for branch in self.stem_branches], dim=1)
            return self.proj(self.layer4(self.layer3(self.layer2(self.layer1(values)))))
        return self.input_proj(values)

    def forward_sequence(self, values: torch.Tensor) -> torch.Tensor:
        """执行可选的序列编码。"""
        if self.transformer_enable:
            return self.transformer(self.pos_encoder(values))
        if self.lstm_enable:
            return self.lstm(values)[0]
        return values

    def pool_features(self, values: torch.Tensor) -> torch.Tensor:
        """将序列特征聚合为样本级 embedding。"""
        if self.pooling_type == "attn":
            return (values * torch.softmax(self.att_pool(values), dim=1)).sum(dim=1)
        return torch.cat([values.mean(dim=1), values.std(dim=1, unbiased=False)], dim=1)

    def forward(self, values: torch.Tensor, return_embedding_enable: bool = False):
        """执行特征提取、序列编码、池化和分类。"""
        features = self.pool_features(self.forward_sequence(self.forward_features(values).permute(0, 2, 1)))
        logits = self.head(features)
        return (logits, features) if return_embedding_enable else logits
