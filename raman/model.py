import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_activation(inplace=True, negative_slope=0.01):
    return nn.LeakyReLU(
        negative_slope=float(negative_slope),
        inplace=inplace,
    )


def make_conv_block(
    in_channels,
    out_channels,
    kernel_size,
    make_activation=None,
    *,
    stride=1,
    padding=None,
    groups=1,
    inplace=True,
):
    # 统一的 Conv1d + BN (+ Activation) 组装函数，避免 backbone 里重复写样板代码。
    if padding is None:
        padding = kernel_size // 2
    layers = [
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
    ]
    if make_activation is not None:
        layers.append(make_activation(inplace=inplace))
    return nn.Sequential(*layers)


class SEBlock1D(nn.Module):
    def __init__(
        self,
        channels,
        reduction,
        se_use,
        make_activation,
    ):
        super().__init__()
        self.se_use = bool(se_use)
        self.latest_scale = None

        hidden_channels = max(int(channels // reduction), 1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            make_activation(inplace=False),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not self.se_use:
            return x

        # 保存当前 batch 的通道权重，供后续分析模块读取。
        batch_size, channels, length = x.size()
        scale = self.pool(x).view(batch_size, channels)
        scale = self.fc(scale)
        self.latest_scale = scale
        scale = scale.unsqueeze(-1).expand(batch_size, channels, length)
        return x * scale


def resolve_mid_channels(
    out_channels,
    block_type,
    cardinality=None,
    base_width=None,
    bottleneck_ratio=None,
):
    block_type = str(block_type).lower()
    if block_type == "resnext":
        mid_channels = int(out_channels * (base_width / 64.0)) * cardinality
        return max(mid_channels, int(cardinality))
    if block_type == "resnet":
        return max(int(out_channels // bottleneck_ratio), 1)
    raise ValueError(f"Unknown cnn_block_type: {block_type}")


class ResidualBottleneck1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_type="resnext",
        cardinality=None,
        base_width=None,
        bottleneck_ratio=4,
        reduction=None,
        se_use=True,
        activation_negative_slope=0.01,
    ):
        super().__init__()
        self.block_type = str(block_type).lower()
        make_activation = partial(
            build_activation,
            negative_slope=activation_negative_slope,
        )

        groups = 1 if self.block_type == "resnet" else int(cardinality)
        mid_channels = resolve_mid_channels(
            out_channels,
            block_type=self.block_type,
            cardinality=cardinality,
            base_width=base_width,
            bottleneck_ratio=bottleneck_ratio,
        )

        self.conv_reduce = make_conv_block(
            in_channels,
            mid_channels,
            kernel_size=1,
            make_activation=make_activation,
            padding=0,
        )
        self.conv_mid = make_conv_block(
            mid_channels,
            mid_channels,
            kernel_size=3,
            make_activation=make_activation,
            groups=groups,
        )
        self.conv_expand = make_conv_block(
            mid_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.se = SEBlock1D(
            out_channels,
            reduction=reduction,
            se_use=se_use,
            make_activation=make_activation,
        )
        if in_channels != out_channels:
            self.shortcut = make_conv_block(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()
        self.out_act = make_activation(inplace=True)

    def forward(self, x):
        # bottleneck 主支路 + shortcut 残差支路。
        identity = self.shortcut(x)
        out = self.conv_reduce(x)
        out = self.conv_mid(out)
        out = self.conv_expand(out)
        out = self.se(out)
        return self.out_act(out + identity)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        length = x.size(1)
        return x + self.pe[:, :length, :]


class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 余弦头只比较方向相似度，不让特征模长直接主导分类分数。
        x = F.normalize(x, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(x, weight.t())


class RamanClassifier1D(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        if isinstance(num_classes, (dict, list, tuple)):
            raise ValueError("num_classes must be a single int.")

        self.config = config
        self.num_classes = int(num_classes)

        self._parse_core_config()
        self._build_backbone()
        self._build_sequence_encoder()
        self._build_pooling()
        self._build_classifier_head()

    def _cfg(self, name, default=None):
        return getattr(self.config, name, default)

    def _parse_core_config(self):
        # 先把结构性配置解析成布尔开关和缓存字段，后面子模块构建只依赖这些状态。
        self.backbone_type = str(self._cfg("backbone_type", "cnn")).lower()
        if self.backbone_type not in ("cnn", "identity"):
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        self.cnn_block_type = str(self._cfg("cnn_block_type", "resnext")).lower()
        if self.cnn_block_type not in ("resnet", "resnext"):
            raise ValueError(f"Unknown cnn_block_type: {self.cnn_block_type}")

        self.encoder_type = str(self._cfg("encoder_type", "transformer")).lower()
        if self.encoder_type not in ("transformer", "lstm", "none"):
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.cnn_backbone_on = self.backbone_type == "cnn"
        self.transformer_on = self.encoder_type == "transformer"
        self.lstm_on = self.encoder_type == "lstm"

        self.in_channels = int(self.config.in_channels)
        self.proj_dim = int(self._cfg("transformer_dim", 192))
        self.reduction = int(self.config.reduction)
        self.se_use = bool(self.config.se_use)

        self.backbone_activation_negative_slope = float(
            self._cfg("backbone_activation_negative_slope", 0.01)
        )
        self.make_backbone_activation = partial(
            build_activation,
            negative_slope=self.backbone_activation_negative_slope,
        )

        self.block_kwargs = {
            "block_type": self.cnn_block_type,
            "cardinality": self.config.cardinality,
            "base_width": self.config.base_width,
            "bottleneck_ratio": int(self._cfg("resnet_bottleneck_ratio", 4)),
            "reduction": self.reduction,
            "se_use": self.se_use,
            "activation_negative_slope": self.backbone_activation_negative_slope,
        }

    def _build_backbone(self):
        if self.cnn_backbone_on:
            self._build_cnn_backbone()
            return
        self._build_identity_backbone()

    def _build_identity_backbone(self):
        pool_kernel = max(1, int(self._cfg("identity_pool_kernel", 16)))
        self.identity_pool_kernel = pool_kernel
        self.identity_pool = (
            nn.Identity()
            if pool_kernel == 1
            else nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        )
        # identity 路径不做 CNN 提特征，只做下采样后投影到统一序列维度。
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_channels, self.proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.proj_dim),
            nn.GELU(),
        )

    def _build_cnn_backbone(self):
        self.stem_out_channels = 64
        self.in_planes = self.stem_out_channels
        kernel_sizes = self._cfg("stem_kernel_sizes", None) or (15,)
        if isinstance(kernel_sizes, int):
            kernel_sizes = (kernel_sizes,)
        kernel_sizes = [int(k) for k in kernel_sizes]
        # 单尺度和多尺度共用同一套逻辑；只传一个 kernel 时自然退化为单分支 stem。
        self.stem_branches = nn.ModuleList(
            [
                make_conv_block(
                    self.in_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    make_activation=self.make_backbone_activation,
                )
                for kernel_size, branch_channels in zip(
                    kernel_sizes,
                    self._split_channels(self.stem_out_channels, len(kernel_sizes)),
                )
            ]
        )
        self.stem_pool = nn.AvgPool1d(kernel_size=2)

        self.layer1 = self._make_stage(64, num_blocks=2, pool_first=False)
        self.layer2 = self._make_stage(128, num_blocks=2, pool_first=True)
        self.layer3 = self._make_stage(256, num_blocks=2, pool_first=True)
        self.layer4 = self._make_stage(384, num_blocks=2, pool_first=True)
        self.proj = nn.Conv1d(384, self.proj_dim, kernel_size=1, bias=False)

    def _split_channels(self, total_channels, num_branches):
        base = total_channels // num_branches
        remainder = total_channels - base * num_branches
        branch_channels = [base] * num_branches
        for idx in range(remainder):
            branch_channels[idx] += 1
        return branch_channels

    def _make_stage(self, out_channels, num_blocks, pool_first):
        modules = []
        if pool_first:
            modules.append(nn.AvgPool1d(kernel_size=2))
        modules.append(self._make_layer(out_channels, num_blocks))
        return nn.Sequential(*modules)

    def _make_layer(self, out_channels, num_blocks):
        layers = [
            ResidualBottleneck1D(
                in_channels=self.in_planes,
                out_channels=out_channels,
                **self.block_kwargs,
            )
        ]
        self.in_planes = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBottleneck1D(
                    in_channels=self.in_planes,
                    out_channels=out_channels,
                    **self.block_kwargs,
                )
            )
        return nn.Sequential(*layers)

    def _build_sequence_encoder(self):
        self.seq_dim = self.proj_dim

        if self.transformer_on:
            # Transformer 接收 [B, L, C] 序列表示，在谱峰之间建上下文关系。
            self.pos_encoder = PositionalEncoding1D(d_model=self.proj_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self.config.transformer_nhead,
                dim_feedforward=self.config.transformer_ffn_dim,
                dropout=self.config.transformer_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.transformer_layers,
            )
            self.lstm = None
            return

        self.pos_encoder = None
        self.transformer = None
        if self.lstm_on:
            # LSTM 路径保留顺序建模能力，但不引入自注意力。
            self.lstm_hidden = int(self._cfg("lstm_hidden", self.proj_dim))
            self.lstm_layers = int(self._cfg("lstm_layers", 1))
            self.lstm_dropout = float(self._cfg("lstm_dropout", 0.0))
            self.lstm_bidirectional = bool(self._cfg("lstm_bidirectional", False))
            self.lstm = nn.LSTM(
                input_size=self.proj_dim,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                dropout=self.lstm_dropout if self.lstm_layers > 1 else 0.0,
                bidirectional=self.lstm_bidirectional,
                batch_first=True,
            )
            self.seq_dim = self.lstm_hidden * (2 if self.lstm_bidirectional else 1)
        else:
            self.lstm = None

    def _build_pooling(self):
        self.pooling_type = str(self._cfg("pooling_type", "attn")).lower()
        if self.pooling_type not in ("attn", "stat"):
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        if self.pooling_type == "attn":
            # 注意力池化学习“哪些波段位置更值得汇聚到最终表征”。
            att_pool_dropout = 0.2
            self.att_pool = nn.Sequential(
                nn.Linear(self.seq_dim, self.seq_dim // 2),
                nn.GELU(),
                nn.Dropout(att_pool_dropout),
                nn.Linear(self.seq_dim // 2, 1),
            )
            self.feat_dim = self.seq_dim
        else:
            self.att_pool = None
            # 统计池化保留全局均值和波动强度两部分信息。
            self.feat_dim = self.seq_dim * 2

    def _build_classifier_head(self):
        self.cosine_head = bool(self._cfg("cosine_head", False))
        self.cosine_scale = float(self._cfg("cosine_scale", 30.0))
        if self.cosine_head:
            self.head = CosineClassifier(
                self.feat_dim,
                self.num_classes,
                scale=self.cosine_scale,
            )
        else:
            self.head = nn.Linear(self.feat_dim, self.num_classes)

    def _forward_feature_extractor(self, x):
        if self.cnn_backbone_on:
            x = torch.cat([branch(x) for branch in self.stem_branches], dim=1)
            x = self.stem_pool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return self.proj(x)

        x = self.identity_pool(x)
        return self.input_proj(x)

    def _forward_sequence_encoder(self, x):
        if self.transformer_on:
            x = self.pos_encoder(x)
            return self.transformer(x)
        if self.lstm_on:
            x, _ = self.lstm(x)
            return x
        return x

    def _pool_features(self, x):
        if self.pooling_type == "attn":
            attn = torch.softmax(self.att_pool(x), dim=1)
            return (x * attn).sum(dim=1)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        return torch.cat([mean, std], dim=1)

    def forward(self, x, return_feat=False):
        # 数据流：局部特征提取 -> 序列建模 -> 池化聚合 -> 分类头。
        features = self._forward_feature_extractor(x)
        features = features.permute(0, 2, 1)
        features = self._forward_sequence_encoder(features)
        feat = self._pool_features(features)
        logits = self.head(feat)
        if return_feat:
            return logits, feat
        return logits
