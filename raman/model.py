import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def build_activation(name, inplace=True, negative_slope=0.01):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=float(negative_slope), inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    raise ValueError(f"Unknown backbone activation: {name}")

class SEBlock1D(nn.Module):
    """
    SE 模块：
    - 支持开关
    - 自动记录 scale 用于重要性分析
    - 保证 backward hook 获取到的梯度有效
    """
    def __init__(
        self,
        channels,
        reduction,
        se_use,
        activation_name="relu",
        activation_negative_slope=0.01,
    ):
        super().__init__()

        self.se_use = se_use
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            build_activation(
                activation_name,
                inplace=False,
                negative_slope=activation_negative_slope,
            ),   # 避免破坏 Autograd 需要的中间值
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        self.latest_scale = None  # ← 记录当前 batch 的 scale

    def forward(self, x):

        if not self.se_use:
            return x   #  SE 开关关闭时直接旁路

        b, c, l = x.size()

        # 通道统计
        y = self.pool(x).view(b, c)

        # 计算 SE 权重（scale）
        y = self.fc(y)  # [B, C]
        self.latest_scale = y   # 保存 scale，供重要性分析用

        # 扩展到序列维度
        y = y.unsqueeze(-1).expand(b, c, l)

        # 注意：这里不要 clone，否则 backward hook 获取不到梯度
        return x * y

# ResNeXt 残差块与 SE 模块
def resolve_mid_channels(
    out_channels,
    block_type,
    cardinality=None,
    base_width=None,
    bottleneck_ratio=None,
):
    """按 block 类型计算 bottleneck 中间通道数。"""
    block_type = str(block_type).lower()
    if block_type == "resnext":
        mid_channels = int(out_channels * (base_width / 64.0)) * cardinality
        return max(mid_channels, cardinality)
    if block_type == "resnet":
        return max(int(out_channels // bottleneck_ratio), 1)
    raise ValueError(f"Unknown cnn_block_type: {block_type}")


class ResidualBottleneck1D(nn.Module):
    """
    统一的 1D bottleneck 残差块。

    - `resnet` 使用普通 3x3 卷积
    - `resnext` 使用 group conv，并按 cardinality/base_width 控制宽度
    """

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
        activation_name="relu",
        activation_negative_slope=0.01,
    ):
        super().__init__()
        self.block_type = str(block_type).lower()
        groups = 1 if self.block_type == "resnet" else int(cardinality)
        mid_channels = resolve_mid_channels(
            out_channels,
            block_type=self.block_type,
            cardinality=cardinality,
            base_width=base_width,
            bottleneck_ratio=bottleneck_ratio,
        )

        self.conv_reduce = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            build_activation(
                activation_name,
                inplace=True,
                negative_slope=activation_negative_slope,
            )
        )

        self.conv_group = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm1d(mid_channels),
            build_activation(
                activation_name,
                inplace=True,
                negative_slope=activation_negative_slope,
            )
        )

        self.conv_expand = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.se = SEBlock1D(
            out_channels,
            reduction=reduction,
            se_use=se_use,
            activation_name=activation_name,
            activation_negative_slope=activation_negative_slope,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.out_act = build_activation(
            activation_name,
            inplace=True,
            negative_slope=activation_negative_slope,
        )

    def forward(self, x):
        identity = x
        out = self.conv_reduce(x)
        out = self.conv_group(out)
        out = self.conv_expand(out)
        out = self.se(out)
        out = out + self.shortcut(identity)
        out = self.out_act(out)
        return out

# 一维位置编码
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1000):  # 位置编码最长1000
        super().__init__()
        # 标准正余弦位置编码
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)  # 不参与训练，但会随模型保存

    def forward(self, x):
        """
        x: [B, L, C]
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]

# 余弦分类头
class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(x, w.t())

# Raman 主模型
# - `backbone_type` 控制前端特征提取器
# - `encoder_type` 控制序列编码器
class RamanClassifier1D(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        self.backbone_type = str(getattr(self.config, "backbone_type", "cnn")).lower()
        if self.backbone_type not in ("cnn", "identity"):
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")
        self.cnn_block_type = str(
            getattr(self.config, "cnn_block_type", "resnext")
        ).lower()
        if self.cnn_block_type not in ("resnet", "resnext"):
            raise ValueError(f"Unknown cnn_block_type: {self.cnn_block_type}")

        self.encoder_type = str(getattr(self.config, "encoder_type", "transformer")).lower()
        if self.encoder_type not in ("transformer", "lstm", "none"):
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.cnn_backbone_on = (self.backbone_type == "cnn")
        self.transformer_on = (self.encoder_type == "transformer")
        self.lstm_on = (self.encoder_type == "lstm")
        self.reduction = self.config.reduction
        self.se_use = self.config.se_use
        self.backbone_activation_name = str(
            getattr(self.config, "backbone_activation", "relu")
        ).lower()
        self.backbone_activation_negative_slope = float(
            getattr(self.config, "backbone_activation_negative_slope", 0.01)
        )
        build_activation(
            self.backbone_activation_name,
            inplace=True,
            negative_slope=self.backbone_activation_negative_slope,
        )
        self.proj_dim = int(getattr(self.config, "transformer_dim", 192))

        if self.cnn_backbone_on:
            self._build_cnn_backbone()
        else:
            self.identity_pool_kernel = max(1, int(getattr(self.config, "identity_pool_kernel", 16)))
            if self.identity_pool_kernel == 1:
                self.identity_pool = nn.Identity()
            else:
                self.identity_pool = nn.AvgPool1d(
                    kernel_size=self.identity_pool_kernel,
                    stride=self.identity_pool_kernel
                )
            self.input_proj = nn.Sequential(
                nn.Conv1d(self.config.in_channels, self.proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.proj_dim),
                nn.GELU()
            )

        # 序列编码器
        self.seq_dim = self.proj_dim
        if self.transformer_on:
            self.pos_encoder = PositionalEncoding1D(d_model=self.proj_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self.config.transformer_nhead,
                dim_feedforward=self.config.transformer_ffn_dim,
                dropout=self.config.transformer_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.transformer_layers
            )
        elif self.lstm_on:
            self.lstm_hidden = int(getattr(self.config, "lstm_hidden", self.proj_dim))
            self.lstm_layers = int(getattr(self.config, "lstm_layers", 1))
            self.lstm_dropout = float(getattr(self.config, "lstm_dropout", 0.0))
            self.lstm_bidirectional = bool(getattr(self.config, "lstm_bidirectional", False))
            self.lstm = nn.LSTM(
                input_size=self.proj_dim,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                dropout=self.lstm_dropout if self.lstm_layers > 1 else 0.0,
                bidirectional=self.lstm_bidirectional,
                batch_first=True
            )
            self.seq_dim = self.lstm_hidden * (2 if self.lstm_bidirectional else 1)

        # ---------------------- Pooling -----------------------
        self.pooling_type = str(getattr(self.config, "pooling_type", "attn")).lower()
        if self.pooling_type not in ("attn", "stat"):
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        if self.pooling_type == "attn":
            self.att_pool_dropout = float(getattr(self.config, "att_pool_dropout", 0.0))
            self.att_pool = nn.Sequential(
                nn.Linear(self.seq_dim, self.seq_dim // 2),
                nn.GELU(),
                nn.Dropout(self.att_pool_dropout),
                nn.Linear(self.seq_dim // 2, 1)
            )
        else:
            self.att_pool = None

        self.feat_dim = self.seq_dim * 2 if self.pooling_type == "stat" else self.seq_dim
        self.cosine_head = bool(getattr(self.config, "cosine_head", False))
        self.cosine_scale = float(getattr(self.config, "cosine_scale", 30.0))

        # ------------------------- Classifier ---------------------------
        if isinstance(num_classes, (dict, list, tuple)):
            raise ValueError("Multi-head classifier is removed; pass a single int num_classes.")
        if self.cosine_head:
            self.head = CosineClassifier(self.feat_dim, int(num_classes), scale=self.cosine_scale)
        else:
            self.head = nn.Linear(self.feat_dim, int(num_classes))

    def _build_cnn_backbone(self):
        # 输入 stem
        self.stem_multiscale = bool(getattr(self.config, "stem_multiscale", False))
        if self.stem_multiscale:
            kernel_sizes = getattr(self.config, "stem_kernel_sizes", None) or [3, 7, 15]
            kernel_sizes = [int(k) for k in kernel_sizes]
            num_branches = max(1, len(kernel_sizes))
            base_ch = 64 // num_branches
            rem = 64 - base_ch * num_branches # 不一定整除，判断剩余
            branch_channels = [base_ch] * num_branches
            for i in range(rem):
                branch_channels[i] += 1 # 按顺序补上
            self.stem_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(
                        self.config.in_channels,
                        out_ch,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_ch),
                    build_activation(
                        self.backbone_activation_name,
                        inplace=True,
                        negative_slope=self.backbone_activation_negative_slope,
                    )
                )
                for k, out_ch in zip(kernel_sizes, branch_channels)
            ])
            self.stem_pool = nn.AvgPool1d(kernel_size=2)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    self.config.in_channels,
                    64,
                    kernel_size=self.config.stem_kernel_size,
                    stride=1,
                    padding=self.config.stem_kernel_size // 2,
                    bias=False
                ),
                nn.BatchNorm1d(64),
                build_activation(
                    self.backbone_activation_name,
                    inplace=True,
                    negative_slope=self.backbone_activation_negative_slope,
                ),
                nn.AvgPool1d(kernel_size=2)
            )

        # CNN 主干
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            self._make_layer(128, 2)
        )
        self.layer3 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            self._make_layer(256, 2)
        )
        self.layer4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            self._make_layer(384, 2)
        )
        self.proj = nn.Conv1d(384, self.proj_dim, kernel_size=1, bias=False)

    def _make_layer(self, planes, blocks):
        layers = []

        # 第一个 block：in_planes -> planes
        layers.append(
            ResidualBottleneck1D(
                in_channels=self.in_planes,
                out_channels=planes,
                block_type=self.cnn_block_type,
                cardinality=self.config.cardinality,
                base_width=self.config.base_width,
                bottleneck_ratio=getattr(self.config, "resnet_bottleneck_ratio", 4),
                reduction=self.reduction,
                se_use=self.se_use,
                activation_name=self.backbone_activation_name,
                activation_negative_slope=self.backbone_activation_negative_slope,
            )
        )
        self.in_planes = planes

        # 后续 block：planes -> planes
        for _ in range(1, blocks):
            layers.append(
                ResidualBottleneck1D(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    block_type=self.cnn_block_type,
                    cardinality=self.config.cardinality,
                    base_width=self.config.base_width,
                    bottleneck_ratio=getattr(self.config, "resnet_bottleneck_ratio", 4),
                    reduction=self.reduction,
                    se_use=self.se_use,
                    activation_name=self.backbone_activation_name,
                    activation_negative_slope=self.backbone_activation_negative_slope,
                )
            )

        return nn.Sequential(*layers)

    def _forward_feature_extractor(self, x):
        if self.cnn_backbone_on:
            if self.stem_multiscale:
                out = torch.cat([branch(x) for branch in self.stem_branches], dim=1)
                out = self.stem_pool(out)
            else:
                out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            return self.proj(out)

        out = self.identity_pool(x)
        return self.input_proj(out)

    # --------------------------- Forward ------------------------------
    def forward(self, x, return_feat=False):
        # x: [B, C, L]
        out = self._forward_feature_extractor(x)

        # [B, C, L] → [B, L, C]
        out = out.permute(0, 2, 1)

        # Transformer / LSTM
        if self.transformer_on:
            out = self.pos_encoder(out)
            # 每个位置的上下文混合信息
            # Transformer 只是让“峰 A 知道峰 B 的存在”
            out = self.transformer(out)
        elif self.lstm_on:
            out, _ = self.lstm(out)

        # Pooling
        if self.pooling_type == "attn":
            att = self.att_pool(out)  # [B, L, 1]
            att = torch.softmax(att, dim=1)
            feat = (out * att).sum(dim=1)  # [B, C]
        else:
            mean = out.mean(dim=1)
            std = out.std(dim=1, unbiased=False)
            feat = torch.cat([mean, std], dim=1)

        # Classification
        logits = self.head(feat)


        if return_feat:
            return logits, feat
        return logits


# 兼容现有导入名
