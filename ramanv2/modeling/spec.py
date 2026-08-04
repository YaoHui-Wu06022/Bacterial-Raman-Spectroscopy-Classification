"""模型结构规格及其构建前校验。"""

from __future__ import annotations

from dataclasses import dataclass

from ramanv2.core.config import ModelConfig
from ramanv2.core.input_spec import InputSpec


@dataclass(frozen=True)
class ModelSpec:
    """只描述会影响模型模块树和前向计算的结构字段。"""

    in_channels: int
    backbone_type: str
    cnn_block_type: str
    cardinality: int
    base_width: int
    resnet_bottleneck_ratio: int
    se_enable: bool
    reduction: int
    activation_negative_slope: float
    stem_kernel_sizes: tuple[int, ...]
    transformer_dim: int
    encoder_type: str
    transformer_nhead: int
    transformer_ffn_dim: int
    transformer_layers: int
    transformer_dropout: float
    lstm_hidden: int
    lstm_layers: int
    lstm_dropout: float
    lstm_bidirectional_enable: bool
    pooling_type: str
    cosine_head_enable: bool
    cosine_scale: float


def build_model_spec(model_config: ModelConfig, input_spec: InputSpec) -> ModelSpec:
    """从模型配置和输入规格提取模型结构字段。"""
    return ModelSpec(
        in_channels=int(input_spec.in_channels),
        backbone_type=str(model_config.backbone_type).lower(),
        cnn_block_type=str(model_config.cnn_block_type).lower(),
        cardinality=int(model_config.cardinality),
        base_width=int(model_config.base_width),
        resnet_bottleneck_ratio=int(model_config.resnet_bottleneck_ratio),
        se_enable=bool(model_config.se_use),
        reduction=int(model_config.reduction),
        activation_negative_slope=float(model_config.backbone_activation_negative_slope),
        stem_kernel_sizes=tuple(int(value) for value in model_config.stem_kernel_sizes),
        transformer_dim=int(model_config.transformer_dim),
        encoder_type=str(model_config.encoder_type).lower(),
        transformer_nhead=int(model_config.transformer_nhead),
        transformer_ffn_dim=int(model_config.transformer_ffn_dim),
        transformer_layers=int(model_config.transformer_layers),
        transformer_dropout=float(model_config.transformer_dropout),
        lstm_hidden=int(model_config.lstm_hidden),
        lstm_layers=int(model_config.lstm_layers),
        lstm_dropout=float(model_config.lstm_dropout),
        lstm_bidirectional_enable=bool(model_config.lstm_bidirectional),
        pooling_type=str(model_config.pooling_type).lower(),
        cosine_head_enable=bool(model_config.cosine_head),
        cosine_scale=float(model_config.cosine_scale),
    )


def validate_model_spec(model_spec: ModelSpec) -> None:
    """在创建网络前校验结构字段是否自洽。"""
    if model_spec.backbone_type not in {"cnn", "direct"}:
        raise ValueError(f"Unknown backbone_type: {model_spec.backbone_type}")
    if model_spec.cnn_block_type not in {"resnet", "resnext"}:
        raise ValueError(f"Unknown cnn_block_type: {model_spec.cnn_block_type}")
    if model_spec.encoder_type not in {"transformer", "lstm", "none"}:
        raise ValueError(f"Unknown encoder_type: {model_spec.encoder_type}")
    if model_spec.pooling_type not in {"attn", "stat"}:
        raise ValueError(f"Unknown pooling_type: {model_spec.pooling_type}")
    if min(model_spec.in_channels, model_spec.reduction, model_spec.transformer_dim, model_spec.transformer_nhead, model_spec.transformer_layers, model_spec.lstm_hidden, model_spec.lstm_layers) <= 0:
        raise ValueError("模型通道、层数和 reduction 必须为正数")
    if model_spec.encoder_type == "transformer" and model_spec.transformer_dim % model_spec.transformer_nhead:
        raise ValueError("transformer_dim 必须能被 transformer_nhead 整除")
    if model_spec.cnn_block_type == "resnext" and model_spec.cardinality <= 0:
        raise ValueError("ResNeXt cardinality 必须为正数")
