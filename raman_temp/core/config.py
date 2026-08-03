"""项目唯一的可编辑配置源。

配置只描述可复现的处理、模型与训练参数；训练范围、输出目录和恢复目录由
入口请求与运行上下文分别持有，避免它们在进程间互相污染。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


DEFAULT_BAD_BANDS = ((890.0, 950.0),)


@dataclass(frozen=True)
class DatasetConfig:
    """常规数据集的 profile 标识。实际目录由 data.profiles 解析。"""

    profile_id: str = "GN"


@dataclass(frozen=True)
class InputConfig:
    """训练与推理共用的光谱输入规格。"""

    # 统一插值波数轴与 CCD 坏段；训练、推理和审核均以此为准。
    cut_min: float = 600.0
    cut_max: float = 1800.0
    target_points: int = 896
    input_grid_mode: str = "standard"
    bad_bands: tuple[tuple[float, float], ...] = DEFAULT_BAD_BANDS
    # 模型输入通道：原谱始终保留，平滑谱和一阶导由开关额外叠加。
    norm_method: str = "snv"
    smooth_use: bool = True
    d1_use: bool = False
    win_smooth: int = 15
    win1: int = 15


@dataclass(frozen=True)
class ModelConfig:
    """影响网络模块树与前向计算的模型参数。"""

    # CNN 主干与可选 SE 注意力模块。
    se_use: bool = True
    reduction: int = 8
    backbone_activation_negative_slope: float = 0.05
    backbone_type: str = "cnn"
    cnn_block_type: str = "resnext"
    cardinality: int = 4
    base_width: int = 4
    resnet_bottleneck_ratio: int = 4
    identity_pool_kernel: int = 8
    # CNN 输出后的序列编码器；none 用于纯 CNN 对照。
    encoder_type: str = "transformer"
    transformer_nhead: int = 8
    transformer_dim: int = 256
    transformer_ffn_dim: int = 512
    transformer_layers: int = 2
    transformer_dropout: float = 0.2
    lstm_hidden: int = 192
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = False
    # 聚合方式与分类头形式。
    pooling_type: str = "attn"
    cosine_head: bool = False
    cosine_scale: float = 25.0
    stem_kernel_sizes: tuple[int, ...] = (3, 7, 15)


@dataclass(frozen=True)
class TrainingConfig:
    """数据划分、增强、优化、损失与早停参数。"""

    # 数据划分与 DataLoader 参数。
    split_by_source_prefix: bool = False
    train_split: float = 0.8
    epochs: int = 80
    patience: int = 50
    batch_size: int = 64
    train_loader_num_workers: int = 2
    val_loader_num_workers: int = 2
    loader_pin_memory: bool = True
    loader_persistent_workers: bool = True
    loader_prefetch_factor: int = 2
    # 优化器、梯度裁剪与余弦退火调度器。
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    scheduler_t_max: int | None = None
    scheduler_eta_min: float = 1e-5
    # 辅助损失仅在对应开关启用时参与总损失。
    use_align_loss: bool = False
    align_loss_weight: float = 0.01
    align_start: int = 20
    align_end: int = 50
    use_supcon_loss: bool = True
    supcon_loss_weight: float = 0.01
    supcon_tau: float = 0.12
    supcon_start: int = 25
    supcon_end: int = 50
    decay_start_ratio: float = 0.7
    # Focal 损失、类别难度 EMA 与早停评分。
    gamma: float = 1.2
    use_ema: bool = True
    ema_alpha: float = 0.9
    ema_start_epoch: int = 1
    ema_difficulty_weight: float = 1.0
    early_stop_w_f1: float = 0.6
    early_stop_w_acc: float = 0.4
    # 训练集随机增强的触发概率与单样本叠加上限。
    p_piecewise_gain: float = 0.40
    p_noise: float = 0.70
    p_axis: float = 0.30
    p_baseline_weak: float = 0.55
    p_baseline_strong: float = 0.35
    p_shift: float = 0.40
    p_broadening: float = 0.45
    p_cut: float = 0.20
    max_pre_augs: int = 4
    max_post_augs: int = 2


@dataclass(frozen=True)
class ExecutionConfig:
    """单次运行的随机性、设备、混合精度与断点保存策略。"""

    # 运行随机性、设备、混合精度和断点保存策略。
    seed: int = 42
    resume_training: bool = True
    checkpoint_interval: int = 20
    use_gpu: bool = True
    use_amp: bool = True
    deterministic: bool = True


@dataclass(frozen=True)
class Config:
    """由五个职责明确的配置组组成的不可变配置对象。"""

    dataset: DatasetConfig = DatasetConfig()
    input: InputConfig = InputConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    execution: ExecutionConfig = ExecutionConfig()

    def to_shared_dict(self) -> dict[str, Any]:
        """导出实验根共享的 profile 与输入快照。"""
        data = _yaml_ready(asdict(self.dataset) | asdict(self.input))
        data["dataset_name"] = data.pop("profile_id")
        return data

    def to_model_dict(self) -> dict[str, Any]:
        """导出模型与训练参数快照。"""
        return _yaml_ready(asdict(self.model) | asdict(self.training))

    def to_execution_dict(self) -> dict[str, Any]:
        """导出运行期执行参数快照。"""
        return _yaml_ready(asdict(self.execution))

    def to_dict(self) -> dict[str, Any]:
        """导出完整扁平快照，供 YAML 与文本日志使用。"""
        return self.to_shared_dict() | self.to_model_dict() | self.to_execution_dict()


def build_config(values: Mapping[str, Any] | None = None) -> Config:
    """用扁平快照或局部覆盖构建不可变配置对象。"""
    source = dict(values or {})
    if "profile_id" not in source and "dataset_name" in source:
        source["profile_id"] = source["dataset_name"]
    if "scheduler_t_max" not in source and "scheduler_Tmax" in source:
        source["scheduler_t_max"] = source["scheduler_Tmax"]
    return Config(
        dataset=_build_group(DatasetConfig, source),
        input=_build_group(InputConfig, source),
        model=_build_group(ModelConfig, source),
        training=_build_group(TrainingConfig, source),
        execution=_build_group(ExecutionConfig, source),
    )


def _build_group(group_type, values: Mapping[str, Any]):
    """按数据类声明字段提取覆盖值，并统一序列字段的容器类型。"""
    selected = {
        item.name: _coerce_field_value(item.name, values[item.name])
        for item in fields(group_type)
        if item.name in values
    }
    return group_type(**selected)


def _coerce_field_value(name: str, value: Any) -> Any:
    """恢复 YAML 中列表表示的固定元组字段。"""
    if name == "bad_bands":
        return tuple((float(start), float(end)) for start, end in value)
    if name == "stem_kernel_sizes":
        return tuple(int(item) for item in value)
    return value


def _yaml_ready(value: Any) -> Any:
    """将配置中的 tuple 递归转换为 YAML 的列表表示。"""
    if isinstance(value, tuple):
        return [_yaml_ready(item) for item in value]
    if isinstance(value, list):
        return [_yaml_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _yaml_ready(item) for key, item in value.items()}
    return value
