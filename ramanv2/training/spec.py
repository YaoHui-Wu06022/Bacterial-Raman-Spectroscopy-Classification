"""训练循环使用的只读规格。"""

from __future__ import annotations

from dataclasses import dataclass

from ramanv2.core.config import ExecutionConfig, TrainingConfig


@dataclass(frozen=True)
class LoaderSpec:
    """单个 DataLoader 的固定构建参数。"""

    batch_size: int
    num_workers: int
    shuffle_enable: bool
    pin_memory_enable: bool
    persistent_workers_enable: bool
    prefetch_factor: int


@dataclass(frozen=True)
class OptimizerSpec:
    """优化器和学习率调度器参数。"""

    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    scheduler_t_max: int
    scheduler_eta_min: float


@dataclass(frozen=True)
class LossSpec:
    """主分类、辅助损失、EMA 与早停的计算参数。"""

    focal_gamma: float
    align_enable: bool
    align_weight: float
    align_start_epoch: int
    align_end_epoch: int
    supcon_enable: bool
    supcon_weight: float
    supcon_tau: float
    supcon_start_epoch: int
    supcon_end_epoch: int
    auxiliary_decay_start_ratio: float
    ema_enable: bool
    ema_alpha: float
    ema_start_epoch: int
    ema_difficulty_weight: float
    early_stop_f1_weight: float
    early_stop_accuracy_weight: float


@dataclass(frozen=True)
class TrainingSpec:
    """训练循环与数据划分共用的不可变训练规格。"""

    seed: int
    epochs: int
    patience: int
    train_ratio: float
    split_by_source_prefix_enable: bool
    train_loader: LoaderSpec
    validation_loader: LoaderSpec
    optimizer: OptimizerSpec
    loss: LossSpec


@dataclass(frozen=True)
class ExecutionSpec:
    """设备、随机性、自动混合精度和续训控制。"""

    resume_enable: bool
    checkpoint_interval: int
    use_gpu_enable: bool
    amp_enable: bool
    deterministic_enable: bool


def build_training_spec(
    training_config: TrainingConfig,
    execution_config: ExecutionConfig,
) -> TrainingSpec:
    """从用户配置构建训练循环所需规格，并补齐调度周期默认值。"""
    scheduler_t_max = training_config.scheduler_t_max or training_config.epochs
    return TrainingSpec(
        seed=int(execution_config.seed),
        epochs=int(training_config.epochs),
        patience=int(training_config.patience),
        train_ratio=float(training_config.train_split),
        split_by_source_prefix_enable=bool(training_config.split_by_source_prefix),
        train_loader=_build_loader_spec(training_config, True),
        validation_loader=_build_loader_spec(training_config, False),
        optimizer=OptimizerSpec(
            learning_rate=float(training_config.learning_rate),
            weight_decay=float(training_config.weight_decay),
            grad_clip_norm=float(training_config.grad_clip_norm),
            scheduler_t_max=int(scheduler_t_max),
            scheduler_eta_min=float(training_config.scheduler_eta_min),
        ),
        loss=LossSpec(
            focal_gamma=float(training_config.gamma),
            align_enable=bool(training_config.use_align_loss),
            align_weight=float(training_config.align_loss_weight),
            align_start_epoch=int(training_config.align_start),
            align_end_epoch=int(training_config.align_end),
            supcon_enable=bool(training_config.use_supcon_loss),
            supcon_weight=float(training_config.supcon_loss_weight),
            supcon_tau=float(training_config.supcon_tau),
            supcon_start_epoch=int(training_config.supcon_start),
            supcon_end_epoch=int(training_config.supcon_end),
            auxiliary_decay_start_ratio=float(training_config.decay_start_ratio),
            ema_enable=bool(training_config.use_ema),
            ema_alpha=float(training_config.ema_alpha),
            ema_start_epoch=int(training_config.ema_start_epoch),
            ema_difficulty_weight=float(training_config.ema_difficulty_weight),
            early_stop_f1_weight=float(training_config.early_stop_w_f1),
            early_stop_accuracy_weight=float(training_config.early_stop_w_acc),
        ),
    )


def build_execution_spec(execution_config: ExecutionConfig) -> ExecutionSpec:
    """从用户配置构建设备与恢复相关的执行规格。"""
    return ExecutionSpec(
        resume_enable=bool(execution_config.resume_training),
        checkpoint_interval=int(execution_config.checkpoint_interval),
        use_gpu_enable=bool(execution_config.use_gpu),
        amp_enable=bool(execution_config.use_amp),
        deterministic_enable=bool(execution_config.deterministic),
    )


def _build_loader_spec(training_config: TrainingConfig, shuffle_enable: bool) -> LoaderSpec:
    """按训练或验证角色构建对应 DataLoader 规格。"""
    workers = (
        training_config.train_loader_num_workers
        if shuffle_enable
        else training_config.val_loader_num_workers
    )
    return LoaderSpec(
        batch_size=int(training_config.batch_size),
        num_workers=int(workers),
        shuffle_enable=shuffle_enable,
        pin_memory_enable=bool(training_config.loader_pin_memory),
        persistent_workers_enable=bool(training_config.loader_persistent_workers),
        prefetch_factor=int(training_config.loader_prefetch_factor),
    )
