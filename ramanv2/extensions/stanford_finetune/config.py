"""Stanford 扩展唯一可编辑的请求配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StanfordPrepareConfig:
    """描述 Stanford 数据恢复、下载和预训练集构建请求。"""

    download_enable: bool = False
    import_enable: bool = False
    rebuild_train_enable: bool = False


@dataclass(frozen=True)
class StanfordPretrainConfig:
    """描述 Stanford 30 类预训练的训练范围和输出位置。"""

    level_name: str = "level_1"
    experiment_dir: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class StanfordTransferConfig:
    """描述 Stanford 来源 run 与常规目标 profile 的单阶段微调策略。"""

    source_run_dir: str = ""
    source_level: str = "level_1"
    target_profile: str = "GN"
    level_name: str = "level_1"
    parent: str | None = None
    global_enable: bool = False
    experiment_dir: str | None = None
    run_name: str | None = None
    learning_rate: float | None = 5e-5
    rebuild_train_enable: bool = False
    trainable_modules: tuple[str, ...] = ()


PREPARE_CONFIG = StanfordPrepareConfig()
PRETRAIN_CONFIG = StanfordPretrainConfig()
TRANSFER_CONFIG = StanfordTransferConfig()
