"""Stanford 预训练权重到常规 profile 的单阶段迁移编排。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from raman_temp.core.config import Config
from raman_temp.core.paths import PROJECT_ROOT
from raman_temp.data.build import build_train
from raman_temp.data.profiles import get_profile
from raman_temp.training.workflow import TrainRequest, run_training

from .compatibility import (
    build_transfer_config,
    load_stanford_snapshot,
    resolve_target_dataset_dir,
    validate_transfer_train_axis,
)
from .config import StanfordTransferConfig
from .dataset import load_reference_wavenumbers
from .initializer import TransferInitializer
from .report import write_transfer_reports


def run_transfer(
    transfer_config: StanfordTransferConfig,
    base_config: Config | None = None,
) -> dict:
    """构建目标数据、重置分类头并执行 Stanford 单阶段迁移训练。"""
    if not transfer_config.source_run_dir:
        raise ValueError("StanfordTransferConfig.source_run_dir 不能为空")
    source_snapshot = load_stanford_snapshot(
        transfer_config.source_run_dir,
        transfer_config.source_level,
    )
    config = build_transfer_config(
        Config() if base_config is None else base_config,
        source_snapshot,
        transfer_config.target_profile,
        transfer_config.learning_rate,
    )
    profile = get_profile(transfer_config.target_profile)
    target_dir = resolve_target_dataset_dir(transfer_config.target_profile)
    reference_axis = load_reference_wavenumbers()
    if transfer_config.rebuild_train_enable:
        build_train(
            profile,
            target_dir,
            input_config=config.input,
            reference_wavenumbers=reference_axis,
        )
    validate_transfer_train_axis(transfer_config.target_profile, config.input)
    only_parent, only_parent_name = _resolve_parent(transfer_config.parent)
    initializer = TransferInitializer(
        source_snapshot.run_dir / f"{transfer_config.source_level}_model.pt",
        transfer_config.trainable_modules,
    )
    experiment_dir = _resolve_experiment_dir(transfer_config, profile.dataset_name)
    result = run_training(
        TrainRequest(
            config=config,
            level_name=transfer_config.level_name,
            only_parent=only_parent,
            only_parent_name=only_parent_name,
            train_per_parent_enable=not transfer_config.global_enable,
            experiment_dir=experiment_dir,
            run_name=transfer_config.run_name,
            initialize_model=initializer.initialize,
            apply_training_mode=initializer.apply_training_mode,
        )
    )
    write_transfer_reports(
        experiment_dir,
        {
            "source_run_dir": str(source_snapshot.run_dir),
            "source_level": transfer_config.source_level,
            "target_profile": transfer_config.target_profile,
            "target_level": transfer_config.level_name,
            "transfer_config": asdict(transfer_config),
        },
        initializer.reports,
    )
    return result


def _resolve_parent(value: str | None) -> tuple[int | None, str | None]:
    """将可选父类值区分为数值索引或类别名称。"""
    if value is None:
        return None, None
    return (int(value), None) if value.isdigit() else (None, value)


def _resolve_experiment_dir(config: StanfordTransferConfig, dataset_name: str) -> Path:
    """解析迁移实验根，缺失时使用目标 profile 的默认输出位置。"""
    if config.experiment_dir is not None:
        return Path(config.experiment_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "output" / dataset_name / timestamp
