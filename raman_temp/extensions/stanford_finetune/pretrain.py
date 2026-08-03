"""Stanford 30 类预训练编排。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from raman_temp.core.config import Config, DatasetConfig
from raman_temp.core.paths import PROJECT_ROOT
from raman_temp.data.index import DatasetIndex
from raman_temp.training.workflow import TrainRequest, run_training

from .config import StanfordPretrainConfig
from .dataset import load_reference_wavenumbers, stanford_dataset_dir
from .report import write_pretrain_report


def build_pretrain_config(base_config: Config) -> Config:
    """从常规可编辑配置派生 Stanford 固定输入和数据集标识。"""
    input_config = replace(
        base_config.input,
        input_grid_mode="stanford_transfer",
        norm_method="minmax",
    )
    reference_axis = load_reference_wavenumbers()
    if reference_axis.size != input_config.target_points:
        raise ValueError("Stanford 参考轴长度与 core/config.py 的 target_points 不一致")
    return replace(
        base_config,
        dataset=DatasetConfig(profile_id="Stanford"),
        input=input_config,
    )


def run_pretrain(
    pretrain_config: StanfordPretrainConfig,
    base_config: Config | None = None,
) -> dict:
    """以 Stanford 已构建 train 目录执行一个全局 30 类预训练任务。"""
    config = build_pretrain_config(Config() if base_config is None else base_config)
    train_dir = stanford_dataset_dir() / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"缺少 Stanford train，请先执行 stanford prepare：{train_dir}")
    experiment_dir = _resolve_experiment_dir(pretrain_config)
    result = run_training(
        TrainRequest(
            config=config,
            level_name=pretrain_config.level_name,
            train_per_parent_enable=False,
            experiment_dir=experiment_dir,
            train_dir=train_dir,
            run_name=pretrain_config.run_name,
        )
    )
    class_names = result["class_names_by_level"].get(pretrain_config.level_name, [])
    write_pretrain_report(experiment_dir, pretrain_config.level_name, class_names)
    return result


def _resolve_experiment_dir(config: StanfordPretrainConfig) -> Path:
    """解析预训练实验根；缺失时使用 Stanford 专属默认输出位置。"""
    if config.experiment_dir is not None:
        return Path(config.experiment_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "output" / "Stanford" / timestamp
