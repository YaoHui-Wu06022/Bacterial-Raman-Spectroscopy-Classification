"""Stanford 来源 run 与目标迁移请求的兼容性校验。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from raman_temp.core.config import Config, DatasetConfig
from raman_temp.core.experiment_reader import RunSnapshot, load_run_snapshot
from raman_temp.data.io import iter_arc_dirs, read_arc_data
from raman_temp.data.profiles import get_dataset_dir, get_profile
from raman_temp.spectra.bands import build_valid_mask

from .dataset import load_reference_wavenumbers


def load_stanford_snapshot(run_dir: Path | str, level_name: str) -> RunSnapshot:
    """读取并校验来源 run 是可用于迁移的 Stanford 预训练模型。"""
    snapshot = load_run_snapshot(run_dir)
    if snapshot.config.dataset.profile_id != "Stanford":
        raise ValueError("来源 run 不是 Stanford 预训练结果")
    if snapshot.config.input.input_grid_mode != "stanford_transfer":
        raise ValueError("来源 run 未使用 stanford_transfer 输入网格")
    if snapshot.config.input.norm_method != "minmax":
        raise ValueError("来源 run 未使用 minmax 归一化")
    model_path = snapshot.run_dir / f"{level_name}_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"来源 run 缺少模型权重：{model_path}")
    reference_axis = load_reference_wavenumbers()
    if reference_axis.size != snapshot.config.input.target_points:
        raise ValueError("Stanford 参考轴长度与来源输入规格不一致")
    return snapshot


def build_transfer_config(
    base_config: Config,
    source_snapshot: RunSnapshot,
    target_profile: str,
    learning_rate: float | None,
) -> Config:
    """以来源输入/模型规格和目标训练策略构建一次性迁移配置。"""
    get_profile(target_profile)
    training = base_config.training
    if learning_rate is not None:
        training = replace(training, learning_rate=float(learning_rate))
    return replace(
        base_config,
        dataset=DatasetConfig(profile_id=target_profile),
        input=source_snapshot.config.input,
        model=source_snapshot.config.model,
        training=training,
    )


def validate_transfer_train_axis(target_profile: str, input_config) -> Path:
    """确认目标 train 已按 Stanford 共享轴和坏段规则构建。"""
    profile = get_profile(target_profile)
    train_dir = get_dataset_dir(profile) / profile.root_train_clean
    if not train_dir.is_dir():
        raise FileNotFoundError(f"缺少目标 train：{train_dir}")
    reference_axis = load_reference_wavenumbers()
    valid_mask = build_valid_mask(reference_axis, input_config.bad_bands)
    expected_axis = reference_axis if valid_mask is None else reference_axis[valid_mask]
    for directory, filenames in iter_arc_dirs(train_dir):
        axis, _intensities = read_arc_data(directory / filenames[0])
        if axis.shape != expected_axis.shape or not np.allclose(axis, expected_axis, rtol=0.0, atol=1e-5):
            raise ValueError(f"目标 train 未按 Stanford 共享轴构建：{directory / filenames[0]}")
        return train_dir
    raise ValueError(f"目标 train 没有 .arc_data 文件：{train_dir}")


def resolve_target_dataset_dir(target_profile: str) -> Path:
    """返回已校验常规 profile 的数据集目录。"""
    profile = get_profile(target_profile)
    return get_dataset_dir(profile)
