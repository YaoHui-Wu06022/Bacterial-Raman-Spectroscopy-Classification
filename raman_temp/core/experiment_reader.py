"""已完成实验的配置快照读取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raman_temp.core.config import Config, build_config
from raman_temp.core.config_file import (
    MODEL_CONFIG_NAME,
    RESOLVED_CONFIG_NAME,
    SHARED_CONFIG_NAME,
    read_yaml_dict,
)


@dataclass(frozen=True)
class RunSnapshot:
    """读取到的配置、实验根、run 路径和非配置元数据。"""

    config: Config
    experiment_dir: Path
    run_dir: Path
    task_values: dict[str, Any]
    path_values: dict[str, Any]


def load_run_snapshot(
    run_dir: Path | str,
    experiment_dir: Path | str | None = None,
) -> RunSnapshot:
    """读取单个 run 快照，不向 Config 写入路径或任务状态。"""
    run_path = Path(run_dir).resolve()
    experiment_path = _resolve_experiment_dir(run_path, experiment_dir)
    shared_path = experiment_path / SHARED_CONFIG_NAME
    model_path = run_path / MODEL_CONFIG_NAME
    resolved_path = run_path / RESOLVED_CONFIG_NAME
    if shared_path.is_file() and model_path.is_file():
        values = read_yaml_dict(shared_path) | read_yaml_dict(model_path)
    elif resolved_path.is_file():
        values = read_yaml_dict(resolved_path)
    else:
        raise FileNotFoundError(f"缺少配置快照：{run_path}")
    task_values = _select_task_values(values)
    path_values = _select_path_values(values)
    return RunSnapshot(build_config(values), experiment_path, run_path, task_values, path_values)


def _resolve_experiment_dir(run_path: Path, experiment_dir: Path | str | None) -> Path:
    """从显式路径或父目录链中定位 shared_config.yaml。"""
    if experiment_dir is not None:
        return Path(experiment_dir).resolve()
    for parent in (run_path, *run_path.parents):
        if (parent / SHARED_CONFIG_NAME).is_file():
            return parent
    raise FileNotFoundError(f"无法定位实验根 shared_config.yaml：{run_path}")


def _select_task_values(values: dict[str, Any]) -> dict[str, Any]:
    """提取任务范围快照字段。"""
    keys = {
        "level_name",
        "only_parent",
        "only_parent_name",
        "filter_level",
        "filter_value",
        "train_per_parent_enable",
    }
    return {key: value for key, value in values.items() if key in keys}


def _select_path_values(values: dict[str, Any]) -> dict[str, Any]:
    """提取运行时解析得到的目录路径字段。"""
    keys = {"experiment_dir", "run_dir"}
    return {key: value for key, value in values.items() if key in keys}
