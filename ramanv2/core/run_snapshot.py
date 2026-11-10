"""已完成实验的配置快照读取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.core.config import Config, build_config
from ramanv2.core.config_snapshot import (
    SHARED_CONFIG_NAME,
    read_yaml_dict,
)
from ramanv2.common.paths import find_ancestor_dir
from ramanv2.common.hierarchy import validate_label_schema


@dataclass(frozen=True)
class RunSnapshot:
    """读取到的配置、实验根与模型槽位路径。"""

    config: Config
    experiment_dir: Path
    run_dir: Path


def load_run_snapshot(
    run_dir: Path | str,
    experiment_dir: Path | str | None = None,
) -> RunSnapshot:
    """读取模型槽位所属实验的完整配置快照。"""
    run_path = Path(run_dir).resolve()
    experiment_path = _resolve_experiment_dir(run_path, experiment_dir)
    shared_path = experiment_path / SHARED_CONFIG_NAME
    if not shared_path.is_file():
        raise FileNotFoundError(f"缺少实验配置快照：{shared_path}")
    values = read_yaml_dict(shared_path)
    validate_label_schema(values)
    return RunSnapshot(build_config(values), experiment_path, run_path)


def _resolve_experiment_dir(run_path: Path, experiment_dir: Path | str | None) -> Path:
    """从显式路径或父目录链中定位 shared_config.yaml。"""
    if experiment_dir is not None:
        return Path(experiment_dir).resolve()
    return find_ancestor_dir(run_path, SHARED_CONFIG_NAME)
