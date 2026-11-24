"""配置快照的 YAML 读写、共享输入校验与已运行实验读取。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ramanv2.common.hierarchy import validate_label_schema
from ramanv2.common.paths import find_ancestor_dir
from ramanv2.core.config import Config, build_config


SHARED_CONFIG_NAME = "shared_config.yaml"


@dataclass(frozen=True)
class RunSnapshot:
    """读取到的配置、实验根与模型槽位路径。"""

    config: Config
    experiment_dir: Path
    run_dir: Path


def read_yaml_dict(path: Path | str) -> dict[str, Any]:
    """读取 YAML 字典；空文件按空字典处理。"""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须包含字典：{path}")
    return data


def write_yaml_dict(path: Path | str, data: Mapping[str, Any]) -> None:
    """原子写入单个 UTF-8 YAML 快照。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_yaml_ready(data)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        yaml.safe_dump(dict(data), file, sort_keys=False, allow_unicode=True)
    try:
        os.replace(temp_path, target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_shared_config(path: Path | str, config: Config) -> None:
    """保存实验根唯一的完整配置快照。"""
    write_yaml_dict(path, config.to_dict())


def assert_shared_compatible(config: Config, shared_values: Mapping[str, Any]) -> None:
    """确认当前完整配置与实验根已有快照一致。"""
    validate_label_schema(shared_values)
    shared = dict(shared_values)
    current = config.to_dict()
    mismatches = [
        f"{key}: {shared.get(key)!r} != {value!r}"
        for key, value in current.items()
        if shared.get(key) != value
    ]
    if mismatches:
        raise ValueError("实验根共享配置不一致：\n" + "\n".join(mismatches))


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


def _assert_yaml_ready(data: Mapping[str, Any]) -> None:
    """在写入前确认快照可被 YAML 安全序列化。"""
    try:
        yaml.safe_dump(dict(data), allow_unicode=True)
    except yaml.YAMLError as exc:
        raise TypeError("配置快照包含无法序列化的值") from exc


__all__ = [
    "RunSnapshot",
    "SHARED_CONFIG_NAME",
    "assert_shared_compatible",
    "load_run_snapshot",
    "read_yaml_dict",
    "write_shared_config",
    "write_yaml_dict",
]