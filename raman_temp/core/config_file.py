"""配置快照的 YAML 读写与共享输入校验。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from raman_temp.core.config import Config


SHARED_CONFIG_NAME = "shared_config.yaml"
MODEL_CONFIG_NAME = "model_config.yaml"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"


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


def save_shared_config(path: Path | str, config: Config) -> None:
    """保存实验根共享的 profile 与输入快照。"""
    write_yaml_dict(path, config.to_shared_dict())


def save_model_config(
    path: Path | str,
    config: Config,
    task_values: Mapping[str, Any] | None = None,
) -> None:
    """保存单个模型 run 的模型、训练与任务范围快照。"""
    write_yaml_dict(path, config.to_model_dict() | dict(task_values or {}))


def save_resolved_config(
    path: Path | str,
    config: Config,
    task_values: Mapping[str, Any] | None = None,
    path_values: Mapping[str, Any] | None = None,
) -> None:
    """保存完整配置、任务范围与已解析路径快照。"""
    data = config.to_dict() | dict(task_values or {}) | dict(path_values or {})
    write_yaml_dict(path, data)


def assert_shared_compatible(config: Config, shared_values: Mapping[str, Any]) -> None:
    """确认当前输入快照与实验根已有快照一致。"""
    current = config.to_shared_dict()
    mismatches = [
        key
        for key, value in current.items()
        if value != shared_values.get(key)
    ]
    if mismatches:
        raise ValueError(f"实验根共享配置不一致：{', '.join(mismatches)}")


def _assert_yaml_ready(data: Mapping[str, Any]) -> None:
    """在写入前确认快照可被 YAML 安全序列化。"""
    try:
        yaml.safe_dump(dict(data), allow_unicode=True)
    except yaml.YAMLError as exc:
        raise TypeError("配置快照包含无法序列化的值") from exc
