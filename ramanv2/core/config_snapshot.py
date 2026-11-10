"""配置快照的 YAML 读写与共享输入校验。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from ramanv2.core.config import Config
from ramanv2.common.hierarchy import validate_label_schema


SHARED_CONFIG_NAME = "shared_config.yaml"


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


def _assert_yaml_ready(data: Mapping[str, Any]) -> None:
    """在写入前确认快照可被 YAML 安全序列化。"""
    try:
        yaml.safe_dump(dict(data), allow_unicode=True)
    except yaml.YAMLError as exc:
        raise TypeError("配置快照包含无法序列化的值") from exc
