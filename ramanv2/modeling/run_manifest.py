"""评估与推理共用的模型 run 清单产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ramanv2.common.file_ops import write_json


def resolve_entry_path(
    experiment_dir: Path,
    entry: Mapping[str, Any],
    key: str,
) -> Path:
    """将模型条目中的相对或绝对路径解析为绝对路径。"""
    value = entry.get(key)
    if not value:
        raise FileNotFoundError(f"模型条目缺少 {key}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (experiment_dir / path).resolve()


def build_used_run_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """提取用于评估报告的稳定 run 定位字段。"""
    return {
        key: values[key]
        for key in ("run_dir", "model_path", "config_path", "trained_at")
        if values.get(key) is not None
    }


def has_model_path(entry: Any) -> bool:
    """判断模型槽位是否已经训练并记录了模型文件。"""
    return isinstance(entry, Mapping) and bool(entry.get("model_path"))


def write_run_manifest(
    output_path: Path | str,
    mode: str,
    level_name: str,
    runs: Mapping[str, Any],
) -> None:
    """写出本次评估或推理实际使用的模型 run。"""
    payload = {"mode": mode, "target_level": level_name, "runs": dict(runs)}
    write_json(output_path, payload)


__all__ = [
    "build_used_run_values",
    "has_model_path",
    "resolve_entry_path",
    "write_run_manifest",
]
