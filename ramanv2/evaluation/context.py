"""验证集评估共用的实验、数据与输出位置解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ramanv2.core.config import Config, build_config
from ramanv2.core.config_file import SHARED_CONFIG_NAME, read_yaml_dict
from ramanv2.core.hierarchy_meta import load_hierarchy_meta
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.data.index import DatasetIndex
from ramanv2.data.profiles import resolve_training_dir
from ramanv2.training.split import load_split_files


@dataclass(frozen=True)
class EvaluationContext:
    """一次评估固定使用的实验快照、数据索引与 train/val 切分。"""

    experiment_dir: Path
    source_dir: Path
    config: Config
    input_spec: InputSpec
    dataset_index: DatasetIndex
    meta: dict[str, Any]
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class RunEntry:
    """层级元数据中的一个明确模型槽位。"""

    level_name: str
    parent_id: int | None
    values: Mapping[str, Any]
    run_dir: Path


def load_evaluation_context(source_dir: Path | str) -> EvaluationContext:
    """从实验或 run 目录读取评估所需的不可变输入。"""
    source_path = Path(source_dir).resolve()
    experiment_dir = resolve_experiment_dir(source_path)
    config = build_config(read_yaml_dict(experiment_dir / SHARED_CONFIG_NAME))
    dataset_index = DatasetIndex(resolve_training_dir(config.dataset.profile_id))
    split = load_split_files(dataset_index, experiment_dir)
    if split is None:
        raise FileNotFoundError(
            f"实验根缺少 train_split.json/val_split.json：{experiment_dir}"
        )
    meta = load_hierarchy_meta(experiment_dir / "hierarchy_meta.json")
    if meta is None:
        raise FileNotFoundError(f"缺少 hierarchy_meta.json：{experiment_dir}")
    train_indices, validation_indices = split
    return EvaluationContext(
        experiment_dir=experiment_dir,
        source_dir=source_path,
        config=config,
        input_spec=build_input_spec(config.input),
        dataset_index=dataset_index,
        meta=meta,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
    )


def resolve_experiment_dir(source_dir: Path | str) -> Path:
    """从实验根或其子目录向上定位共享配置快照。"""
    source_path = Path(source_dir).resolve()
    for candidate in (source_path, *source_path.parents):
        if (candidate / SHARED_CONFIG_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"无法定位实验根 shared_config.yaml：{source_path}")


def resolve_level_name(context: EvaluationContext, level_value: str) -> str:
    """校验并规范化待评估业务层级名称。"""
    return context.dataset_index.resolve_level_name(level_value)


def resolve_run_entry(context: EvaluationContext) -> RunEntry:
    """从输入目录在层级元数据中定位唯一模型 run。"""
    for level_name, values in (context.meta.get("level_models") or {}).items():
        if not isinstance(values, dict):
            continue
        run_dir = resolve_entry_path(context.experiment_dir, values, "run_dir")
        if run_dir == context.source_dir:
            return RunEntry(str(level_name), None, values, run_dir)
    for level_name, entries in (context.meta.get("parent_models") or {}).items():
        for parent_text, values in (entries or {}).items():
            if not isinstance(values, dict) or not values.get("run_dir"):
                continue
            run_dir = resolve_entry_path(context.experiment_dir, values, "run_dir")
            if run_dir == context.source_dir:
                return RunEntry(str(level_name), int(parent_text), values, run_dir)
    raise FileNotFoundError(f"hierarchy_meta.json 未记录 run：{context.source_dir}")


def resolve_entry_path(
    experiment_dir: Path,
    values: Mapping[str, Any],
    key: str,
) -> Path:
    """将元数据中的相对或绝对路径解析为绝对路径。"""
    value = values.get(key)
    if not value:
        raise FileNotFoundError(f"模型条目缺少 {key}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (experiment_dir / path).resolve()


def resolve_result_dir(
    context: EvaluationContext,
    level_name: str,
    kind: str,
    mode: str,
    run_entry: RunEntry | None = None,
) -> Path:
    """解析模型或 baseline 的既有结果目录槽位。"""
    names = {"model": "val_result", "baseline": "baseline_val_result"}
    if kind not in names:
        raise ValueError(f"未知评估类型：{kind}")
    if mode == "run":
        if run_entry is None:
            raise ValueError("run 模式必须提供 RunEntry")
        return run_entry.run_dir / names[kind]
    roots = {
        "parent-routed": "level_only_result",
        "cascade": "cascade_result",
    }
    if mode not in roots:
        raise ValueError(f"未知评估模式：{mode}")
    return context.experiment_dir / level_name / roots[mode] / names[kind]


def build_used_runs(values: Mapping[str, Any]) -> dict[str, Any]:
    """提取用于模型评估报告的稳定 run 定位字段。"""
    return {
        key: values[key]
        for key in ("run_dir", "model_path", "config_path", "trained_at")
        if values.get(key) is not None
    }
