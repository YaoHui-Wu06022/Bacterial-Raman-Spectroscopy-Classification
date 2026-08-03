"""实验根与单个模型 run 的目录、快照和日志上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from raman_temp.core.config import Config

from raman_temp.core.config_file import (
    MODEL_CONFIG_NAME,
    RESOLVED_CONFIG_NAME,
    SHARED_CONFIG_NAME,
    assert_shared_compatible,
    read_yaml_dict,
    save_model_config,
    save_resolved_config,
    save_shared_config,
)


TRAIN_SPLIT_FILE_NAME = "train_split.json"
VALIDATION_SPLIT_FILE_NAME = "val_split.json"
HIERARCHY_META_FILE_NAME = "hierarchy_meta.json"
RUN_LOG_FILE_NAME = "run.log"
CONFIG_LOG_FILE_NAME = "config.txt"


@dataclass(frozen=True)
class ExperimentContext:
    """一个实验根目录持有的共享产物路径。"""

    experiment_dir: Path
    shared_config_path: Path
    train_split_path: Path
    validation_split_path: Path
    hierarchy_meta_path: Path


@dataclass
class RunContext:
    """单个模型 run 持有的目录、产物路径和日志文件。"""

    experiment_context: ExperimentContext
    run_dir: Path
    log_dir: Path
    model_config_path: Path
    resolved_config_path: Path
    model_path: Path
    se_stats_path: Path
    checkpoint_path: Path
    diagnostic_path: Path
    log_path: Path
    config_log_path: Path
    _log_file: TextIO
    _config_log_file: TextIO

    def write_log(self, message: str) -> None:
        """写入一条 run 日志，同时输出到控制台。"""
        print(message)
        self._log_file.write(message + "\n")

    def close(self) -> None:
        """关闭当前 run 打开的日志文件。"""
        self._config_log_file.close()
        self._log_file.close()


def open_experiment_context(
    experiment_dir: Path | str,
    config: Config,
) -> ExperimentContext:
    """打开实验根目录，并创建或校验共享配置快照。"""
    target_dir = Path(experiment_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    shared_config_path = target_dir / SHARED_CONFIG_NAME
    if shared_config_path.is_file():
        assert_shared_compatible(config, read_yaml_dict(shared_config_path))
    else:
        save_shared_config(shared_config_path, config)
    return ExperimentContext(
        experiment_dir=target_dir,
        shared_config_path=shared_config_path,
        train_split_path=target_dir / TRAIN_SPLIT_FILE_NAME,
        validation_split_path=target_dir / VALIDATION_SPLIT_FILE_NAME,
        hierarchy_meta_path=target_dir / HIERARCHY_META_FILE_NAME,
    )


def resolve_run_dir(
    experiment_context: ExperimentContext,
    level_name: str,
    parent_id: int | None,
    run_name: str,
    resume_run_dir: Path | str | None = None,
) -> Path:
    """计算新 run 目录，或校验恢复训练的 run 位于目标模型槽位内。"""
    slot_dir = experiment_context.experiment_dir / level_name
    if parent_id is not None:
        slot_dir = slot_dir / f"{level_name}_{int(parent_id)}"
    if resume_run_dir is None:
        return slot_dir / run_name

    target_dir = Path(resume_run_dir).resolve()
    if not target_dir.is_dir() or not target_dir.name.startswith("run_"):
        raise ValueError(f"恢复目录必须是已有 run_* 目录：{target_dir}")
    if target_dir.parent != slot_dir.resolve():
        raise ValueError(
            f"恢复目录与当前训练槽位不一致：run={target_dir}，期望位于 {slot_dir}"
        )
    return target_dir


def open_run_context(
    experiment_context: ExperimentContext,
    run_dir: Path | str,
    model_tag: str,
    config: Config,
    resume_enable: bool,
    task_values: Mapping[str, Any] | None = None,
) -> RunContext:
    """打开单个 run，并写入配置快照与可追加日志。"""
    target_dir = Path(run_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_dir = target_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / RUN_LOG_FILE_NAME
    log_mode = "a" if resume_enable and log_path.is_file() else "w"
    config_log_path = log_dir / CONFIG_LOG_FILE_NAME
    config_log_mode = "a" if resume_enable and config_log_path.is_file() else "w"
    path_values = {
        "experiment_dir": str(experiment_context.experiment_dir),
        "run_dir": str(target_dir),
    }
    save_model_config(
        target_dir / MODEL_CONFIG_NAME,
        config,
        task_values=task_values,
    )
    save_resolved_config(
        target_dir / RESOLVED_CONFIG_NAME,
        config,
        task_values=task_values,
        path_values=path_values,
    )
    context = RunContext(
        experiment_context=experiment_context,
        run_dir=target_dir,
        log_dir=log_dir,
        model_config_path=target_dir / MODEL_CONFIG_NAME,
        resolved_config_path=target_dir / RESOLVED_CONFIG_NAME,
        model_path=target_dir / f"{model_tag}_model.pt",
        se_stats_path=target_dir / f"{model_tag}_se_stats.pt",
        checkpoint_path=target_dir / f"{model_tag}_checkpoint.pt",
        diagnostic_path=log_dir / f"{model_tag}_numerical_diagnostic.json",
        log_path=log_path,
        config_log_path=config_log_path,
        _log_file=log_path.open(log_mode, buffering=1, encoding="utf-8"),
        _config_log_file=config_log_path.open(
            config_log_mode,
            buffering=1,
            encoding="utf-8",
        ),
    )
    _write_config_log(context, config, path_values, task_values)
    return context


def _write_config_log(
    run_context: RunContext,
    config: Config,
    path_values: Mapping[str, str],
    task_values: Mapping[str, Any] | None,
) -> None:
    """写入当前 run 的路径信息和完整配置字段。"""
    run_context._config_log_file.write("===== Run Meta =====\n")
    run_context._config_log_file.write(
        f"Experiment dir: {path_values['experiment_dir']}\n"
    )
    run_context._config_log_file.write(f"Run dir: {path_values['run_dir']}\n")
    run_context._config_log_file.write("=====================\n")
    for key, value in config.to_dict().items():
        run_context._config_log_file.write(f"{key}: {value}\n")
    for key, value in (task_values or {}).items():
        run_context._config_log_file.write(f"{key}: {value}\n")
