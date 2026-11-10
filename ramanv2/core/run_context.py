"""实验根与单个模型槽位的目录、快照和日志上下文。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ramanv2.core.config import Config

from ramanv2.core.config_snapshot import (
    SHARED_CONFIG_NAME,
    assert_shared_compatible,
    read_yaml_dict,
    write_shared_config,
)


TRAIN_SPLIT_FILE_NAME = "train_split.json"
VALIDATION_SPLIT_FILE_NAME = "val_split.json"
HIERARCHY_META_FILE_NAME = "hierarchy_meta.json"
RUN_LOG_FILE_NAME = "run.log"
METRICS_LOG_FILE_NAME = "training_metrics.jsonl"
DIAGNOSTIC_FILE_NAME = "numerical_diagnostic.json"
MODEL_FILE_NAME = "model.pt"
SE_STATS_FILE_NAME = "se_stats.pt"
CHECKPOINT_FILE_NAME = "checkpoint.pt"
VAL_RESULT_DIR_NAME = "val_result"
BASELINE_VAL_RESULT_DIR_NAME = "baseline_val_result"
ANALYSIS_RESULT_DIR_NAME = "analysis_result"
INFERENCE_RESULT_DIR_NAME = "inference_result"
METRICS_FILE_NAME = "metrics.json"
CONFUSION_MATRIX_FILE_NAME = "confusion_matrix.png"
UMAP_FIGURE_NAME = "umap_hier_train_val.png"
ANALYSIS_LOG_FILE_NAME = "analysis_log.txt"


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
    """单个模型槽位持有的目录、产物路径和日志文件。"""

    experiment_context: ExperimentContext
    run_dir: Path
    log_dir: Path
    model_path: Path
    se_stats_path: Path
    checkpoint_path: Path
    diagnostic_path: Path
    metrics_path: Path
    log_path: Path
    _log_file: TextIO

    def write_log(self, message: str) -> None:
        """写入一条 run 日志，同时输出到控制台。"""
        print(message)
        self._log_file.write(message + "\n")

    def close(self) -> None:
        """关闭当前 run 打开的日志文件。"""
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
        write_shared_config(shared_config_path, config)
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
    resume_run_dir: Path | str | None = None,
) -> Path:
    """计算模型槽位目录，或校验恢复训练指向的目标槽位。"""
    slot_dir = experiment_context.experiment_dir / level_name
    if parent_id is not None:
        slot_dir = slot_dir / f"{level_name}_{int(parent_id)}"
    if resume_run_dir is None:
        return slot_dir

    target_dir = Path(resume_run_dir).resolve()
    if not target_dir.is_dir():
        raise ValueError(f"恢复目录不存在：{target_dir}")
    if target_dir != slot_dir.resolve():
        raise ValueError(
            f"恢复目录与当前训练槽位不一致：run={target_dir}，期望位于 {slot_dir}"
        )
    return target_dir


def open_run_context(
    experiment_context: ExperimentContext,
    run_dir: Path | str,
    *,
    resume_enable: bool,
) -> RunContext:
    """打开单个模型槽位，清理旧派生结果并写入追加日志。"""
    target_dir = Path(run_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    if not resume_enable:
        _clear_derived_results(target_dir)
    log_dir = target_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / RUN_LOG_FILE_NAME
    log_mode = "a" if resume_enable and log_path.is_file() else "w"
    return RunContext(
        experiment_context=experiment_context,
        run_dir=target_dir,
        log_dir=log_dir,
        model_path=target_dir / MODEL_FILE_NAME,
        se_stats_path=target_dir / SE_STATS_FILE_NAME,
        checkpoint_path=target_dir / CHECKPOINT_FILE_NAME,
        diagnostic_path=log_dir / DIAGNOSTIC_FILE_NAME,
        metrics_path=log_dir / METRICS_LOG_FILE_NAME,
        log_path=log_path,
        _log_file=log_path.open(log_mode, buffering=1, encoding="utf-8"),
    )


def _clear_derived_results(run_dir: Path) -> None:
    """重训已有槽位前删除旧模型对应的派生结果目录。"""
    for name in (
        VAL_RESULT_DIR_NAME,
        BASELINE_VAL_RESULT_DIR_NAME,
        ANALYSIS_RESULT_DIR_NAME,
        INFERENCE_RESULT_DIR_NAME,
    ):
        result_dir = run_dir / name
        if result_dir.is_dir():
            shutil.rmtree(result_dir)
