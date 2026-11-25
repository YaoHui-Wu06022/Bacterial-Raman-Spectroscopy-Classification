"""管理训练槽位的可写日志与产物。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ramanv2.experiments.run_context import (
    ANALYSIS_RESULT_DIR_NAME,
    BASELINE_VAL_RESULT_DIR_NAME,
    CHECKPOINT_FILE_NAME,
    DIAGNOSTIC_FILE_NAME,
    METRICS_LOG_FILE_NAME,
    MODEL_FILE_NAME,
    RUN_LOG_FILE_NAME,
    SE_STATS_FILE_NAME,
    TEST_RESULT_DIR_NAME,
    VAL_RESULT_DIR_NAME,
    ExperimentContext,
)


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
        TEST_RESULT_DIR_NAME,
        ANALYSIS_RESULT_DIR_NAME,
    ):
        result_dir = run_dir / name
        if result_dir.is_dir():
            shutil.rmtree(result_dir)
