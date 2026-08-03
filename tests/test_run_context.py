from __future__ import annotations

from pathlib import Path

from raman_temp.core.config import build_config
from raman_temp.core.config_file import read_yaml_dict
from raman_temp.core.run_context import (
    open_experiment_context,
    open_run_context,
    resolve_run_dir,
)


def test_run_context_writes_snapshots_without_mutating_config(tmp_path: Path) -> None:
    config = build_config()
    experiment_context = open_experiment_context(tmp_path / "experiment", config)
    run_dir = resolve_run_dir(experiment_context, "level_2", 3, "run_20260803_120000")
    run_context = open_run_context(
        experiment_context,
        run_dir,
        "level_2_3",
        config,
        resume_enable=False,
    )
    run_context.write_log("task started")
    run_context.close()

    assert run_context.model_path.name == "level_2_3_model.pt"
    assert run_context.se_stats_path.name == "level_2_3_se_stats.pt"
    assert run_context.checkpoint_path.name == "level_2_3_checkpoint.pt"
    assert run_context.diagnostic_path.name == "level_2_3_numerical_diagnostic.json"
    assert run_context.log_path.read_text(encoding="utf-8") == "task started\n"
    resolved = read_yaml_dict(run_context.resolved_config_path)
    assert resolved["experiment_dir"] == str(experiment_context.experiment_dir)
    assert resolved["run_dir"] == str(run_dir)


def test_run_context_checks_shared_config_and_resume_slot(tmp_path: Path) -> None:
    config = build_config()
    experiment_context = open_experiment_context(tmp_path / "experiment", config)
    run_dir = resolve_run_dir(experiment_context, "level_1", None, "run_20260803_120000")
    run_dir.mkdir(parents=True)
    assert resolve_run_dir(
        experiment_context,
        "level_1",
        None,
        "ignored",
        resume_run_dir=run_dir,
    ) == run_dir.resolve()
