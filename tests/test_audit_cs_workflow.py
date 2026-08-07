from pathlib import Path

import numpy as np

from ramanv2.core.config import InputConfig
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.io import write_arc_data
from ramanv2.data.profiles import DatasetProfile
from ramanv2.audit.workflow import run_cs_clean_dir


def test_audit_cs_workflow_runs_stage1_and_folder_stage2_without_labels(tmp_path: Path) -> None:
    profile = DatasetProfile("test", "test")
    axis = np.linspace(600.0, 700.0, 101)
    for index in range(9):
        intensity = np.sin(axis / 15.0) + index * 0.001
        write_arc_data(tmp_path / "init" / "CS01" / f"sample_{index}.arc_data", axis, intensity)

    run_dir = run_cs_clean_dir(
        profile,
        tmp_path,
        InputConfig(cut_min=600.0, cut_max=700.0, target_points=101, bad_bands=()),
        DataBuildConfig(baseline_max_iter=1, cosmic_ray_profile_ids=()),
    )

    assert (run_dir / "stage1" / "stage1_raw_scores.csv").is_file()
    assert (run_dir / "stage2" / "stage2_scores.csv").is_file()
    assert (run_dir / "stage2" / "stage2_candidates.csv").is_file()
    assert not (run_dir / "stage3").exists()
    header = (run_dir / "stage2" / "stage2_scores.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "genus" not in header
