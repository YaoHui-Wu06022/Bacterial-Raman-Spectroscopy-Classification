from pathlib import Path

import numpy as np

from ramanv2.audit.config import AuditConfig
from ramanv2.audit.workflow import (
    HIERARCHY_RAW_FIELDS,
    HIERARCHY_SIMILARITY_FIELDS,
    build_hierarchy_raw_rows,
    build_hierarchy_similarity_rows,
    run_stage1,
    run_stage2,
    run_stage3,
)
from ramanv2.audit.workspace import build_work_dir
from ramanv2.core.config import InputConfig
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.io import write_arc_data
from ramanv2.data.profiles import DatasetProfile


def test_audit_runs_three_stages_without_shift_or_test_pool(tmp_path: Path) -> None:
    source_dir = tmp_path / "init"
    axis = np.linspace(600.0, 700.0, 101)
    for folder in ("AA01", "AA02"):
        for index in range(9):
            intensity = np.sin(axis / 15.0) + index * 0.001
            write_arc_data(source_dir / "GenusA" / folder / f"sample_{index}.arc_data", axis, intensity)

    config = AuditConfig(
        input=InputConfig(cut_min=600.0, cut_max=700.0, target_points=101, bad_bands=()),
        cleaning=DataBuildConfig(baseline_max_iter=1, cosmic_ray_profile_ids=()),
    )
    profile = DatasetProfile("alldata", "alldata")
    run_dir = tmp_path / "audit_run"
    run_dir.mkdir()
    work_dir = build_work_dir(run_dir, source_dir)
    input_dir = work_dir / "init"

    assert run_stage1(
        input_dir,
        work_dir,
        run_dir,
        folder_depth=2,
        config=config,
        fields=HIERARCHY_RAW_FIELDS,
        build_report_rows=build_hierarchy_raw_rows,
    ) == 18
    assert run_stage2(
        input_dir,
        work_dir,
        run_dir,
        profile,
        config,
        folder_depth=2,
        is_multi_folder_only=True,
        fields=HIERARCHY_SIMILARITY_FIELDS,
        build_report_rows=build_hierarchy_similarity_rows,
    ) == 18
    assert run_stage3(
        input_dir,
        work_dir,
        run_dir,
        profile,
        config,
        folder_depth=2,
        fields=HIERARCHY_SIMILARITY_FIELDS,
        build_report_rows=build_hierarchy_similarity_rows,
    ) == 18
    assert (run_dir / "stage1" / "stage1_raw_scores.csv").is_file()
    assert (run_dir / "stage2" / "stage2_scores.csv").is_file()
    assert (run_dir / "stage2" / "stage2_candidates.csv").is_file()
    assert (run_dir / "stage3" / "stage3_scores.csv").is_file()
    assert (run_dir / "stage3" / "stage3_candidates.csv").is_file()
