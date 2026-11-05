"""按日期从 shift 更新 init/init_test，验证检测、分流保持与全量摘要合并。"""

import json
import shutil
from pathlib import Path

import pytest

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.builders.init import build_full_init
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.pipeline import dataset_sync
from ramanv2.pipeline.dataset_sync import collect_date_update_candidates, run_date_dataset_update
from ramanv2.pipeline.lineage import load_folder_lineage, record_restore
from ramanv2.pipeline.calibration import rebuild_shift_data
from ramanv2.pipeline.state import PipelineState, write_pipeline_state


DATE = "20240101"
NEW_DATE = "20240102"


def _write_affine(shift_root: Path, date_name: str) -> None:
    report = shift_root / date_name / "shift/affine.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"status": "manual_approved", "scale": 1, "offset": 0}), encoding="utf-8")


def _write_date(cosmic: Path, date_name: str, folder_files: dict[str, tuple[str, ...]]) -> None:
    for folder_name, names in folder_files.items():
        for name in names:
            write_arc_data(cosmic / date_name / folder_name / name, [400, 1000, 2000], [10, 20, 30])


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def layout(tmp_path, monkeypatch):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    _write_date(cosmic, DATE, {"EC01": ("a.arc_data", "b.arc_data"), "EC02": ("a.arc_data", "b.arc_data"), "EC03": ("a.arc_data", "b.arc_data")})
    _write_affine(shift, DATE)
    rebuild_shift_data(cosmic, shift, [DATE])
    report = build_full_init(shift, init, mapping, [DATE], selected_test_folders=[f"{DATE}/EC03"])
    state = PipelineState(
        affine_completed_dates={DATE},
        init_generated_dates={DATE},
        dataset_builds={"full_init": report},
    )
    state_file = tmp_path / ".raman_pipeline/state.json"
    write_pipeline_state(state_file, state)
    monkeypatch.setattr(dataset_sync, "build_prefix_plots", lambda *args, **kwargs: [])
    return cosmic, shift, init, mapping, state, state_file


def test_new_date_update_merges_report_and_keeps_existing_dates(layout):
    cosmic, shift, init, mapping, state, state_file = layout
    _write_date(cosmic, NEW_DATE, {"EC01": ("a.arc_data",), "EC02": ("a.arc_data", "b.arc_data"), "EC03": ("a.arc_data",)})
    _write_affine(shift, NEW_DATE)
    rebuild_shift_data(cosmic, shift, [NEW_DATE])
    old_snapshot = _snapshot(init)

    candidates = collect_date_update_candidates(cosmic, shift, init, state.dataset_builds["full_init"])
    candidate = next(item for item in candidates if item["date"] == NEW_DATE)
    assert candidate["status"] == "new"
    assert candidate["is_new"] is True
    assert candidate["file_count"] == 4
    assert candidate["init_folder_count"] == 3
    assert candidate["test_folder_count"] == 0

    result = run_date_dataset_update(
        cosmic, shift, init, mapping, state, state_file,
        [NEW_DATE], [f"{NEW_DATE}/EC03"],
    )

    assert result["new_dates"] == [NEW_DATE]
    assert result["changed_dates"] == []
    assert result["spectrum_count"] == 4
    assert result["plot_error"] is None
    new_snapshot = _snapshot(init)
    assert {key: value for key, value in new_snapshot.items() if key.startswith("Escherichia/EC20240101_")} == {
        key: value for key, value in old_snapshot.items() if key.startswith("Escherichia/EC20240101_")
    }
    test_root = init.parent / "init_test"
    assert (test_root / "EC20240102_03").is_dir()
    report = state.dataset_builds["full_init"]
    assert report["source_dates"] == [DATE, NEW_DATE]
    assert report["status"] == "ready"
    assert report["pending_dates"] == []
    assert {entry["source"] for entry in report["test_exports"]} == {f"{DATE}/EC03", f"{NEW_DATE}/EC03"}
    assert report["output_files"] == len(list(init.rglob("*.arc_data"))) + len(list(test_root.rglob("*.arc_data")))
    lineage = load_folder_lineage(cosmic)["folders"]
    for source_id in (f"{NEW_DATE}/EC01", f"{NEW_DATE}/EC02", f"{NEW_DATE}/EC03"):
        assert lineage[source_id]["outputs"]["init"] or lineage[source_id]["outputs"]["test"]
        assert not lineage[source_id].get("init_pending")


def test_changed_date_rebuilds_only_selected_date_and_preserves_test_route(layout):
    cosmic, shift, init, mapping, state, state_file = layout
    old_route = next(
        entry["target"] for entry in state.dataset_builds["full_init"]["test_exports"]
        if entry["source"] == f"{DATE}/EC03"
    )
    old_ec01 = _snapshot(Path(load_folder_lineage(cosmic)["folders"][f"{DATE}/EC03"]["outputs"]["test"]))
    (cosmic / DATE / "EC02" / "a.arc_data").unlink()
    write_arc_data(cosmic / DATE / "EC01" / "restored.arc_data", [400, 1000, 2000], [11, 21, 31])
    write_arc_data(shift / DATE / "EC01" / "restored.arc_data", [402, 1002, 2002], [11, 21, 31])
    record_restore(cosmic, [f"{DATE}/EC01/restored.arc_data"])

    candidates = collect_date_update_candidates(cosmic, shift, init, state.dataset_builds["full_init"])
    candidate = next(item for item in candidates if item["date"] == DATE)
    assert candidate["status"] == "changed"
    assert f"{DATE}/EC01" in candidate["pending_folders"]
    assert f"{DATE}/EC02" in candidate["drifted_folders"]
    assert candidate["init_folder_count"] == 2
    assert candidate["test_folder_count"] == 1

    result = run_date_dataset_update(cosmic, shift, init, mapping, state, state_file, [DATE])

    assert result["changed_dates"] == [DATE]
    assert result["spectrum_count"] == 6
    lineage = load_folder_lineage(cosmic)["folders"]
    ec01_target = Path(lineage[f"{DATE}/EC01"]["outputs"]["init"])
    ec02_target = Path(lineage[f"{DATE}/EC02"]["outputs"]["init"])
    assert {path.name for path in ec01_target.glob("*.arc_data")} == {"a.arc_data", "b.arc_data", "restored.arc_data"}
    assert {path.name for path in ec02_target.glob("*.arc_data")} == {"b.arc_data"}
    report = state.dataset_builds["full_init"]
    export = next(entry for entry in report["test_exports"] if entry["source"] == f"{DATE}/EC03")
    assert export["target"] == old_route
    assert export["file_count"] == 2
    assert old_ec01 == _snapshot(Path(lineage[f"{DATE}/EC03"]["outputs"]["test"]))
    assert ec01_target.is_dir()
    assert not (init / "Escherichia" / old_route).exists()
    assert report["source_dates"] == [DATE]
    assert report["status"] == "ready"
    assert report["pending_dates"] == []
    assert DATE in state.init_generated_dates
    assert report["date_stats"][DATE]["output_files"] == 6


def test_candidate_reports_shift_missing_before_update(layout):
    cosmic, shift, init, mapping, state, state_file = layout
    (shift / DATE / "EC02" / "a.arc_data").unlink()

    candidates = collect_date_update_candidates(cosmic, shift, init, state.dataset_builds["full_init"])
    candidate = next(item for item in candidates if item["date"] == DATE)
    assert candidate["status"] == "shift_missing"
    assert candidate["shift_missing_folders"] == (f"{DATE}/EC02",)

    with pytest.raises(ValueError, match="shift_data 缺少活动光谱"):
        run_date_dataset_update(cosmic, shift, init, mapping, state, state_file, [DATE])


def test_deleted_date_removes_outputs_and_clears_report_entries(layout):
    cosmic, shift, init, mapping, state, state_file = layout
    archive = cosmic / "delete/folder_delete"
    archive.mkdir(parents=True)
    shutil.move(str(cosmic / DATE), str(archive / DATE))

    candidates = collect_date_update_candidates(cosmic, shift, init, state.dataset_builds["full_init"])
    candidate = next(item for item in candidates if item["date"] == DATE)
    assert candidate["status"] == "changed"
    assert candidate["folder_count"] == 0
    assert candidate["init_folder_count"] == 0
    assert candidate["test_folder_count"] == 0

    result = run_date_dataset_update(cosmic, shift, init, mapping, state, state_file, [DATE])

    assert result["changed_dates"] == [DATE]
    assert result["spectrum_count"] == 0
    assert not list(init.rglob("*.arc_data"))
    assert not list((init.parent / "init_test").rglob("*.arc_data"))
    report = state.dataset_builds["full_init"]
    assert report["test_exports"] == []
    assert report["output_files"] == 0
    assert report["date_stats"][DATE]["output_files"] == 0
    assert report["status"] == "ready"
