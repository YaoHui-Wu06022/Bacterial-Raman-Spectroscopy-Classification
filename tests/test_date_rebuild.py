"""按日期只更新 shift，保留下游文件、待同步状态，并验证事务回滚。"""

import json
import shutil

import numpy as np
import pytest

from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.data.builders.init import build_full_init
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline import calibration
from ramanv2.pipeline.calibration import run_date_rebuild
from ramanv2.pipeline.lineage import load_folder_lineage, record_restore
from ramanv2.pipeline.shift import rebuild_shift_data
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


DATE = "20240101"
OTHER_DATE = "20240102"


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def layout(tmp_path):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    for date, cs_name in ((DATE, "EC03"), (OTHER_DATE, "EC04")):
        for folder in ("EC01", "EC02", cs_name):
            write_arc_data(cosmic / date / folder / "a.arc_data", [600, 1000, 1800], [10, 20, 30])
        report = shift / date / "shift/affine.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"status": "manual_approved", "scale": 2, "offset": 10}), encoding="utf-8")
    assert all(result["status"] == "rebuilt" for result in rebuild_shift_data(cosmic, shift, [DATE, OTHER_DATE]))
    report = build_full_init(shift, init, mapping, [DATE, OTHER_DATE], selected_test_folders=[f"{DATE}/EC03", f"{OTHER_DATE}/EC04"])
    state = PipelineState(affine_completed_dates={DATE, OTHER_DATE}, init_generated_dates={DATE, OTHER_DATE}, dataset_builds={"full_init": report, "profiles": {"GN": {"status": "ready"}}})
    state_file = tmp_path / ".raman_pipeline/state.json"
    save_pipeline_state(state_file, state)
    write_arc_data(tmp_path / "GN/existing.arc_data", [600, 700], [1, 2])
    figure = tmp_path / "fig_init/Escherichia/EC.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"old plot")
    return cosmic, shift, state, state_file


@pytest.mark.parametrize("delete_cs", [False, True])
def test_selected_date_updates_only_shift_preserving_downstream(layout, delete_cs, monkeypatch):
    cosmic, shift, state, state_file = layout
    root = cosmic.parent
    before = {name: _snapshot(root / name) for name in ("init", "init_test", "GN", "fig_init")}
    other_before = _snapshot(shift / OTHER_DATE)
    shutil.rmtree(cosmic / DATE / "EC01")
    write_arc_data(cosmic / DATE / "EC02/a.arc_data", [602, 1002, 1802], [11, 21, 31])
    if delete_cs:
        shutil.rmtree(cosmic / DATE / "EC03")
    record_restore(cosmic, [f"{DATE}/EC02/a.arc_data"])
    state.dataset_builds["full_init"].update(status="pending", pending_dates=[DATE, OTHER_DATE])
    report_before = json.dumps(state.dataset_builds["full_init"], sort_keys=True)
    raw_before = _snapshot(cosmic)

    def unexpected(*args, **kwargs):
        raise AssertionError("按日期更新 shift 不应构建下游数据或图")

    monkeypatch.setattr(calibration, "build_spectrum_folder", unexpected)
    monkeypatch.setattr(calibration, "build_test_folder", unexpected)
    monkeypatch.setattr(calibration, "build_prefix_plots_for_groups", unexpected)
    preserve = calibration.preserve_paths

    def preserve_selected_paths(paths):
        assert set(paths) == {shift / DATE, root / ".raman_pipeline/folder_lineage.json", state_file}
        return preserve(paths)

    monkeypatch.setattr(calibration, "preserve_paths", preserve_selected_paths)
    result = run_date_rebuild(*layout, DATE)
    assert result["status"] == "rebuilt"
    axis, intensity = read_arc_data(shift / DATE / "EC02/a.arc_data")
    np.testing.assert_array_equal(axis, np.array([602, 1002, 1802]) * 2 + 10)
    np.testing.assert_array_equal(intensity, [11, 21, 31])
    assert not (shift / DATE / "EC01").exists()
    assert (shift / DATE / "EC03").is_dir() == (not delete_cs)
    assert _snapshot(cosmic) == raw_before
    assert _snapshot(shift / OTHER_DATE) == other_before
    for name, snapshot in before.items():
        assert _snapshot(root / name) == snapshot
    assert json.dumps(state.dataset_builds["full_init"], sort_keys=True) == report_before
    records = load_folder_lineage(cosmic)["folders"]
    assert records[f"{DATE}/EC02"]["init_pending"]
    assert not records[f"{DATE}/EC03"].get("init_pending")
    assert state.dataset_builds["profiles"] == {"GN": {"status": "ready"}}
    assert not list(root.rglob(".*.tmp"))


















