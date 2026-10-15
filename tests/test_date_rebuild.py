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
    for date, cs_name in ((DATE, "CS01EC"), (OTHER_DATE, "CS02EC")):
        for folder in ("EC01", "EC02", cs_name):
            write_arc_data(cosmic / date / folder / "a.arc_data", [600, 1000, 1800], [10, 20, 30])
        report = shift / date / "shift/affine.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"status": "manual_approved", "scale": 2, "offset": 10}), encoding="utf-8")
    assert all(result["status"] == "rebuilt" for result in rebuild_shift_data(cosmic, shift, [DATE, OTHER_DATE]))
    report = build_full_init(shift, init, mapping, [DATE, OTHER_DATE], selected_cs_folders=[f"{DATE}/CS01EC", f"{OTHER_DATE}/CS02EC"])
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
    before = {name: _snapshot(root / name) for name in ("init", "CSdata", "GN", "fig_init")}
    other_before = _snapshot(shift / OTHER_DATE)
    shutil.rmtree(cosmic / DATE / "EC01")
    write_arc_data(cosmic / DATE / "EC02/a.arc_data", [602, 1002, 1802], [11, 21, 31])
    if delete_cs:
        shutil.rmtree(cosmic / DATE / "CS01EC")
    record_restore(cosmic, [f"{DATE}/EC02/a.arc_data"])
    state.dataset_builds["full_init"].update(status="pending", pending_dates=[DATE, OTHER_DATE])
    report_before = json.dumps(state.dataset_builds["full_init"], sort_keys=True)
    raw_before = _snapshot(cosmic)

    def unexpected(*args, **kwargs):
        raise AssertionError("按日期更新 shift 不应构建下游数据或图")

    monkeypatch.setattr(calibration, "build_spectrum_folder", unexpected)
    monkeypatch.setattr(calibration, "build_cs_folder", unexpected)
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
    assert (shift / DATE / "CS01EC").is_dir() == (not delete_cs)
    assert _snapshot(cosmic) == raw_before
    assert _snapshot(shift / OTHER_DATE) == other_before
    for name, snapshot in before.items():
        assert _snapshot(root / name) == snapshot
    assert json.dumps(state.dataset_builds["full_init"], sort_keys=True) == report_before
    records = load_folder_lineage(cosmic)["folders"]
    assert records[f"{DATE}/EC02"]["init_pending"]
    assert not records[f"{DATE}/CS01EC"].get("init_pending")
    assert state.dataset_builds["profiles"] == {"GN": {"status": "ready"}}
    assert not list(root.rglob(".*.tmp"))


def test_date_rebuild_works_without_init_manifest_or_classification(layout):
    cosmic, shift, state, state_file = layout
    state.dataset_builds.pop("full_init")
    result = run_date_rebuild(*layout, DATE)
    assert result["status"] == "rebuilt"
    assert "full_init" not in state.dataset_builds


def test_date_rebuild_scans_only_selected_date_and_batches_lineage_writes(layout, monkeypatch):
    from pathlib import Path
    from ramanv2.pipeline import lineage

    cosmic, shift, state, state_file = layout
    before = load_folder_lineage(cosmic)["folders"]
    other_records = {key: value for key, value in before.items() if value["date"] == OTHER_DATE}
    rglob = Path.rglob
    save = lineage.save_folder_lineage
    writes = []

    def scoped_rglob(path, pattern):
        if path.is_relative_to(cosmic):
            assert OTHER_DATE not in path.relative_to(cosmic).parts
        return rglob(path, pattern)

    def save_record(root, payload):
        writes.append(root)
        return save(root, payload)

    monkeypatch.setattr(Path, "rglob", scoped_rglob)
    monkeypatch.setattr(lineage, "save_folder_lineage", save_record)
    run_date_rebuild(*layout, DATE)
    records = load_folder_lineage(cosmic)["folders"]
    assert {key: records[key] for key in other_records} == other_records
    assert len(writes) == 2
    for name in ("EC01", "EC02", "CS01EC"):
        assert records[f"{DATE}/{name}"]["outputs"]["shift"] == str(shift / DATE / name)


@pytest.mark.parametrize("failure", ["shift", "state"])
def test_date_update_failure_restores_shift_and_state(layout, monkeypatch, failure):
    cosmic, shift, state, state_file = layout
    write_arc_data(cosmic / DATE / "EC01/a.arc_data", [604, 1004, 1804], [13, 23, 33])
    before = _snapshot(cosmic.parent)
    state_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("injected date update failure")

    if failure == "shift":
        monkeypatch.setattr(calibration, "rebuild_shift_data", lambda *a: [{"status": "error", "error": "injected date update failure"}])
    else:
        monkeypatch.setattr(calibration, "save_pipeline_state", fail)
    with pytest.raises((OSError, RuntimeError), match="injected date update failure"):
        run_date_rebuild(*layout, DATE)
    assert _snapshot(cosmic.parent) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == state_before


def test_date_rebuild_ui_updates_shift_without_loading_classification(layout):
    from streamlit.testing.v1 import AppTest

    cosmic, shift, state, state_file = layout
    before = _snapshot(shift / DATE)
    init_before = _snapshot(cosmic.parent / "init")
    write_arc_data(cosmic / OTHER_DATE / "EC01/a.arc_data", [605, 1005, 1805], [14, 24, 34])
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.ui.pages.affine_calibration import render_affine_calibration_page
root = Path({str(cosmic.parent)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", shift_root=root / "shift_data", init_root=root / "init", state_file=root / ".raman_pipeline/state.json")
render_affine_calibration_page(context)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    app.selectbox(key="affine_rebuild_date").select(OTHER_DATE).run()
    app.button(key="affine_rebuild_date_confirm").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any(OTHER_DATE in message.value and "shift_data" in message.value for message in app.success)
    np.testing.assert_array_equal(read_arc_data(shift / OTHER_DATE / "EC01/a.arc_data")[0], np.array([605, 1005, 1805]) * 2 + 10)
    assert _snapshot(cosmic.parent / "init") == init_before
    assert _snapshot(shift / DATE) == before


def test_date_change_hints_distinguish_deletion_restore_and_shift_update(layout):
    from ramanv2.pipeline.lineage import record_manual_move
    from ramanv2.ui.pages.affine_calibration import _collect_folder_changes

    cosmic, shift, state, state_file = layout
    record_manual_move(cosmic, [f"{DATE}/EC01/a.arc_data"], "manual_selection")
    record_restore(cosmic, [f"{OTHER_DATE}/EC01/a.arc_data"])
    rows = _collect_folder_changes(cosmic)
    by_date = {row["date"]: row for row in rows}
    assert len(rows) == 2
    assert {row["folder"] for row in rows} == {"EC01"}
    assert rows[0]["date"] == OTHER_DATE
    assert by_date[DATE]["operation"] == "删除光谱"
    assert by_date[OTHER_DATE]["operation"] == "恢复光谱"
    assert by_date[DATE]["is_shift_pending"]
    assert by_date[OTHER_DATE]["is_shift_pending"]

    run_date_rebuild(*layout, DATE)
    updated_rows = _collect_folder_changes(cosmic)
    by_date = {row["date"]: row for row in updated_rows}
    assert by_date[DATE]["operation"] == "删除光谱"
    assert by_date[DATE]["changed_at"] == next(row for row in rows if row["date"] == DATE)["changed_at"]
    assert updated_rows[0]["date"] == OTHER_DATE
    assert not by_date[DATE]["is_shift_pending"]
    assert by_date[OTHER_DATE]["is_shift_pending"]


def test_single_folder_shift_update_does_not_clear_sibling_hint(layout):
    from ramanv2.pipeline.shift import rebuild_shift_folder
    from ramanv2.ui.pages.affine_calibration import _collect_folder_changes

    cosmic, shift, state, state_file = layout
    record_restore(cosmic, [f"{DATE}/EC01/a.arc_data", f"{DATE}/EC02/a.arc_data"])
    rebuild_shift_folder(cosmic, shift, DATE, "EC01")
    rows = {row["source_id"]: row for row in _collect_folder_changes(cosmic)}
    assert not rows[f"{DATE}/EC01"]["is_shift_pending"]
    assert rows[f"{DATE}/EC02"]["is_shift_pending"]
    rebuild_shift_folder(cosmic, shift, DATE, "EC02")
    assert all(not row["is_shift_pending"] for row in _collect_folder_changes(cosmic))


def test_date_rebuild_clears_deleted_folder_shift_hint(layout):
    from ramanv2.pipeline.lineage import record_folder_move
    from ramanv2.ui.pages.affine_calibration import _collect_folder_changes

    cosmic, shift, state, state_file = layout
    record_folder_move(cosmic, [f"{DATE}/EC02/a.arc_data"])
    shutil.rmtree(cosmic / DATE / "EC02")
    assert next(row for row in _collect_folder_changes(cosmic) if row["date"] == DATE)["is_shift_pending"]
    run_date_rebuild(*layout, DATE)
    assert not next(row for row in _collect_folder_changes(cosmic) if row["date"] == DATE)["is_shift_pending"]


def test_date_change_hints_render_read_only_and_refresh_after_update(layout):
    from streamlit.testing.v1 import AppTest

    cosmic, shift, state, state_file = layout
    record_restore(cosmic, [f"{DATE}/EC01/a.arc_data"])
    before = _snapshot(cosmic.parent)
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.pipeline.state import load_pipeline_state
from ramanv2.ui.pages.affine_calibration import _render_date_rebuild_section
root = Path({str(cosmic.parent)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", shift_root=root / "shift_data", state_file=root / ".raman_pipeline/state.json")
_render_date_rebuild_section(context, load_pipeline_state(context.state_file))
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert any(DATE in item.value for item in app.warning)
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["文件夹"].tolist() == ["EC01"]
    assert app.dataframe[0].value["操作"].tolist() == ["恢复光谱"]
    assert _snapshot(cosmic.parent) == before
    app.selectbox(key="affine_rebuild_date").select(DATE).run()
    app.button(key="affine_rebuild_date_confirm").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert not app.warning
    assert app.dataframe[0].value["操作"].tolist() == ["恢复光谱"]
    assert app.dataframe[0].value["shift 状态"].tolist() == ["已更新"]


def test_folder_change_hints_keep_delete_and_restore_batches(layout):
    from ramanv2.pipeline.lineage import record_manual_move
    from ramanv2.ui.pages.affine_calibration import _collect_folder_changes

    cosmic, shift, state, state_file = layout
    assert _collect_folder_changes(cosmic) == []
    record_manual_move(cosmic, [f"{DATE}/EC01/a.arc_data", f"{DATE}/EC01/b.arc_data"], "manual_selection")
    record_restore(cosmic, [f"{DATE}/EC01/a.arc_data"])
    rows = _collect_folder_changes(cosmic)
    assert len(rows) == 2
    assert [row["operation"] for row in rows] == ["恢复光谱", "删除光谱"]
    assert all(row["date"] == DATE and row["folder"] == "EC01" for row in rows)
