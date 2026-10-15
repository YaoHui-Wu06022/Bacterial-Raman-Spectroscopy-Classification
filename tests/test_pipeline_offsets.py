"""补偿到 init 的事务边界、下游隔离及页面输入语义。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.data.builders.init import build_full_init
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline import calibration
from ramanv2.pipeline.calibration import run_folder_offset
from ramanv2.pipeline.shift import rebuild_shift_data
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


DATE = "20240101"


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _layout(tmp_path: Path, cs_folder="CS01EC", export_cs=True):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    axis = np.linspace(400, 2000, 81)
    intensity = 100 + np.sin(axis / 20)
    for relative in (f"{DATE}/EC01/a.arc_data", f"delete/manual/{DATE}/EC01/b.arc_data", f"delete/folder_delete/{DATE}/EC01/c.arc_data", f"{DATE}/{cs_folder}/test.arc_data"):
        write_arc_data(cosmic / relative, axis, intensity, fmt="%.17g")
    report = shift / DATE / "shift" / "affine.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "manual_approved", "scale": 2.0, "offset": 10.0, "folder_compensations": {"EC01": 99}}), encoding="utf-8")
    mapping = DatasetMapping(genera={"Escherichia": ("EC", "CST")}, datasets={"GN": ("Escherichia",)})
    assert rebuild_shift_data(cosmic, shift, [DATE])[0]["status"] == "rebuilt"
    audit = shift / "audit_runs" / "run" / "summary.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"status": "applied", "folders": [f"{DATE}/EC01"]}), encoding="utf-8")
    full_report = build_full_init(shift, init, mapping, [DATE], selected_cs_folders=[f"{DATE}/{cs_folder}"] if export_cs else [])
    state = PipelineState(affine_completed_dates={DATE}, init_generated_dates={DATE}, dataset_builds={"full_init": full_report, "profiles": {"GN": {"status": "ready"}}})
    state_file = tmp_path / ".raman_pipeline" / "state.json"
    save_pipeline_state(state_file, state)
    write_arc_data(tmp_path / "GN" / "init" / "old.arc_data", axis, intensity)
    figure_root = tmp_path / "fig_init"
    figure_root.mkdir()
    (figure_root / "old.png").write_bytes(b"old plot")
    return cosmic, shift, init, mapping, state, state_file, figure_root


def test_offset_updates_sources_and_init_preserving_audit_cs_and_profiles(tmp_path, monkeypatch):
    args = _layout(tmp_path)
    cosmic, shift, init, mapping, state, state_file, figures = args
    audit_before = (shift / "audit_runs" / "run" / "summary.json").read_bytes()
    cs_before, gn_before = _snapshot(tmp_path / "CSdata"), _snapshot(tmp_path / "GN")
    original_axis, original_intensity = read_arc_data(cosmic / DATE / "EC01" / "a.arc_data")
    plotted = []

    def plot(source, output):
        plotted.append(Path(source))
        (output / "new.png").write_bytes(b"updated init plot")

    monkeypatch.setattr(calibration, "build_prefix_plots", plot)
    result = run_folder_offset(cosmic, shift, init, mapping, state, state_file, f"{DATE}/EC01", 2.5, figures)
    assert result == {"changed_folders": ["EC01"], "spectrum_count": 3}
    for relative in (f"{DATE}/EC01/a.arc_data", f"delete/manual/{DATE}/EC01/b.arc_data", f"delete/folder_delete/{DATE}/EC01/c.arc_data"):
        axis, intensity = read_arc_data(cosmic / relative)
        np.testing.assert_array_equal(axis, original_axis + 2.5)
        np.testing.assert_array_equal(intensity, original_intensity)
    files = list(init.rglob("*.arc_data"))
    assert [path.name for path in files] == ["a.arc_data"]
    np.testing.assert_array_equal(read_arc_data(files[0])[0], (original_axis + 2.5) * 2 + 10)
    np.testing.assert_array_equal(read_arc_data(files[0])[1], original_intensity)
    assert (shift / "audit_runs" / "run" / "summary.json").read_bytes() == audit_before
    assert _snapshot(tmp_path / "CSdata") == cs_before
    assert _snapshot(tmp_path / "GN") == gn_before
    assert state.dataset_builds["profiles"] == {"GN": {"status": "ready"}}
    assert plotted == [init]
    saved = state_file.read_text(encoding="utf-8")
    assert "affine_compensations" not in saved
    assert "folder_compensations" not in (shift / DATE / "shift" / "affine.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["read", "shift", "init", "plot", "state"])
def test_offset_failure_restores_sources_outputs_and_state(tmp_path, monkeypatch, failure):
    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path)
    before = _snapshot(tmp_path)
    state_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure == "read":
        from ramanv2.data.calibration import affine
        monkeypatch.setattr(affine, "read_arc_data", fail)
    elif failure == "shift":
        monkeypatch.setattr(calibration, "rebuild_shift_data", lambda *a, **kw: [{"status": "error", "error": "injected failure"}])
    elif failure == "init":
        monkeypatch.setattr(calibration, "build_full_init", fail)
    elif failure == "plot":
        monkeypatch.setattr(calibration, "build_prefix_plots", fail)
    else:
        monkeypatch.setattr(calibration, "build_prefix_plots", lambda *a: [])
        monkeypatch.setattr(calibration, "save_pipeline_state", fail)
    with pytest.raises((OSError, RuntimeError), match="injected failure"):
        run_folder_offset(cosmic, shift, init, mapping, state, state_file, f"{DATE}/EC01", 2, figures)
    assert _snapshot(tmp_path) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == state_before


def test_offset_refuses_to_drop_an_unready_init_date(tmp_path, monkeypatch):
    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path)
    write_arc_data(cosmic / "20240102" / "EC01" / "other.arc_data", [600, 700], [1, 2])
    state.dataset_builds["full_init"]["source_dates"].append("20240102")
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="20240102"):
        run_folder_offset(cosmic, shift, init, mapping, state, state_file, f"{DATE}/EC01", 2, figures)
    assert _snapshot(tmp_path) == before


def test_offset_ui_resets_to_zero_and_calls_pipeline(tmp_path, monkeypatch):
    from ramanv2.ui.pages import dataset_acquisition

    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path)
    session = {"folder_offset_reset": True, "folder_offset_delta": 9.0}
    seen = []

    def number_input(label, **kwargs):
        seen.append(session[kwargs["key"]])
        return session[kwargs["key"]]

    fake = SimpleNamespace(session_state=session, subheader=lambda *a: None, info=lambda *a: None, success=lambda *a: None, selectbox=lambda label, options, **kwargs: f"{DATE}/EC01", number_input=number_input, button=lambda *a, **kw: True)
    monkeypatch.setattr(dataset_acquisition, "st", fake)
    called = []
    monkeypatch.setattr(dataset_acquisition, "run_folder_offset", lambda *args: called.append(args) or {"changed_folders": []})
    context = SimpleNamespace(data_root=cosmic, shift_root=shift, init_root=init, state_file=state_file)
    dataset_acquisition._render_affine_compensation(context, state, mapping)
    assert seen == [0.0]
    assert called[0][6:8] == (f"{DATE}/EC01", 0.0)


@pytest.mark.parametrize("selected_folder", ["EC01", "CS01EC"])
def test_streamlit_offset_button_runs_and_resets_input(tmp_path, selected_folder):
    from streamlit.testing.v1 import AppTest

    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path)
    gn_before = _snapshot(tmp_path / "GN")
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.state import load_pipeline_state
from ramanv2.ui.pages.dataset_acquisition import _render_affine_compensation
context = SimpleNamespace(data_root=Path({str(cosmic)!r}), shift_root=Path({str(shift)!r}), init_root=Path({str(init)!r}), state_file=Path({str(state_file)!r}))
mapping = DatasetMapping(genera={{"Escherichia": ("EC",)}}, datasets={{"GN": ("Escherichia",)}})
_render_affine_compensation(context, load_pipeline_state(context.state_file), mapping)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert len(app.selectbox) == 1
    assert app.number_input[0].value == 0
    app.selectbox[0].select(f"{DATE}/{selected_folder}").run(timeout=30)
    app.number_input[0].set_value(2.0)
    app.button(key="apply_affine_compensation").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.number_input[0].value == 0
    if selected_folder == "EC01":
        assert len(list(figures.rglob("*.png"))) >= 2
    else:
        assert len(list(figures.rglob("*.png"))) == 1
        source_axis = read_arc_data(cosmic / DATE / selected_folder / "test.arc_data")[0]
        target_axis = read_arc_data(tmp_path / "CSdata" / selected_folder / "test.arc_data")[0]
        np.testing.assert_array_equal(target_axis, source_axis * 2 + 10)
    assert _snapshot(tmp_path / "GN") == gn_before


@pytest.mark.parametrize("cs_folder,export_cs", [("CS01EC", True), ("CS02ZZ", False)])
def test_exported_cs_offset_updates_only_its_csdata_group(tmp_path, monkeypatch, cs_folder, export_cs):
    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path, cs_folder, export_cs)
    target = tmp_path / "CSdata" / cs_folder
    source = cosmic / DATE / cs_folder / "test.arc_data"
    axis, intensity = read_arc_data(source)
    for archive in ("manual", "folder_delete"):
        write_arc_data(cosmic / "delete" / archive / DATE / cs_folder / "removed.arc_data", axis, intensity, fmt="%.17g")
    write_arc_data(target / "stale.arc_data", axis, intensity)
    other = tmp_path / "CSdata" / "CS99EC"
    write_arc_data(other / "untouched.arc_data", axis, intensity)
    init_before, figures_before = _snapshot(init), _snapshot(figures)
    other_before, gn_before = _snapshot(other), _snapshot(tmp_path / "GN")
    audit_path = shift / "audit_runs" / "run" / "summary.json"
    audit_before = audit_path.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("CSdata 分支不应重建 init 或审核图")

    monkeypatch.setattr(calibration, "build_full_init", unexpected)
    monkeypatch.setattr(calibration, "build_prefix_plots", unexpected)
    for _ in range(2):
        result = run_folder_offset(cosmic, shift, init, mapping, state, state_file, f"{DATE}/{cs_folder}", 1.5, figures)
        assert result["csdata_updated"] is True
        assert result["spectrum_count"] == 3
    np.testing.assert_array_equal(read_arc_data(target / "test.arc_data")[0], (axis + 3) * 2 + 10)
    np.testing.assert_array_equal(read_arc_data(target / "test.arc_data")[1], intensity)
    for archive in ("manual", "folder_delete"):
        np.testing.assert_array_equal(read_arc_data(cosmic / "delete" / archive / DATE / cs_folder / "removed.arc_data")[0], axis + 3)
    assert [path.name for path in target.iterdir()] == ["test.arc_data"]
    assert _snapshot(init) == init_before
    assert _snapshot(figures) == figures_before
    assert _snapshot(other) == other_before
    assert _snapshot(tmp_path / "GN") == gn_before
    assert audit_path.read_bytes() == audit_before


@pytest.mark.parametrize("cs_folder", ["CS01EC", "CST01"])
def test_unexported_cs_prefix_folder_still_updates_init(tmp_path, monkeypatch, cs_folder):
    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path, cs_folder, False)
    axis, _ = read_arc_data(cosmic / DATE / cs_folder / "test.arc_data")
    cs_before = _snapshot(tmp_path / "CSdata")
    monkeypatch.setattr(calibration, "build_prefix_plots", lambda *args: [])
    result = run_folder_offset(cosmic, shift, init, mapping, state, state_file, f"{DATE}/{cs_folder}", 2, figures)
    assert not result.get("csdata_updated")
    target = next(init.rglob("test.arc_data"))
    np.testing.assert_array_equal(read_arc_data(target)[0], (axis + 2) * 2 + 10)
    assert _snapshot(tmp_path / "CSdata") == cs_before


@pytest.mark.parametrize("failure", ["build", "state"])
def test_cs_offset_failure_restores_source_csdata_and_metadata(tmp_path, monkeypatch, failure):
    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path)
    # 导出后新增源谱，确保构建成功后计数变化也纳入回滚检查。
    write_arc_data(cosmic / DATE / "CS01EC" / "new.arc_data", [600, 700], [1, 2])
    before = _snapshot(tmp_path)
    state_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("CS injected failure")

    monkeypatch.setattr(calibration, "build_cs_folder" if failure == "build" else "save_pipeline_state", fail)
    with pytest.raises(OSError, match="CS injected failure"):
        run_folder_offset(cosmic, shift, init, mapping, state, state_file, f"{DATE}/CS01EC", 2, figures)
    assert _snapshot(tmp_path) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == state_before


def test_affine_ui_has_no_audit_step_and_can_rebuild_directly(tmp_path):
    from streamlit.testing.v1 import AppTest

    cosmic, shift, init, mapping, state, state_file, figures = _layout(tmp_path)
    state.cleaning.folder_catalog = {DATE: {"EC01", "CS01EC"}}
    state.cleaning.completed_folder_keys = {f"{DATE}/EC01", f"{DATE}/CS01EC"}
    save_pipeline_state(state_file, state)
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.ui.pages.affine_calibration import render_affine_calibration_page
context = SimpleNamespace(data_root=Path({str(cosmic)!r}), shift_root=Path({str(shift)!r}), init_root=Path({str(init)!r}), state_file=Path({str(state_file)!r}))
render_affine_calibration_page(context)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
    assert [button.label for button in app.button] == ["一键分析并校准"]
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any("直接建立 init" in message.value for message in app.success)
