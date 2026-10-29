"""补偿到 init 的事务边界、下游隔离及页面输入语义。"""

import json
from pathlib import Path

import numpy as np

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


def _layout(tmp_path: Path, test_folder="EC03", export_cs=True):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    axis = np.linspace(400, 2000, 81)
    intensity = 100 + np.sin(axis / 20)
    for relative in (f"{DATE}/EC01/a.arc_data", f"delete/manual/{DATE}/EC01/b.arc_data", f"delete/folder_delete/{DATE}/EC01/c.arc_data", f"{DATE}/{test_folder}/test.arc_data"):
        write_arc_data(cosmic / relative, axis, intensity, fmt="%.17g")
    report = shift / DATE / "shift" / "affine.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "manual_approved", "scale": 2.0, "offset": 10.0, "folder_compensations": {"EC01": 99}}), encoding="utf-8")
    mapping = DatasetMapping(genera={"Escherichia": ("EC", "CST")}, datasets={"GN": ("Escherichia",)})
    assert rebuild_shift_data(cosmic, shift, [DATE])[0]["status"] == "rebuilt"
    audit = shift / "audit_runs" / "run" / "summary.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"status": "applied", "folders": [f"{DATE}/EC01"]}), encoding="utf-8")
    full_report = build_full_init(shift, init, mapping, [DATE], selected_test_folders=[f"{DATE}/{test_folder}"] if export_cs else [])
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
    affine_before = (shift / DATE / "shift" / "affine.json").read_bytes()
    test_before, gn_before = _snapshot(tmp_path / "init_test"), _snapshot(tmp_path / "GN")
    original_axis, original_intensity = read_arc_data(cosmic / DATE / "EC01" / "a.arc_data")
    plotted = []

    def plot(source, output, groups):
        plotted.append((Path(source), groups))
        target = output / "Escherichia" / "EC.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"updated init plot")

    monkeypatch.setattr(calibration, "build_prefix_plots_for_groups", plot)
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
    assert _snapshot(tmp_path / "init_test") == test_before
    assert _snapshot(tmp_path / "GN") == gn_before
    assert state.dataset_builds["profiles"] == {"GN": {"status": "ready"}}
    assert plotted == [(init, {("Escherichia", "EC")})]
    saved = state_file.read_text(encoding="utf-8")
    assert "affine_compensations" not in saved
    assert (shift / DATE / "shift" / "affine.json").read_bytes() == affine_before




















