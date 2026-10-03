from __future__ import annotations

import json
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.restoration import _prefix_for_folder, restore_deleted_spectra, scan_deleted_spectra
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


def _mapping() -> DatasetMapping:
    return DatasetMapping(
        genera={"Escherichia": ("EC",)},
        datasets={"GN": ("Escherichia",)},
    )


def test_species_prefix_starting_with_cs_is_not_test_folder() -> None:
    assert _prefix_for_folder("CST01") == "CST"
    assert _prefix_for_folder("CS02PAE") == "PAE"


def test_scan_deleted_spectra_ignores_bead_folder(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data" / "delete" / "manual" / "20240101"
    (root / "EC01").mkdir(parents=True)
    (root / "小球").mkdir()
    (root / "EC01" / "cell3.arc_data").write_text("1 2\n", encoding="utf-8")
    (root / "小球" / "bead.arc_data").write_text("1 2\n", encoding="utf-8")

    records = scan_deleted_spectra(tmp_path / "cosmic_data", _mapping())

    assert [record.relative_path for record in records] == ["20240101/EC01/cell3.arc_data"]


def test_restore_bead_without_affine_only_restores_cosmic_data(tmp_path: Path) -> None:
    data_root = tmp_path / "cosmic_data"
    manual = data_root / "delete" / "manual" / "20240101" / "小球"
    manual.mkdir(parents=True)
    source = manual / "bead1.arc_data"
    source.write_text("600 1\n1000 2\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    save_pipeline_state(state_path, PipelineState())

    result = restore_deleted_spectra(
        data_root,
        tmp_path / "shift_data",
        tmp_path / "init",
        state_path,
        _mapping(),
        ["20240101/小球/bead1.arc_data"],
    )

    assert result["status"] == "bead_restored"
    assert (data_root / "20240101" / "小球" / "bead1.arc_data").is_file()
    assert not source.exists()


def test_restore_rebuilds_from_cosmic_and_updates_init(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    init_root = tmp_path / "init"
    state_path = tmp_path / "state.json"

    manual = data_root / "delete" / "manual" / "20240101" / "EC01"
    manual.mkdir(parents=True)
    (manual / "cell3.arc_data").write_text("100 1\n200 2\n", encoding="utf-8")
    (shift_root / "20240101" / "EC01").mkdir(parents=True)
    affine_dir = shift_root / "20240101" / "shift"
    affine_dir.mkdir(parents=True)
    (affine_dir / "affine.json").write_text(
        json.dumps({"status": "manual_approved", "base_scale": 1.0, "base_offset": 10.0}),
        encoding="utf-8",
    )
    init_target = init_root / "Escherichia" / "EC20240101_01"
    init_target.mkdir(parents=True)
    (init_target / "old.arc_data").write_text("120 1\n220 2\n", encoding="utf-8")
    save_pipeline_state(state_path, PipelineState())

    calls: list[tuple[str, str]] = []

    def fake_audit(root, date_name=None, folder_name=None, **kwargs):
        calls.append((str(date_name), str(folder_name)))
        return {"status": "applied", "items": []}

    monkeypatch.setattr("ramanv2.pipeline.restoration.run_folder_audit", fake_audit)
    result = restore_deleted_spectra(
        data_root,
        shift_root,
        init_root,
        state_path,
        _mapping(),
        ["20240101/EC01/cell3.arc_data"],
    )

    assert result["restored_file_count"] == 1
    assert calls == [("20240101", "EC01")]
    assert (data_root / "20240101" / "EC01" / "cell3.arc_data").is_file()
    assert not (data_root / "delete" / "manual" / "20240101" / "EC01" / "cell3.arc_data").exists()
    assert not (shift_root / "delete" / "20240101" / "EC01").exists()
    assert (init_target / "cell3.arc_data").is_file()
    assert not (init_target / "cell2.arc_data").exists()
    assert read_arc_data(init_target / "cell3.arc_data")[0].tolist() == [110.0, 210.0]


def test_restore_rebuilds_without_prior_shift_output(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    manual = data_root / "delete" / "manual" / "20240101" / "EC01"
    manual.mkdir(parents=True)
    source = manual / "cell3.arc_data"
    source.write_text("1 2\n2 3\n", encoding="utf-8")
    (shift_root / "20240101" / "EC01").mkdir(parents=True)
    affine_dir = shift_root / "20240101" / "shift"
    affine_dir.mkdir(parents=True)
    (affine_dir / "affine.json").write_text(
        json.dumps({"status": "manual_approved", "base_scale": 1.0, "base_offset": 10.0}),
        encoding="utf-8",
    )
    (tmp_path / "init" / "Escherichia" / "EC20240101_01").mkdir(parents=True)
    state_path = tmp_path / "state.json"
    save_pipeline_state(state_path, PipelineState())
    monkeypatch.setattr(
        "ramanv2.pipeline.restoration.run_folder_audit",
        lambda *args, **kwargs: {"status": "applied", "items": []},
    )

    result = restore_deleted_spectra(
        data_root,
        shift_root,
        tmp_path / "init",
        state_path,
        _mapping(),
        ["20240101/EC01/cell3.arc_data"],
    )

    assert result["restored_file_count"] == 1
    assert (data_root / "20240101" / "EC01" / "cell3.arc_data").is_file()
    assert read_arc_data(shift_root / "20240101" / "EC01" / "cell3.arc_data")[0].tolist() == [11.0, 12.0]
