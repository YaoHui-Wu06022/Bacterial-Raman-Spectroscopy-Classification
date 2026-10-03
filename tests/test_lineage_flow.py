from __future__ import annotations

import json
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.data.builders.init import build_init_datasets
from ramanv2.data.calibration.affine import apply_folder_affine_compensation
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import initialize_folder_lineage, load_folder_lineage, remove_source_folder


def _spectrum(path: Path, start: float = 100.0) -> None:
    write_arc_data(path, [start, start + 1.0], [1.0, 2.0])


def _mapping() -> DatasetMapping:
    return DatasetMapping.from_payload({"genera": {"Enterobacter": ["EAB"]}, "datasets": {"GN": ["Enterobacter"]}})


def test_lineage_initializes_active_and_imported_deleted(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    _spectrum(root / "20240101" / "EAB01" / "a.arc_data")
    _spectrum(root / "delete" / "manual" / "20240101" / "EAB01" / "b.arc_data")

    payload = initialize_folder_lineage(root, _mapping())
    record = payload["folders"]["20240101/EAB01"]
    assert record["genus"] == "Enterobacter"
    assert record["files"]["a.arc_data"]["status"] == "active"
    assert record["files"]["b.arc_data"]["status"] == "manual_deleted"


def test_compensation_only_updates_report_and_init(tmp_path: Path) -> None:
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    init = tmp_path / "init"
    _spectrum(cosmic / "20240101" / "EAB01" / "a.arc_data")
    _spectrum(shift / "20240101" / "EAB01" / "a.arc_data", 110.0)
    affine = shift / "20240101" / "shift"
    affine.mkdir(parents=True)
    (affine / "affine.json").write_text(json.dumps({"status": "manual_approved", "scale": 1.0, "offset": 10.0}), encoding="utf-8")
    initialize_folder_lineage(cosmic, _mapping())

    before, intensity = read_arc_data(shift / "20240101" / "EAB01" / "a.arc_data")
    apply_folder_affine_compensation(cosmic, shift, "20240101", {"EAB01": 2.0})
    after, after_intensity = read_arc_data(shift / "20240101" / "EAB01" / "a.arc_data")
    assert after.tolist() == before.tolist()
    assert after_intensity.tolist() == intensity.tolist()

    report = build_init_datasets(shift, init, {}, _mapping(), ["20240101"], test_root=tmp_path / "CSdata")
    assert report["status"] == "ready"
    output_axis, output_intensity = read_arc_data(init / "Enterobacter" / "EAB20240101_01" / "a.arc_data")
    assert output_axis.tolist() == [112.0, 113.0]
    assert output_intensity.tolist() == intensity.tolist()


def test_source_folder_delete_removes_downstream_recorded_outputs(tmp_path: Path) -> None:
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    _spectrum(cosmic / "20240101" / "EAB01" / "a.arc_data")
    initialize_folder_lineage(cosmic, _mapping())
    output = tmp_path / "init" / "Enterobacter" / "EAB20240101_01"
    output.mkdir(parents=True)
    (output / "a.arc_data").write_text("100 1\n101 2\n", encoding="utf-8")
    payload = load_folder_lineage(cosmic)
    payload["folders"]["20240101/EAB01"]["outputs"]["init"] = str(output)
    from ramanv2.pipeline.lineage import save_folder_lineage

    save_folder_lineage(cosmic, payload)

    result = remove_source_folder(cosmic, "20240101/EAB01", shift_root=shift)
    assert result["status"] == "ready"
    assert not (cosmic / "20240101" / "EAB01").exists()
    assert (cosmic / "delete" / "manual" / "20240101" / "EAB01" / "a.arc_data").is_file()
    assert not output.exists()


def test_source_folder_delete_removes_audit_record(tmp_path: Path) -> None:
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    _spectrum(cosmic / "20240101" / "EAB01" / "a.arc_data")
    initialize_folder_lineage(cosmic, _mapping())
    run_dir = shift / "audit_runs" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "applied", "folders": ["20240101/EAB01"], "items": []}),
        encoding="utf-8",
    )

    remove_source_folder(cosmic, "20240101/EAB01", shift_root=shift)

    assert not run_dir.exists()
