import json
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.data.calibration.affine import apply_folder_affine_compensation
from ramanv2.data.catalog import ensure_dataset_mapping


def _write_spectrum(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def test_folder_offset_is_raw_increment_and_does_not_record_history(tmp_path: Path) -> None:
    cosmic = tmp_path / "cosmic_data"
    output = tmp_path / "shift_data"
    active = cosmic / "20261001" / "EAB01" / "a.arc_data"
    deleted = cosmic / "delete" / "manual" / "20261001" / "EAB01" / "b.arc_data"
    archived = cosmic / "delete" / "folder_delete" / "20261001" / "EAB01" / "c.arc_data"
    for path in (active, deleted, archived):
        _write_spectrum(path, 1)
    report = output / "20261001" / "shift" / "affine.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "manual_approved", "scale": 2.0, "offset": 10.0, "folder_compensations": {"EAB01": 99}}), encoding="utf-8")
    before = report.read_bytes()

    for _ in range(2):
        result = apply_folder_affine_compensation(cosmic, output, "20261001", {"EAB01": 1.5})
        assert result == {"changed_folders": ["EAB01"], "spectrum_count": 3}
    for path in (active, deleted, archived):
        assert read_arc_data(path)[0].tolist() == [103.0, 104.0]
        assert read_arc_data(path)[1].tolist() == [1.0, 2.0]
    assert report.read_bytes() == before
    assert not list(cosmic.rglob("*.tmp"))


def test_zero_offset_does_not_touch_source(tmp_path: Path) -> None:
    source = tmp_path / "cosmic_data" / "20261001" / "EAB01" / "a.arc_data"
    _write_spectrum(source, 1)
    before = source.read_bytes()
    result = apply_folder_affine_compensation(tmp_path / "cosmic_data", tmp_path / "shift_data", "20261001", {"EAB01": 0})
    assert result["changed_folders"] == []
    assert source.read_bytes() == before


def test_invalid_profile_catalog_is_rebuilt_from_workbook(tmp_path: Path) -> None:
    workbook = Path("dataset/病原菌分类与规范简称.xlsx")
    target = tmp_path / "profile_catalog.json"
    target.write_text(json.dumps({"profiles": {"GN": []}}), encoding="utf-8")

    mapping = ensure_dataset_mapping(workbook, target)

    assert set(mapping.datasets) == {"MICRO", "GN", "GP", "FUNG"}
    assert set(json.loads(target.read_text(encoding="utf-8"))["datasets"]) == set(mapping.datasets)
