import json
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.data.builders.init import update_compensated_init
from ramanv2.data.calibration.affine import apply_folder_affine_compensation
from ramanv2.data.catalog import DatasetMapping, ensure_dataset_mapping


def _write_spectrum(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def test_folder_affine_compensation_rebuilds_from_raw_and_records_history(tmp_path: Path) -> None:
    source = tmp_path / "cosmic_data" / "20261001"
    output = tmp_path / "shift_data"
    _write_spectrum(source / "EAB01" / "a.arc_data", 1)
    _write_spectrum(source / "EAB02" / "b.arc_data", 2)
    affine = output / "20261001" / "shift"
    affine.mkdir(parents=True)
    (affine / "affine.json").write_text(
        json.dumps({"status": "manual_approved", "scale": 2.0, "offset": 10.0}),
        encoding="utf-8",
    )

    first = apply_folder_affine_compensation(source.parent, output, "20261001", {"EAB01": 1.0, "EAB02": -2.0})
    assert not (output / "20261001" / "EAB01" / "a.arc_data").exists()
    assert first["folder_compensations"] == {"EAB01": 1.0, "EAB02": -2.0}

    second = apply_folder_affine_compensation(source.parent, output, "20261001", {"EAB01": 3.0})
    assert not (output / "20261001" / "EAB01" / "a.arc_data").exists()
    assert not (output / "20261001" / "EAB02" / "b.arc_data").exists()
    assert len(second["compensation_history"]) == 2


def test_compensation_updates_existing_init_targets_only(tmp_path: Path) -> None:
    source = tmp_path / "cosmic_data" / "20261001"
    output = tmp_path / "shift_data"
    init_root = tmp_path / "init"
    profile_root = tmp_path / "GN" / "init"
    _write_spectrum(source / "EAB01" / "a.arc_data", 1)
    _write_spectrum(source / "EAB02" / "b.arc_data", 2)
    affine = output / "20261001" / "shift"
    affine.mkdir(parents=True)
    (affine / "affine.json").write_text(
        json.dumps({"status": "manual_approved", "scale": 2.0, "offset": 10.0}),
        encoding="utf-8",
    )
    apply_folder_affine_compensation(source.parent, output, "20261001", {"EAB01": 1.0, "EAB02": 0.0})
    (output / "20261001" / "EAB01").mkdir(parents=True)
    (output / "20261001" / "EAB01" / "a.arc_data").write_text("210 1\n212 2\n", encoding="utf-8")
    _write_spectrum(init_root / "Enterobacter" / "EAB20261001_01" / "a.arc_data", 99)
    _write_spectrum(profile_root / "Enterobacter" / "EAB20261001_01" / "a.arc_data", 99)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["EAB"]}, "datasets": {"GN": ["Enterobacter"]}}
    )

    report = update_compensated_init(
        output,
        "20261001",
        {"EAB01": 1.0},
        init_root,
        {"GN": profile_root},
        mapping,
    )

    axis, intensity = read_arc_data(init_root / "Enterobacter" / "EAB20261001_01" / "a.arc_data")
    profile_axis, profile_intensity = read_arc_data(
        profile_root / "Enterobacter" / "EAB20261001_01" / "a.arc_data"
    )
    assert report["status"] == "ready"
    assert len(report["updated_targets"]) == 2
    assert axis.tolist() == [211.0, 213.0]
    assert profile_axis.tolist() == axis.tolist()
    assert intensity.tolist() == [1.0, 2.0]
    assert profile_intensity.tolist() == intensity.tolist()


def test_invalid_profile_catalog_is_rebuilt_from_workbook(tmp_path: Path) -> None:
    workbook = Path("dataset/病原菌分类与规范简称.xlsx")
    target = tmp_path / "profile_catalog.json"
    target.write_text(json.dumps({"profiles": {"GN": []}}), encoding="utf-8")

    mapping = ensure_dataset_mapping(workbook, target)

    assert set(mapping.datasets) == {"MICRO", "GN", "GP", "FUNG"}
    assert set(json.loads(target.read_text(encoding="utf-8"))["datasets"]) == set(mapping.datasets)
