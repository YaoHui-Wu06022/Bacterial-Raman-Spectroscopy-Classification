from pathlib import Path

from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.pipeline.calibration import load_date_affine_parameters, run_identity_affine
from ramanv2.records.lineage import load_folder_lineage


def write_spectrum(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_identity_affine_copies_bytes_and_publishes_report(tmp_path: Path) -> None:
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    write_spectrum(cosmic / "20240101" / "BC" / "a.arc_data", "100 1\n101 2\n")
    write_spectrum(cosmic / "20240101" / "Ecoli" / "b.arc_data", "100 3\n101 4\n")
    write_spectrum(cosmic / "20240102" / "Lm" / "c.arc_data", "100 5\n101 6\n")

    reports = run_identity_affine(cosmic, ["20240101", "20240102"], shift)

    assert all(report["status"] == "manual_approved" for report in reports)
    assert (shift / "20240101" / "BC" / "a.arc_data").read_bytes() == (
        cosmic / "20240101" / "BC" / "a.arc_data"
    ).read_bytes()
    assert (shift / "20240101" / "Ecoli" / "b.arc_data").is_file()
    assert (shift / "20240102" / "Lm" / "c.arc_data").is_file()
    report = load_calibration_report(shift, "20240101")
    assert report["mode"] == "identity"
    assert report["scale"] == 1.0
    assert report["offset"] == 0.0
    assert load_date_affine_parameters(shift, "20240101") == (1.0, 0.0)

    lineage = load_folder_lineage(cosmic)
    assert lineage["folders"]["20240101/BC"]["affine"]["status"] == "approved"
    assert lineage["folders"]["20240101/BC"]["outputs"]["shift"] == str(
        (shift / "20240101" / "BC").resolve()
    )
