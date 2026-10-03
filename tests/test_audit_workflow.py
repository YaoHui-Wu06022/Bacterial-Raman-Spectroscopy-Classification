from pathlib import Path

import numpy as np
import pytest

from ramanv2.audit.config import AuditConfig
from ramanv2.audit import folder
from ramanv2.audit.folder import analyze_folder, analyze_folders, apply_folder_outliers, run_folder_audit
from ramanv2.core.config import InputConfig
from ramanv2.data.config import DataBuildConfig
from ramanv2.common.arc_data import write_arc_data


def _write_folder(root: Path, date_name: str, folder_name: str, outlier: bool = False) -> Path:
    folder = root / date_name / folder_name
    axis = np.linspace(600.0, 700.0, 101)
    for index in range(10):
        intensity = np.sin(axis / 15.0) + index * 0.001
        if outlier and index == 9:
            intensity = np.cos(axis / 4.0) * 4.0
        write_arc_data(folder / f"sample_{index}.arc_data", axis, intensity)
    return folder


def _config() -> AuditConfig:
    return AuditConfig(
        input=InputConfig(cut_min=600.0, cut_max=700.0, target_points=101, bad_bands=()),
        cleaning=DataBuildConfig(baseline_max_iter=1, cosmic_ray_profile_ids=()),
    )


def test_folder_audit_isolated_and_moves_outlier_to_date_folder(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data_shifted"
    _write_folder(source_root, "20240101", "AA01", outlier=True)
    _write_folder(source_root, "20240101", "AA02", outlier=False)

    report = analyze_folders(source_root, config=_config())

    candidates = [item for item in report["items"] if item["state"] == "candidate"]
    assert len(candidates) == 1
    assert candidates[0]["corr_limit"] is not None
    assert candidates[0]["rmse_limit"] is not None
    assert candidates[0]["reasons"]
    assert candidates[0]["move_status"] == "pending"
    assert all(item["date"] == "20240101" for item in report["items"])
    assert report["folders"] == ["20240101/AA01", "20240101/AA02"]
    applied = apply_folder_outliers(report)
    candidate = candidates[0]
    target = source_root / "delete" / "20240101" / "AA01" / Path(str(candidate["relative_path"])).name
    assert applied["status"] == "applied"
    assert candidate["move_status"] == "moved"
    assert target.is_file()
    assert not (source_root / str(candidate["relative_path"])).exists()
    assert (source_root / "20240101" / "AA02").is_dir()


def test_folder_audit_reports_scores_and_skips_insufficient_references(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data_shifted"
    folder = source_root / "20240101" / "AA01"
    axis = np.linspace(600.0, 700.0, 101)
    for index in range(8):
        write_arc_data(folder / f"sample_{index}.arc_data", axis, np.sin(axis / 15.0))

    report = analyze_folder(source_root, "20240101", "AA01", _config())

    assert report["candidate_count"] == 0
    assert all(item["state"] == "insufficient_reference" for item in report["items"])
    summary = Path(str(report["run_dir"])) / "summary.json"
    candidates = Path(str(report["run_dir"])) / "candidates.csv"
    assert summary.is_file()
    assert candidates.is_file()


def test_folder_audit_keeps_unscorable_spectrum_without_removing_it(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data_shifted"
    folder = source_root / "20240101" / "AA01"
    folder.mkdir(parents=True)
    malformed = folder / "broken.arc_data"
    malformed.write_text("not a spectrum\n", encoding="utf-8")

    report = analyze_folders(source_root, config=_config())

    assert report["candidate_count"] == 0
    assert report["items"][0]["state"] == "unscorable"
    assert malformed.is_file()


def test_folder_audit_rerun_is_idempotent(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data_shifted"
    _write_folder(source_root, "20240101", "AA01", outlier=True)

    first = run_folder_audit(source_root, config=_config())
    second = run_folder_audit(source_root, config=_config())

    assert first["moved_count"] == 1
    assert second["moved_count"] == 0
    assert second["candidate_count"] == 0


def test_folder_audit_target_conflict_preserves_source(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data_shifted"
    _write_folder(source_root, "20240101", "AA01", outlier=True)
    report = analyze_folders(source_root, config=_config())
    candidate = next(item for item in report["items"] if item["state"] == "candidate")
    target = source_root / "delete" / "20240101" / "AA01" / Path(str(candidate["relative_path"])).name
    target.parent.mkdir(parents=True)
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        apply_folder_outliers(report)

    assert (source_root / str(candidate["relative_path"])).is_file()


def test_folder_audit_rolls_back_partial_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = tmp_path / "cosmic_data_shifted"
    first = source_root / "20240101" / "AA01" / "first.arc_data"
    second = source_root / "20240101" / "AA01" / "second.arc_data"
    write_arc_data(first, [600.0, 601.0], [1.0, 2.0])
    write_arc_data(second, [600.0, 601.0], [1.0, 2.0])
    report = {
        "status": "analyzed",
        "source_root": str(source_root),
        "run_dir": str(source_root / "audit_runs" / "run"),
        "folders": ["20240101/AA01"],
        "candidate_count": 2,
        "item_count": 2,
        "items": [
            {
                "date": "20240101",
                "folder": "AA01",
                "relative_path": "20240101/AA01/first.arc_data",
                "state": "candidate",
                "reasons": ["test"],
                "reference_count": 9,
                "neighbor_count": 3,
                "neighbor_corr": 0.0,
                "rmse": 1.0,
            },
            {
                "date": "20240101",
                "folder": "AA01",
                "relative_path": "20240101/AA01/second.arc_data",
                "state": "candidate",
                "reasons": ["test"],
                "reference_count": 9,
                "neighbor_count": 3,
                "neighbor_corr": 0.0,
                "rmse": 1.0,
            },
        ],
    }
    original_move = folder.shutil.move
    call_count = 0

    def fail_on_second_move(source, target):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated move failure")
        return original_move(source, target)

    monkeypatch.setattr(folder.shutil, "move", fail_on_second_move)
    with pytest.raises(OSError):
        apply_folder_outliers(report)

    assert first.is_file()
    assert second.is_file()
    assert not (source_root / "delete").exists()
