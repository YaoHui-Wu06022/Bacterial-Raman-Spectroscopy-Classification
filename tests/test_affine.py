import json
from pathlib import Path

import numpy as np

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.calibration import affine
from ramanv2.data.calibration.affine import (
    AUXILIARY_PEAKS,
    PRIMARY_PEAKS,
    analyze_all_dates,
    analyze_date,
    approve_and_apply_date,
)


TRUE_SCALE = 1.002
TRUE_OFFSET = -3.5


def _write_bead(
    path: Path,
    include_auxiliary: bool = True,
    shift: float = 0.0,
    auxiliary_shift: float = 0.0,
    scale: float = TRUE_SCALE,
    offset: float = TRUE_OFFSET,
) -> None:
    axis = np.linspace(400.0, 1800.0, 1401)
    targets = list(PRIMARY_PEAKS)
    if include_auxiliary:
        targets.extend(AUXILIARY_PEAKS)
    centers = [
        (target - offset) / scale
        + shift
        + (auxiliary_shift if target in AUXILIARY_PEAKS else 0.0)
        for target in targets
    ]
    amplitudes = [900.0, 1500.0, 1100.0, 350.0, 450.0]
    intensities = np.full(axis.shape, 100.0)
    for center, amplitude in zip(centers, amplitudes):
        intensities += amplitude * np.exp(-0.5 * np.square((axis - center) / 3.0))
    write_arc_data(path, axis, intensities)


def _write_date(root: Path, date_name: str, bead_count: int = 3, include_auxiliary: bool = True) -> Path:
    date_dir = root / date_name
    for index in range(bead_count):
        _write_bead(date_dir / "小球" / f"bead{index + 1}.arc_data", include_auxiliary, shift=index * 0.03)
    write_arc_data(
        date_dir / "EC01" / "sample.arc_data",
        np.linspace(400.0, 1800.0, 1401),
        np.linspace(1.0, 2.0, 1401),
    )
    return date_dir


def test_analyze_date_recovers_affine_and_writes_bead_plots(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = _write_date(source_root, "20240101")

    report = analyze_date(date_dir, tmp_path / "cosmic_data_shifted")

    assert report["status"] == "ready"
    assert report["valid_repeat_count"] == 3
    assert abs(float(report["scale"]) - TRUE_SCALE) < 0.001
    assert abs(float(report["offset"]) - TRUE_OFFSET) < 0.2
    shift_dir = tmp_path / "cosmic_data_shifted" / "20240101" / "shift"
    assert (shift_dir / "affine.json").is_file()
    assert len(list(shift_dir.glob("*_calibration.png"))) == 3


def test_auxiliary_residual_uses_corrected_axis_only_once(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = source_root / "20240101"
    for index in range(3):
        _write_bead(
            date_dir / "小球" / f"bead{index + 1}.arc_data",
            scale=0.997373109,
            offset=21.264244785,
        )
    report = analyze_date(date_dir, tmp_path / "cosmic_data_shifted")

    assert report["status"] == "ready"
    assert max(abs(float(value)) for value in report["auxiliary_residuals_cm-1"].values()) <= 2.0


def test_approve_writes_corrected_spectra_without_bead_folder(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = _write_date(source_root, "20240101")
    output_root = tmp_path / "cosmic_data_shifted"
    analyze_date(date_dir, output_root)

    report = approve_and_apply_date(date_dir, output_root)

    assert report["status"] == "manual_approved"
    output_date = output_root / "20240101"
    assert not (output_date / "小球").exists()
    corrected = np.loadtxt(output_date / "EC01" / "sample.arc_data")
    source = np.loadtxt(date_dir / "EC01" / "sample.arc_data")
    np.testing.assert_allclose(
        corrected[:, 0], float(report["scale"]) * source[:, 0] + float(report["offset"]), atol=1e-7
    )
    np.testing.assert_allclose(corrected[:, 1], source[:, 1], atol=1e-7)
    saved = json.loads((output_date / "shift" / "affine.json").read_text(encoding="utf-8"))
    assert saved["manual_approval_enable"] is True


def test_single_valid_repeat_is_limited_but_approvable(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = _write_date(source_root, "20240101", bead_count=1)
    output_root = tmp_path / "cosmic_data_shifted"

    report = analyze_date(date_dir, output_root)

    assert report["status"] == "limited_replicates"
    approve_and_apply_date(date_dir, output_root)
    assert (output_root / "20240101" / "EC01" / "sample.arc_data").is_file()


def test_median_repeat_aggregation_reduces_outlier_influence(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = _write_date(source_root, "20240101", bead_count=3)
    _write_bead(date_dir / "小球" / "bead3.arc_data", shift=10.0)

    report = analyze_date(date_dir, tmp_path / "cosmic_data_shifted")

    assert report["status"] == "ready"
    assert abs(float(report["scale"]) - TRUE_SCALE) < 0.001
    assert abs(float(report["offset"]) - TRUE_OFFSET) < 0.2


def test_auxiliary_residual_over_limit_requires_review(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = source_root / "20240101"
    _write_bead(date_dir / "小球" / "bead1.arc_data", auxiliary_shift=8.0)
    write_arc_data(
        date_dir / "EC01" / "sample.arc_data",
        np.linspace(400.0, 1800.0, 1401),
        np.linspace(1.0, 2.0, 1401),
    )

    report = analyze_date(date_dir, tmp_path / "cosmic_data_shifted")

    assert report["status"] == "needs_review"
    assert max(abs(float(value)) for value in report["auxiliary_residuals_cm-1"].values()) > 2.0


def test_missing_auxiliary_peaks_requires_review(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = _write_date(source_root, "20240101", include_auxiliary=False)

    report = analyze_date(date_dir, tmp_path / "cosmic_data_shifted")

    assert report["status"] == "needs_review"
    assert report["valid_repeat_count"] == 0
    assert not (tmp_path / "cosmic_data_shifted" / "20240101" / "EC01").exists()


def test_batch_analysis_keeps_failed_date_diagnostic_only(tmp_path: Path) -> None:
    source_root = tmp_path / "cosmic_data"
    _write_date(source_root, "20240101", bead_count=1)
    failed_date = source_root / "20240102" / "EC01"
    write_arc_data(failed_date / "sample.arc_data", [400.0, 401.0], [1.0, 2.0])

    reports = analyze_all_dates(source_root, tmp_path / "cosmic_data_shifted")

    assert [report["date"] for report in reports] == ["20240101", "20240102"]
    assert reports[1]["status"] == "needs_review"
    assert (tmp_path / "cosmic_data_shifted" / "20240102" / "shift" / "affine.json").is_file()
    assert not (tmp_path / "cosmic_data_shifted" / "20240102" / "EC01").exists()


def test_failed_rerun_preserves_previous_shift(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "cosmic_data"
    date_dir = _write_date(source_root, "20240101", bead_count=1)
    output_root = tmp_path / "cosmic_data_shifted"
    analyze_date(date_dir, output_root)
    affine_path = output_root / "20240101" / "shift" / "affine.json"
    previous = affine_path.read_text(encoding="utf-8")

    def fail_figure(*args, **kwargs):
        raise RuntimeError("simulated publish failure")

    monkeypatch.setattr(affine, "_save_calibration_figure", fail_figure)
    try:
        analyze_date(date_dir, output_root)
    except RuntimeError:
        pass
    else:
        raise AssertionError("分析失败应当抛出异常")

    assert affine_path.read_text(encoding="utf-8") == previous
    assert not any(path.name.startswith("shift_previous_") for path in (output_root / "20240101").iterdir())
