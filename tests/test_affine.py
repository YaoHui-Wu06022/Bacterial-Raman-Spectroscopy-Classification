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
