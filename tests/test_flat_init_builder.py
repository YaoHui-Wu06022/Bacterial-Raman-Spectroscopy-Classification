import json
from pathlib import Path

import pytest

from ramanv2.data.builders.init import build_init_datasets
from ramanv2.data.hierarchy_config import HierarchySpec


def write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def approve_date(shift: Path, date_name: str) -> None:
    shift_dir = shift / date_name / "shift"
    shift_dir.mkdir(parents=True)
    (shift_dir / "affine.json").write_text(
        json.dumps({"status": "manual_approved", "scale": 1.0, "offset": 0.0}),
        encoding="utf-8",
    )


def test_flat_init_uses_prefix_class_directories(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    for date_name in ("20240101", "20240102"):
        approve_date(shift, date_name)
    write_spectrum(shift / "20240101" / "BC" / "a.arc_data", 1)
    write_spectrum(shift / "20240101" / "Ecoli" / "b.arc_data", 2)
    write_spectrum(shift / "20240102" / "Lm" / "c.arc_data", 3)
    init = tmp_path / "init"
    init_test = tmp_path / "init_test"

    report = build_init_datasets(
        shift,
        init,
        {},
        None,
        {"20240101", "20240102"},
        test_root=init_test,
        selected_test_folders={"20240101/BC"},
        hierarchy=HierarchySpec.flat(),
    )

    assert report["status"] == "ready"
    assert sorted(path.name for path in init.iterdir()) == ["Ecoli", "Lm"]
    assert (init / "Ecoli" / "Ecoli20240101_01" / "b.arc_data").is_file()
    assert (init / "Lm" / "Lm20240102_01" / "c.arc_data").is_file()
    assert sorted(path.name for path in init_test.iterdir()) == ["BC20240101_01"]
    assert (init_test / "BC20240101_01" / "a.arc_data").is_file()
    assert report["test_folders"] == ["BC20240101_01"]


def test_flat_init_rejects_empty_source_groups(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approve_date(shift, "20240101")

    with pytest.raises(ValueError, match="没有可发布的来源组"):
        build_init_datasets(
            shift,
            tmp_path / "init",
            {},
            None,
            {"20240101"},
            test_root=tmp_path / "init_test",
            hierarchy=HierarchySpec.flat(),
        )
