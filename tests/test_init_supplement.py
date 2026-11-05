from pathlib import Path

from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.pipeline.supplement import apply_supplement_plan, scan_supplement_plan


def write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def make_mapping() -> DatasetMapping:
    return DatasetMapping(
        genera={"Corynebacterium": ("CAMY",)},
        datasets={"GP": ("Corynebacterium",)},
    )


def test_scan_and_apply_supplement_plan(tmp_path: Path) -> None:
    scan_root = tmp_path / "scan"
    root = scan_root / "补充数据"
    write_spectrum(root / "CAMY20261027" / "cell1.arc_data", 1)
    write_spectrum(scan_root / "init" / "Corynebacterium" / "CAMY20261027" / "cell1.arc_data", 1)
    write_spectrum(scan_root / "init" / "Corynebacterium" / "CAMY20261026" / "cell1.arc_data", 2)

    plan = scan_supplement_plan(root, scan_root / "init", make_mapping())

    assert plan.errors == ()
    assert plan.unbacked_init_folders == ("Corynebacterium/CAMY20261026",)

    apply_root = tmp_path / "apply"
    root = apply_root / "补充数据"
    write_spectrum(root / "CAMY20261027" / "new.arc_data", 1)
    init = apply_root / "init"
    write_spectrum(init / "Corynebacterium" / "CAMY20261027" / "old.arc_data", 2)
    plan = scan_supplement_plan(root, init, make_mapping())

    result = apply_supplement_plan(init, plan)

    target = init / "Corynebacterium" / "CAMY20261027"
    assert result["folder_count"] == 1
    assert result["file_count"] == 1
    assert result["targets"] == ["Corynebacterium/CAMY20261027"]
    assert sorted(path.name for path in target.iterdir()) == ["new.arc_data"]
    assert (root / "CAMY20261027" / "new.arc_data").is_file()
