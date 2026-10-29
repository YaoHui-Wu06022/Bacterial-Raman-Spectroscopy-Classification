from pathlib import Path


from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.supplement import (
    apply_supplement_plan,
    scan_supplement_plan,
    validate_supplement_plan,
)


def write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def make_mapping() -> DatasetMapping:
    return DatasetMapping(
        genera={"Corynebacterium": ("CAMY",)},
        datasets={"GP": ("Corynebacterium",)},
    )




def test_scan_supplement_plan_reports_unbacked_direct_init_folder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "补充数据"
    write_spectrum(root / "CAMY20261027" / "cell1.arc_data", 1)
    write_spectrum(tmp_path / "init" / "Corynebacterium" / "CAMY20261027" / "cell1.arc_data", 1)
    write_spectrum(tmp_path / "init" / "Corynebacterium" / "CAMY20261026" / "cell1.arc_data", 2)

    plan = scan_supplement_plan(root, tmp_path / "init", make_mapping())

    assert plan.errors == ()
    assert plan.unbacked_init_folders == ("Corynebacterium/CAMY20261026",)








def test_apply_supplement_plan_replaces_target_and_keeps_source(tmp_path: Path) -> None:
    root = tmp_path / "补充数据"
    write_spectrum(root / "CAMY20261027" / "new.arc_data", 1)
    init = tmp_path / "init"
    write_spectrum(init / "Corynebacterium" / "CAMY20261027" / "old.arc_data", 2)
    plan = scan_supplement_plan(root, init, make_mapping())

    result = apply_supplement_plan(init, plan)

    target = init / "Corynebacterium" / "CAMY20261027"
    assert result["folder_count"] == 1
    assert result["file_count"] == 1
    assert result["targets"] == ["Corynebacterium/CAMY20261027"]
    assert sorted(path.name for path in target.iterdir()) == ["new.arc_data"]
    assert (root / "CAMY20261027" / "new.arc_data").is_file()


