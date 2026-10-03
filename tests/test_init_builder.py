import json
from pathlib import Path

import pytest

from ramanv2.data.builders.init import (
    InitBuildConflictError,
    build_full_init,
    build_profile_datasets,
    build_init_datasets,
)
from ramanv2.data.catalog import DatasetMapping


def write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def approve_date(root: Path, date_name: str) -> Path:
    shift = root / date_name / "shift"
    shift.mkdir(parents=True)
    (shift / "affine.json").write_text(
        json.dumps({"status": "manual_approved"}), encoding="utf-8"
    )
    return root / date_name


def test_build_init_copies_only_approved_dates_and_profiles(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    write_spectrum(approved / "ECL01" / "sample.arc_data", 1)
    write_spectrum(approved / "小球" / "bead.arc_data", 2)
    rejected = shift / "20240102" / "ECL02"
    write_spectrum(rejected / "sample.arc_data", 3)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )

    report = build_init_datasets(
        shift,
        tmp_path / "init",
        {"GN": tmp_path / "GN" / "init"},
        mapping,
        ["20240101", "20240102"],
    )

    target = tmp_path / "init" / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    profile_target = tmp_path / "GN" / "init" / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    assert target.is_file()
    assert profile_target.is_file()
    assert not (tmp_path / "init" / "Enterobacter" / "ECL20240102_01").exists()
    assert report["source_dates"] == ["20240101"]
    assert report["source_files"] == 1
    assert report["mode"] == "full"
    assert report["date_status"]["20240102"]["reason"] == "affine_not_approved"
    assert (target.read_bytes() == (approved / "ECL01" / "sample.arc_data").read_bytes())


def test_mapping_rejects_unknown_and_duplicate_prefixes() -> None:
    with pytest.raises(ValueError, match="同时属于多个属"):
        DatasetMapping.from_payload(
            {"genera": {"A": ["EC"], "B": ["EC"]}, "datasets": {"GN": ["A"]}}
        )
    with pytest.raises(ValueError, match="未定义属"):
        DatasetMapping.from_payload(
            {"genera": {"A": ["EC"]}, "datasets": {"GN": ["B"]}}
        )


def test_incremental_build_preserves_existing_output_and_adds_new_date(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    first_date = approve_date(shift, "20240101")
    second_date = approve_date(shift, "20240102")
    first_source = first_date / "ECL01" / "sample.arc_data"
    second_source = second_date / "ECL01" / "sample.arc_data"
    write_spectrum(first_source, 1)
    write_spectrum(second_source, 2)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )
    init_root = tmp_path / "init"
    profile_root = tmp_path / "GN" / "init"
    build_init_datasets(shift, init_root, {"GN": profile_root}, mapping, ["20240101"])
    old_target = init_root / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"

    report = build_init_datasets(
        shift,
        init_root,
        {"GN": profile_root},
        mapping,
        ["20240101", "20240102"],
        mode="incremental",
        date_names=["20240102"],
    )

    assert report["mode"] == "incremental"
    assert report["source_dates"] == ["20240102"]
    assert old_target.read_bytes() == first_source.read_bytes()
    assert (init_root / "Enterobacter" / "ECL20240102_01" / "sample.arc_data").read_bytes() == second_source.read_bytes()


def test_incremental_conflict_leaves_existing_outputs_untouched(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    date_dir = approve_date(shift, "20240101")
    source = date_dir / "ECL01" / "sample.arc_data"
    write_spectrum(source, 1)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )
    init_root = tmp_path / "init"
    profile_root = tmp_path / "GN" / "init"
    build_init_datasets(shift, init_root, {"GN": profile_root}, mapping, ["20240101"])
    target = init_root / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    before = target.read_bytes()

    with pytest.raises(InitBuildConflictError) as error:
        build_init_datasets(
            shift,
            init_root,
            {"GN": profile_root},
            mapping,
            ["20240101"],
            mode="incremental",
            date_names=["20240101"],
        )

    assert error.value.conflicts
    assert target.read_bytes() == before


def test_failed_full_rebuild_preserves_previous_outputs(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    first_date = approve_date(shift, "20240101")
    first_source = first_date / "ECL01" / "sample.arc_data"
    write_spectrum(first_source, 1)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )
    init_root = tmp_path / "init"
    profile_root = tmp_path / "GN" / "init"
    build_init_datasets(shift, init_root, {"GN": profile_root}, mapping, ["20240101"])
    target = init_root / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    before = target.read_bytes()

    bad_date = approve_date(shift, "20240102")
    bad_file = bad_date / "ECL01" / "broken.arc_data"
    bad_file.parent.mkdir(parents=True)
    bad_file.write_text("broken", encoding="utf-8")

    with pytest.raises(ValueError, match="无效光谱"):
        build_init_datasets(
            shift,
            init_root,
            {"GN": profile_root},
            mapping,
            ["20240101", "20240102"],
        )

    assert target.read_bytes() == before


def test_build_full_init_and_profiles_are_separate_stages(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    write_spectrum(approved / "ECL01" / "sample.arc_data", 1)
    write_spectrum(approved / "KP01" / "sample.arc_data", 2)
    cs_source = approved / "CS01KP" / "test.arc_data"
    write_spectrum(cs_source, 3)
    mapping = DatasetMapping.from_payload(
        {
            "genera": {"Enterobacter": ["ECL"], "Klebsiella": ["KP"]},
            "datasets": {"GN": ["Enterobacter", "Klebsiella"], "ECL_ONLY": ["Enterobacter"]},
        }
    )
    init_root = tmp_path / "init"
    full_report = build_full_init(
        shift,
        init_root,
        mapping,
        ["20240101"],
        selected_cs_folders=["20240101/CS01KP"],
    )
    assert full_report["source_files"] == 3
    assert full_report["test_files"] == 1
    assert (tmp_path / "CSdata" / "CS01KP" / "test.arc_data").read_bytes() == cs_source.read_bytes()
    assert not (init_root / "CS01KP").exists()
    assert (init_root / "Enterobacter" / "ECL20240101_01" / "sample.arc_data").is_file()
    assert (init_root / "Klebsiella" / "KP20240101_01" / "sample.arc_data").is_file()
    profile_roots = {
        "GN": tmp_path / "GN" / "init",
        "ECL_ONLY": tmp_path / "ECL_ONLY" / "init",
    }
    profile_report = build_profile_datasets(init_root, profile_roots, mapping)
    assert profile_report["profiles"]["GN"]["source_files"] == 2
    assert profile_report["profiles"]["ECL_ONLY"]["source_files"] == 1
    assert (profile_roots["GN"] / "Klebsiella" / "KP20240101_01" / "sample.arc_data").is_file()
    assert not (profile_roots["GN"] / "CS01KP").exists()
    assert not (profile_roots["ECL_ONLY"] / "Klebsiella").exists()
    assert (init_root / "Klebsiella" / "KP20240101_01" / "sample.arc_data").is_file()


def test_cs_folder_does_not_require_mapping_and_is_published_with_full_init(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    source = approved / "CS02AA" / "sample.arc_data"
    write_spectrum(source, 4)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )

    report = build_full_init(shift, tmp_path / "init", mapping, ["20240101"])

    target = tmp_path / "CSdata" / "CS02AA" / "sample.arc_data"
    assert target.read_bytes() == source.read_bytes()
    assert report["test_folders"] == ["CS02AA"]


def test_only_unmapped_cs_is_default_selected(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    write_spectrum(approved / "ECL01" / "sample.arc_data", 1)
    write_spectrum(approved / "CS02AA" / "unknown.arc_data", 2)
    write_spectrum(approved / "CS01KP" / "mapped.arc_data", 3)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Klebsiella": ["KP"], "Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter", "Klebsiella"]}}
    )

    report = build_full_init(shift, tmp_path / "init", mapping, ["20240101"])

    assert report["test_folders"] == ["CS02AA"]
    assert (tmp_path / "CSdata" / "CS02AA" / "unknown.arc_data").is_file()
    assert not (tmp_path / "CSdata" / "CS01KP").exists()


def test_mapped_cs_without_selection_is_built_as_regular_init_source(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    source = approved / "CS05EC" / "sample.arc_data"
    write_spectrum(source, 2)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Escherichia": ["EC"]}, "datasets": {"GN": ["Escherichia"]}}
    )

    report = build_full_init(shift, tmp_path / "init", mapping, ["20240101"])

    target = tmp_path / "init" / "Escherichia" / "EC20240101_01" / "sample.arc_data"
    assert target.read_bytes() == source.read_bytes()
    assert not (tmp_path / "CSdata" / "CS05EC").exists()
    assert report["test_folders"] == []


def test_cs_outputs_are_preserved_when_full_build_validation_fails(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    good = approved / "CS01KP" / "sample.arc_data"
    write_spectrum(good, 1)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )
    build_full_init(
        shift,
        tmp_path / "init",
        mapping,
        ["20240101"],
        selected_cs_folders=["20240101/CS01KP"],
    )
    init_target = tmp_path / "init"
    cs_target = tmp_path / "CSdata" / "CS01KP" / "sample.arc_data"
    before_init = init_target.exists()
    before_cs = cs_target.read_bytes()

    bad = approve_date(shift, "20240102") / "CS02AA" / "broken.arc_data"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError, match="无效光谱"):
        build_full_init(
            shift,
            init_target,
            mapping,
            ["20240101", "20240102"],
            selected_cs_folders=["20240101/CS01KP", "20240102/CS02AA"],
        )

    assert before_init and cs_target.read_bytes() == before_cs


def test_duplicate_cs_folder_names_across_dates_fail_without_replacing_outputs(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    first = approve_date(shift, "20240101")
    write_spectrum(first / "CS01KP" / "sample.arc_data", 1)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )
    build_full_init(
        shift,
        tmp_path / "init",
        mapping,
        ["20240101"],
        selected_cs_folders=["20240101/CS01KP"],
    )
    target = tmp_path / "CSdata" / "CS01KP" / "sample.arc_data"
    before = target.read_bytes()

    second = approve_date(shift, "20240102")
    write_spectrum(second / "CS01KP" / "sample.arc_data", 2)
    with pytest.raises(InitBuildConflictError):
        build_full_init(
            shift,
            tmp_path / "init",
            mapping,
            ["20240101", "20240102"],
            selected_cs_folders=["20240101/CS01KP", "20240102/CS01KP"],
        )
    assert target.read_bytes() == before

def test_profile_build_failure_keeps_previous_output(tmp_path: Path) -> None:
    init_root = tmp_path / "init"
    write_spectrum(init_root / "Enterobacter" / "ECL20240101_01" / "sample.arc_data", 1)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )
    profile_root = tmp_path / "GN" / "init"
    build_profile_datasets(init_root, {"GN": profile_root}, mapping)
    target = profile_root / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    before = target.read_bytes()
    (init_root / "Enterobacter" / "broken.arc_data").write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError, match="无效光谱"):
        build_profile_datasets(init_root, {"GN": profile_root}, mapping)
    assert target.read_bytes() == before
