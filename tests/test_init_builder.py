import json
from pathlib import Path

import pytest

from ramanv2.data.builders.init import (
    build_full_init,
    build_init_datasets,
    build_profile_datasets,
)
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import load_folder_lineage, save_folder_lineage


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

    progress_calls: list[tuple[str, int, int]] = []

    report = build_init_datasets(
        shift,
        tmp_path / "init",
        {"GN": tmp_path / "GN" / "init"},
        mapping,
        ["20240101", "20240102"],
        progress_callback=lambda date_name, processed, total: progress_calls.append(
            (date_name, processed, total)
        ),
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
    assert progress_calls == [("20240101", 1, 1)]


def test_rebuild_ignores_stale_test_names_and_keeps_unified_numbers(tmp_path: Path) -> None:
    """历史测试组名不得覆盖统一编号：重建后同一前缀同日期的编号必须唯一。"""
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    init = tmp_path / "init"
    date = "20240101"
    approved = approve_date(shift, date)
    for name in ("EC01", "EC02"):
        write_spectrum(approved / name / "sample.arc_data", 1)
        write_spectrum(cosmic / date / name / "sample.arc_data", 1)
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["EC"]}, "datasets": {"GN": ["Enterobacter"]}}
    )

    build_full_init(shift, init, mapping, [date], selected_test_folders=[f"{date}/EC02"])
    # 模拟历史数据：EC02 的测试目录曾被命名成会与 init 撞号的 _01。
    payload = load_folder_lineage(cosmic)
    payload["folders"][f"{date}/EC02"]["outputs"]["test"] = str(
        tmp_path / "init_test" / f"EC{date}_01"
    )
    save_folder_lineage(cosmic, payload)

    build_full_init(shift, init, mapping, [date], selected_test_folders=[f"{date}/EC02"])

    assert (init / "Enterobacter" / f"EC{date}_01").is_dir()
    assert (tmp_path / "init_test" / f"EC{date}_02").is_dir()
    assert not (tmp_path / "init_test" / f"EC{date}_01").exists()


def test_build_init_rejects_invalid_spectrum(tmp_path: Path) -> None:
    """坏谱在复制阶段即被拒绝，不留下半成品 init。"""
    shift = tmp_path / "shift_data"
    approved = approve_date(shift, "20240101")
    write_spectrum(approved / "ECL01" / "ok.arc_data", 1)
    (approved / "ECL01" / "broken.arc_data").write_text("broken", encoding="utf-8")
    mapping = DatasetMapping.from_payload(
        {"genera": {"Enterobacter": ["ECL"]}, "datasets": {"GN": ["Enterobacter"]}}
    )

    with pytest.raises(ValueError, match="无效光谱"):
        build_init_datasets(shift, tmp_path / "init", {}, mapping, ["20240101"])
    assert not (tmp_path / "init").exists()


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
