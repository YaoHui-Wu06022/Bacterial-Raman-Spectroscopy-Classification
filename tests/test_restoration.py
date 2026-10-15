from __future__ import annotations

import json
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.cleaning import refresh_completed_dates, scan_review_folders
from ramanv2.pipeline.lineage import load_folder_lineage
from ramanv2.pipeline.restoration import (
    _prefix_for_folder,
    restore_deleted_folder,
    restore_deleted_spectra,
    scan_deleted_folders,
    scan_deleted_spectra,
)
from ramanv2.pipeline.state import PipelineState, load_pipeline_state, save_pipeline_state


def _mapping() -> DatasetMapping:
    return DatasetMapping(
        genera={"Escherichia": ("EC",)},
        datasets={"GN": ("Escherichia",)},
    )


def _write_approved_report(
    shift_root: Path,
    date_name: str,
    scale: float = 1.0,
    offset: float = 10.0,
) -> Path:
    affine_dir = shift_root / date_name / "shift"
    affine_dir.mkdir(parents=True, exist_ok=True)
    report_path = affine_dir / "affine.json"
    report_path.write_text(
        json.dumps({"status": "manual_approved", "base_scale": scale, "base_offset": offset}),
        encoding="utf-8",
    )
    return report_path


def _write_applied_report(shift_root: Path, date_name: str, folder_name: str) -> Path:
    run_dir = shift_root / "audit_runs" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"status": "applied", "folders": [f"{date_name}/{folder_name}"]}),
        encoding="utf-8",
    )
    return summary_path


def test_species_prefix_starting_with_cs_is_not_test_folder() -> None:
    assert _prefix_for_folder("CST01") == "CST"
    assert _prefix_for_folder("CS02PAE") == "PAE"


def test_scan_deleted_entries_read_only_their_own_archive(tmp_path: Path) -> None:
    """人工删除只扫 ``delete/manual``（忽略小球），整夹删除只扫 ``delete/folder_delete``。"""
    manual = tmp_path / "cosmic_data" / "delete" / "manual" / "20240101"
    (manual / "EC01").mkdir(parents=True)
    (manual / "小球").mkdir()
    (manual / "EC01" / "cell3.arc_data").write_text("1 2\n", encoding="utf-8")
    (manual / "小球" / "bead.arc_data").write_text("1 2\n", encoding="utf-8")
    archive = tmp_path / "cosmic_data" / "delete" / "folder_delete" / "20240101" / "EC02"
    archive.mkdir(parents=True)
    (archive / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")

    spectra = scan_deleted_spectra(tmp_path / "cosmic_data", _mapping())
    folders = scan_deleted_folders(tmp_path / "cosmic_data")

    assert [record.relative_path for record in spectra] == ["20240101/EC01/cell3.arc_data"]
    assert [(item.relative_path, item.file_count) for item in folders] == [("20240101/EC02", 1)]


def test_scan_deleted_spectra_orders_naturally_and_ignores_non_spectra(tmp_path: Path) -> None:
    """扫描保持自然序（cell2 在 cell10 前），只收 .arc_data，小球默认忽略、可显式纳入。"""
    manual = tmp_path / "cosmic_data" / "delete" / "manual" / "20240101"
    for folder_name, names in (
        ("EC10", ("cell2.arc_data", "cell10.arc_data")),
        ("EC2", ("cell1.arc_data",)),
        ("小球", ("bead1.arc_data",)),
    ):
        (manual / folder_name).mkdir(parents=True)
        for name in names:
            (manual / folder_name / name).write_text("1 2\n", encoding="utf-8")
    (manual / "EC2" / "notes.txt").write_text("x", encoding="utf-8")

    assert [
        record.relative_path for record in scan_deleted_spectra(tmp_path / "cosmic_data", _mapping())
    ] == [
        "20240101/EC2/cell1.arc_data",
        "20240101/EC10/cell2.arc_data",
        "20240101/EC10/cell10.arc_data",
    ]
    with_bead = scan_deleted_spectra(tmp_path / "cosmic_data", _mapping(), include_bead=True)
    assert [record.relative_path for record in with_bead] == [
        "20240101/EC2/cell1.arc_data",
        "20240101/EC10/cell2.arc_data",
        "20240101/EC10/cell10.arc_data",
        "20240101/小球/bead1.arc_data",
    ]


def test_restore_spectra_does_not_read_folder_delete(tmp_path: Path) -> None:
    archive = tmp_path / "cosmic_data" / "delete" / "folder_delete" / "20240101" / "EC01"
    archive.mkdir(parents=True)
    (archive / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    save_pipeline_state(state_path, PipelineState())

    try:
        restore_deleted_spectra(
            tmp_path / "cosmic_data",
            tmp_path / "shift_data",
            tmp_path / "init",
            state_path,
            _mapping(),
            ["20240101/EC01/cell1.arc_data"],
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("人工恢复接口不应读取 folder_delete")


def test_restore_folder_moves_cosmic_only_and_requeues(tmp_path: Path) -> None:
    """整夹恢复：只动 cosmic、不写 shift、不改仿射参数、不作废审核，并把文件夹放回清洗队列。"""
    date_name = "20240101"
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    (data_root / date_name / "EC01").mkdir(parents=True)
    (data_root / date_name / "EC01" / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")
    archive = data_root / "delete" / "folder_delete" / date_name / "EC02"
    archive.mkdir(parents=True)
    (archive / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")
    (archive / "cell2.arc_data").write_text("2 3\n", encoding="utf-8")
    report_path = _write_approved_report(shift_root, date_name)
    summary_path = _write_applied_report(shift_root, date_name, "EC02")
    state_path = tmp_path / "state.json"
    state = PipelineState()
    state.affine_completed_dates.add(date_name)
    state.cleaning.folder_catalog[date_name] = {"EC01", "EC02"}
    state.cleaning.completed_folder_keys = {f"{date_name}/EC01", f"{date_name}/EC02"}
    refresh_completed_dates(state.cleaning)
    save_pipeline_state(state_path, state)

    result = restore_deleted_folder(
        data_root,
        shift_root,
        tmp_path / "init",
        state_path,
        None,
        f"{date_name}/EC02",
    )

    assert result["status"] == "folder_restored"
    assert result["restored_file_count"] == 2
    assert (data_root / date_name / "EC02" / "cell1.arc_data").is_file()
    assert not archive.exists()
    # 恢复不派生 shift：shift_data 完全由重建入口生成。
    assert not (shift_root / date_name / "EC02").exists()
    # 小球未变 → 仿射参数保持有效；审核只做一次 → 报告与状态都不作废。
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "manual_approved"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "applied"
    restored = load_pipeline_state(state_path)
    assert restored.affine_completed_dates == {date_name}
    # 该文件夹回到待清洗队列，其他文件夹不受影响。
    refresh_completed_dates(restored.cleaning)
    assert restored.cleaning.completed_date_names == set()
    folders = scan_review_folders(
        data_root,
        [date_name],
        restored.cleaning.folder_catalog,
        restored.cleaning.completed_folder_keys,
    )
    assert [folder.folder_name for folder in folders] == ["EC02"]


def test_restore_spectra_writes_cosmic_only_without_audit(tmp_path: Path) -> None:
    """单条恢复：原样写回 cosmic、不写 shift/init、不改仿射参数、不作废审核、不回清洗队列。"""
    date_name = "20240101"
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    init_root = tmp_path / "init"
    (data_root / date_name / "EC01").mkdir(parents=True)
    (data_root / date_name / "EC01" / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")
    manual = data_root / "delete" / "manual" / date_name / "EC01"
    manual.mkdir(parents=True)
    (manual / "cell3.arc_data").write_text("100 1\n200 2\n", encoding="utf-8")
    shift_folder = shift_root / date_name / "EC01"
    shift_folder.mkdir(parents=True)
    report_path = _write_approved_report(shift_root, date_name, scale=2.0, offset=10.0)
    summary_path = _write_applied_report(shift_root, date_name, "EC01")
    init_target = init_root / "Escherichia" / "EC20240101_01"
    init_target.mkdir(parents=True)
    (init_target / "old.arc_data").write_text("120 1\n220 2\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state = PipelineState()
    state.cleaning.folder_catalog[date_name] = {"EC01"}
    state.cleaning.completed_folder_keys = {f"{date_name}/EC01"}
    refresh_completed_dates(state.cleaning)
    state.init_generated_dates.add(date_name)
    save_pipeline_state(state_path, state)

    result = restore_deleted_spectra(
        data_root,
        shift_root,
        init_root,
        state_path,
        _mapping(),
        [f"{date_name}/EC01/cell3.arc_data"],
    )

    assert result["restored_file_count"] == 1
    assert result["targets"] == [f"{date_name}/EC01"]
    # 源谱原样写回 cosmic（不做任何仿射变换），人工删除区清空。
    restored_spectrum = data_root / date_name / "EC01" / "cell3.arc_data"
    assert read_arc_data(restored_spectrum)[0].tolist() == [100.0, 200.0]
    assert not (manual / "cell3.arc_data").exists()
    # 恢复不写 shift、不写 init。
    assert not any(shift_folder.iterdir())
    assert (init_target / "old.arc_data").is_file()
    assert not (init_target / "cell3.arc_data").exists()
    # 仿射参数与审核结论都保持原样（审核只做一次）。
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "manual_approved"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "applied"
    restored_state = load_pipeline_state(state_path)
    assert restored_state.init_generated_dates == set()
    # 清洗完成状态保持不变，不回待清洗队列。
    refresh_completed_dates(restored_state.cleaning)
    assert restored_state.cleaning.completed_date_names == {date_name}
    assert (
        scan_review_folders(
            data_root,
            [date_name],
            restored_state.cleaning.folder_catalog,
            restored_state.cleaning.completed_folder_keys,
        )
        == []
    )


def test_restore_bead_without_affine_only_restores_cosmic_data(tmp_path: Path) -> None:
    data_root = tmp_path / "cosmic_data"
    manual = data_root / "delete" / "manual" / "20240101" / "小球"
    manual.mkdir(parents=True)
    source = manual / "bead1.arc_data"
    source.write_text("600 1\n1000 2\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    save_pipeline_state(state_path, PipelineState())

    result = restore_deleted_spectra(
        data_root,
        tmp_path / "shift_data",
        tmp_path / "init",
        state_path,
        _mapping(),
        ["20240101/小球/bead1.arc_data"],
    )

    assert result["status"] == "bead_restored"
    assert (data_root / "20240101" / "小球" / "bead1.arc_data").is_file()
    assert not source.exists()


def test_restore_bead_invalidates_affine_parameters(tmp_path: Path) -> None:
    """小球谱变化是唯一需要重算仿射参数的场景：报告必须回到 pending。"""
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    manual = data_root / "delete" / "manual" / "20240101" / "小球"
    manual.mkdir(parents=True)
    (manual / "bead1.arc_data").write_text("600 1\n1000 2\n", encoding="utf-8")
    report_path = _write_approved_report(shift_root, "20240101")
    state_path = tmp_path / "state.json"
    state = PipelineState()
    state.affine_completed_dates.add("20240101")
    save_pipeline_state(state_path, state)

    restore_deleted_spectra(
        data_root,
        shift_root,
        tmp_path / "init",
        state_path,
        _mapping(),
        ["20240101/小球/bead1.arc_data"],
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pending"
    assert report["invalidated_reason"] == "restore_invalidated"
    assert load_pipeline_state(state_path).affine_completed_dates == set()


def test_restore_ignores_legacy_compensation(tmp_path: Path) -> None:
    """旧累计补偿不重放，归档谱按当前内容原样恢复。"""
    date_name = "20240101"
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    (data_root / date_name / "EC01").mkdir(parents=True)
    (data_root / date_name / "EC01" / "cell1.arc_data").write_text("102 1\n103 2\n", encoding="utf-8")
    manual = data_root / "delete" / "manual" / date_name / "EC01"
    manual.mkdir(parents=True)
    (manual / "cell3.arc_data").write_text("100 1\n101 2\n", encoding="utf-8")
    affine_dir = shift_root / date_name / "shift"
    affine_dir.mkdir(parents=True)
    (affine_dir / "affine.json").write_text(
        json.dumps(
            {
                "status": "manual_approved",
                "base_scale": 1.0,
                "base_offset": 10.0,
                "folder_compensations": {"EC01": 2.0},
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "state.json"
    save_pipeline_state(state_path, PipelineState())

    restore_deleted_spectra(
        data_root,
        shift_root,
        tmp_path / "init",
        state_path,
        _mapping(),
        [f"{date_name}/EC01/cell3.arc_data"],
    )

    restored = data_root / date_name / "EC01" / "cell3.arc_data"
    assert read_arc_data(restored)[0].tolist() == [100.0, 101.0]
    assert read_arc_data(restored)[1].tolist() == [1.0, 2.0]
