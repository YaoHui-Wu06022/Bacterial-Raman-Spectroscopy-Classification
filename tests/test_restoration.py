from __future__ import annotations

import json
from pathlib import Path

from ramanv2.pipeline.cleaning import refresh_completed_dates, scan_review_folders
from ramanv2.pipeline.restoration import (
    restore_deleted_folder,
    restore_deleted_spectra,
    scan_deleted_folders,
    scan_deleted_spectra,
)
from ramanv2.pipeline.state import PipelineState, load_pipeline_state, save_pipeline_state




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








