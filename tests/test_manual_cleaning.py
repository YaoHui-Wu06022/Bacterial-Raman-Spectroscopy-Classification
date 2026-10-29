from pathlib import Path

from ramanv2.pipeline.cleaning import (
    commit_marked_items,
    commit_date_items,
    commit_folder_items,
    commit_paths,
    complete_folder,
    reconcile_cleaning_records,
    folder_has_pending,
    mark_paths,
    refresh_completed_dates,
    reopen_date_review,
    scan_review_folders,
    seed_default_marks,
    toggle_mark_path,
)
from ramanv2.pipeline.state import CleaningState, PipelineState, load_pipeline_state, save_pipeline_state


def write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def test_scan_review_folders_sorts_naturally_and_skips_special_dirs(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    write_spectrum(root / "20240101" / "小球" / "bead.arc_data", 1)
    write_spectrum(root / "20240101" / "EC" / "cell10.arc_data", 10)
    write_spectrum(root / "20240101" / "EC" / "cell2.arc_data", 2)
    write_spectrum(root / "20240101" / "delete" / "manual" / "EC" / "cell1.arc_data", 9)
    write_spectrum(root / "20240102" / "EC" / "cell1.arc_data", 3)

    folders = scan_review_folders(root, date_names=["20240101"])

    # 只扫描指定日期、忽略 delete 等特殊目录，文件夹与文件均按自然序。
    assert [folder.folder_name for folder in folders] == ["EC", "小球"]
    assert [path for folder in folders for path in folder.relative_paths] == [
        "20240101/EC/cell2.arc_data",
        "20240101/EC/cell10.arc_data",
        "20240101/小球/bead.arc_data",
    ]
















def test_commit_paths_moves_only_requested_folder_paths(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    first = "20240101/EC01/cell1.arc_data"
    second = "20240101/EC02/cell1.arc_data"
    write_spectrum(root / first, 1)
    write_spectrum(root / second, 2)
    state = PipelineState()
    state_path = tmp_path / ".raman_pipeline" / "state.json"
    result = commit_paths(root, state, state_path, [first], "folder_remove")
    assert result.moved_paths == (first,)
    assert (root / "delete" / "manual" / first).is_file()
    assert (root / second).is_file()














def test_save_state_retries_transient_windows_replace_lock(tmp_path: Path, monkeypatch) -> None:
    import ramanv2.common.publishing as publishing_module

    state_path = tmp_path / ".raman_pipeline" / "state.json"
    original_replace = publishing_module.os.replace
    attempts = 0

    def replace_with_transient_lock(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "access denied")
        original_replace(source, target)

    monkeypatch.setattr(publishing_module.os, "replace", replace_with_transient_lock)
    save_pipeline_state(state_path, PipelineState())

    assert attempts == 3
    assert state_path.is_file()


