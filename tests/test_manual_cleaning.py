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
    scan_review_folders,
    seed_default_marks,
    toggle_mark_path,
)
from ramanv2.pipeline.state import CleaningState, PipelineState, load_pipeline_state, save_pipeline_state


def write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def test_scan_uses_natural_sort_and_exact_date_batch(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    write_spectrum(root / "20240101" / "小球" / "bead.arc_data", 1)
    write_spectrum(root / "20240101" / "EC" / "cell10.arc_data", 10)
    write_spectrum(root / "20240101" / "EC" / "cell2.arc_data", 2)
    write_spectrum(root / "20240102" / "EC" / "cell1.arc_data", 3)
    folders = scan_review_folders(root, date_names=["20240101"])
    assert [path for folder in folders for path in folder.relative_paths] == [
        "20240101/EC/cell2.arc_data",
        "20240101/EC/cell10.arc_data",
        "20240101/小球/bead.arc_data",
    ]


def test_scan_ignores_manual_delete_directory(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    write_spectrum(root / "20240101" / "EC01" / "cell3.arc_data", 1)
    write_spectrum(root / "20240101" / "delete" / "manual" / "20240101" / "EC01" / "cell1.arc_data", 2)
    folders = scan_review_folders(root, date_names=["20240101"])
    assert [folder.folder_name for folder in folders] == ["EC01"]


def test_state_round_trip_preserves_v1_catalog_and_completion(tmp_path: Path) -> None:
    path = tmp_path / ".raman_pipeline" / "state.json"
    state = PipelineState()
    state.cleaning.active_date_names = ["20240101"]
    state.cleaning.folder_catalog = {"20240101": {"EC01", "小球"}}
    state.cleaning.completed_folder_keys = {"20240101/EC01", "20240101/小球"}
    state.cleaning.completed_date_names = {"20240101"}
    state.cleaning.marked_paths = {"20240102/EC/cell10.arc_data"}
    save_pipeline_state(path, state)
    loaded = load_pipeline_state(path)
    assert loaded.cleaning.folder_catalog == state.cleaning.folder_catalog
    assert loaded.cleaning.completed_date_names == {"20240101"}
    assert loaded.cleaning.marked_paths == state.cleaning.marked_paths


def test_default_seed_matches_cell1_cell2_case_insensitively_and_skips_bead(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    names = ("CELL1_Area01_000.arc_data", "cell2.arc_data", "cell10.arc_data", "cell3.arc_data")
    for index, name in enumerate(names):
        write_spectrum(root / "20240101" / "EC01" / name, index)
    write_spectrum(root / "20240101" / "小球" / "CELL1_Area01_000.arc_data", 10)
    folders = scan_review_folders(root, date_names=["20240101"])
    state = CleaningState()
    assert seed_default_marks(folders, state) == 2
    assert state.marked_paths == {
        "20240101/EC01/CELL1_Area01_000.arc_data",
        "20240101/EC01/cell2.arc_data",
    }
    assert "20240101/小球" not in state.seeded_folder_keys


def test_commit_removes_empty_folder_and_date_directory(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    relative_path = "20240101/EC01/cell1.arc_data"
    write_spectrum(root / relative_path, 1)
    state = PipelineState()
    state.cleaning.marked_paths = {relative_path}
    state_path = tmp_path / ".raman_pipeline" / "state.json"
    result = commit_marked_items(root, state, state_path)
    assert result.moved_paths == (relative_path,)
    assert (root / "delete" / "manual" / relative_path).is_file()
    assert not (root / "20240101" / "EC01").exists()
    assert not (root / "20240101").exists()


def test_catalog_keeps_removed_folder_visible_for_completion(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    root.mkdir()
    state = CleaningState(folder_catalog={"20240101": {"EC01"}})
    folders = scan_review_folders(root, date_names=["20240101"], folder_catalog=state.folder_catalog)
    assert len(folders) == 1
    complete_folder(state, folders[0])
    assert state.completed_folder_keys == {"20240101/EC01"}
    assert state.completed_date_names == {"20240101"}


def test_completion_is_blocked_until_marked_items_are_committed(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    path = "20240101/EC01/cell1.arc_data"
    write_spectrum(root / path, 1)
    folder = scan_review_folders(root, date_names=["20240101"])[0]
    state = CleaningState(marked_paths={path})
    assert folder_has_pending(state, folder)
    try:
        complete_folder(state, folder)
    except ValueError:
        pass
    else:
        raise AssertionError("有待提交移除项时不应允许完成文件夹")

    mark_paths(state, [], "manual_selection")
    state.marked_paths.clear()
    state.mark_reasons.clear()
    complete_folder(state, folder)
    refresh_completed_dates(state)
    assert state.completed_date_names == {"20240101"}


def test_toggle_mark_path_adds_and_removes_manual_mark() -> None:
    state = CleaningState(marked_paths={"20240101/EC01/cell1.arc_data"})
    assert toggle_mark_path(state, "20240101/EC01/cell1.arc_data") is False
    assert not state.marked_paths
    assert toggle_mark_path(state, "20240101/EC01/cell2.arc_data") is True
    assert state.marked_paths == {"20240101/EC01/cell2.arc_data"}
    assert state.mark_reasons["20240101/EC01/cell2.arc_data"] == {"manual_selection"}


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


def test_folder_and_date_commit_apis_include_expected_scope(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    first = "20240101/EC01/cell1.arc_data"
    bead = "20240101/小球/bead.arc_data"
    write_spectrum(root / first, 1)
    write_spectrum(root / bead, 2)
    folders = scan_review_folders(root, ["20240101"])
    state = PipelineState()
    state_path = tmp_path / ".raman_pipeline" / "state.json"
    folder = next(item for item in folders if item.folder_name == "EC01")
    first_result = commit_folder_items(root, state, state_path, folder)
    second_result = commit_date_items(root, state, state_path, tuple(folders))
    assert first_result.moved_paths == (first,)
    assert second_result.moved_paths == (bead,)


def test_date_commit_removes_empty_source_folders_and_date(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    bead = "20240101/小球/bead.arc_data"
    write_spectrum(root / bead, 2)
    (root / "20240101" / "EC01").mkdir(parents=True)
    folders = scan_review_folders(root, ["20240101"])
    state = PipelineState()
    state_path = tmp_path / ".raman_pipeline" / "state.json"

    result = commit_date_items(root, state, state_path, tuple(folders))

    assert result.moved_paths == (bead,)
    assert not (root / "20240101").exists()
    assert (root / "delete" / "manual" / bead).is_file()


def test_date_removal_drops_cleaning_records(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    bead = "20240101/小球/bead.arc_data"
    write_spectrum(root / bead, 2)
    state = PipelineState()
    state.cleaning.folder_catalog = {"20240101": {"小球"}}
    state.cleaning.completed_folder_keys = {"20240101/小球"}
    state.cleaning.completed_date_names = {"20240101"}
    state_path = tmp_path / ".raman_pipeline" / "state.json"

    commit_date_items(root, state, state_path, tuple(scan_review_folders(root, ["20240101"])))

    assert "20240101" not in state.cleaning.folder_catalog
    assert "20240101" not in state.cleaning.completed_date_names
    assert "20240101/小球" not in state.cleaning.completed_folder_keys


def test_reconcile_drops_missing_folder_record(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    write_spectrum(root / "20240101" / "EC02" / "cell1.arc_data", 1)
    state = CleaningState(
        folder_catalog={"20240101": {"EC01", "EC02"}},
        completed_folder_keys={"20240101/EC01", "20240101/EC02"},
        completed_date_names={"20240101"},
    )

    assert reconcile_cleaning_records(root, state)
    assert state.folder_catalog == {"20240101": {"EC02"}}
    assert state.completed_folder_keys == {"20240101/EC02"}
    assert state.completed_date_names == {"20240101"}


def test_save_state_retries_transient_windows_replace_lock(tmp_path: Path, monkeypatch) -> None:
    import ramanv2.pipeline.state as state_module

    state_path = tmp_path / ".raman_pipeline" / "state.json"
    original_replace = state_module.os.replace
    attempts = 0

    def replace_with_transient_lock(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "access denied")
        original_replace(source, target)

    monkeypatch.setattr(state_module.os, "replace", replace_with_transient_lock)
    save_pipeline_state(state_path, PipelineState())

    assert attempts == 3
    assert state_path.is_file()
