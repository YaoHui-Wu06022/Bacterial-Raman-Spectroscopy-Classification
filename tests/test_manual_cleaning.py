import json
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


def test_folder_commit_moves_only_its_own_paths(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    first = "20240101/EC01/cell1.arc_data"
    bead = "20240101/小球/bead.arc_data"
    write_spectrum(root / first, 1)
    write_spectrum(root / bead, 2)
    folders = scan_review_folders(root, ["20240101"])
    state = PipelineState()
    state_path = tmp_path / ".raman_pipeline" / "state.json"
    folder = next(item for item in folders if item.folder_name == "EC01")

    result = commit_folder_items(root, state, state_path, folder)

    assert result.moved_paths == (first,)
    assert (root / "delete" / "folder_delete" / first).is_file()
    assert (root / bead).is_file()


def test_date_commit_moves_all_folders_and_drops_records(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    first = "20240101/EC01/cell1.arc_data"
    bead = "20240101/小球/bead.arc_data"
    write_spectrum(root / first, 1)
    write_spectrum(root / bead, 2)
    (root / "20240101" / "EC02").mkdir(parents=True)
    folders = scan_review_folders(root, ["20240101"])
    state = PipelineState()
    state.cleaning.folder_catalog = {"20240101": {"EC01", "EC02", "小球"}}
    state.cleaning.completed_folder_keys = {"20240101/EC01", "20240101/EC02", "20240101/小球"}
    state.cleaning.completed_date_names = {"20240101"}
    state_path = tmp_path / ".raman_pipeline" / "state.json"

    result = commit_date_items(root, state, state_path, tuple(folders))

    # 整日期提交覆盖普通文件夹与小球，清空空目录并同步丢弃该日期的清洗记录。
    assert result.moved_paths == (first, bead)
    assert (root / "delete" / "manual" / first).is_file()
    assert (root / "delete" / "manual" / bead).is_file()
    assert not (root / "20240101").exists()
    assert "20240101" not in state.cleaning.folder_catalog
    assert "20240101" not in state.cleaning.completed_date_names
    assert not state.cleaning.completed_folder_keys


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


def test_commit_paths_preserves_shift_and_audit_marks_init_pending(tmp_path: Path) -> None:
    """已校准日期照常清洗：shift 副本保留、init 待重建，但审核只做一次，结论保持完成。"""
    from ramanv2.pipeline.lineage import initialize_folder_lineage, record_affine_approval

    root = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    relative_path = "20240101/EC01/cell1.arc_data"
    write_spectrum(root / relative_path, 1)
    write_spectrum(shift / "20240101" / "EC01" / "cell1.arc_data", 1)
    initialize_folder_lineage(root)
    record_affine_approval(root, "20240101", "v1", 1.0, 0.0)
    run_dir = shift / "audit_runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "applied", "folders": ["20240101/EC01"]}),
        encoding="utf-8",
    )
    state = PipelineState()
    state.init_generated_dates.add("20240101")
    state_path = tmp_path / ".raman_pipeline" / "state.json"

    result = commit_paths(root, state, state_path, [relative_path])

    assert result.moved_paths == (relative_path,)
    assert (root / "delete" / "manual" / relative_path).is_file()
    assert (shift / "20240101" / "EC01" / "cell1.arc_data").exists()
    # 清洗删除不作废审核结论：审核只做一次，报告与状态保持完成。
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "applied"
    assert "20240101" not in state.init_generated_dates


def test_delete_state_write_failure_restores_source(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from ramanv2.pipeline import cleaning

    root = tmp_path / "cosmic_data"
    relative = "20240101/EC01/a.arc_data"
    write_spectrum(root / relative, 1)
    state_path = tmp_path / "state.json"
    state = PipelineState()
    before = (root / relative).read_bytes()

    def fail(*args):
        raise OSError("state write failed")

    monkeypatch.setattr(cleaning, "save_pipeline_state", fail)
    with pytest.raises(OSError, match="state write failed"):
        commit_paths(root, state, state_path, [relative])
    assert (root / relative).read_bytes() == before
    assert not (root / "delete" / "manual" / relative).exists()
    assert state.cleaning.committed_paths == set()


def test_folder_delete_state_write_failure_restores_archives(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from ramanv2.pipeline import cleaning

    root = tmp_path / "cosmic_data"
    relative = "20240101/EC01/a.arc_data"
    write_spectrum(root / relative, 1)
    write_spectrum(root / "delete" / "manual" / "20240101/EC01/b.arc_data", 2)
    folder = scan_review_folders(root, ["20240101"])[0]
    state = PipelineState()

    def fail(*args):
        raise OSError("state write failed")

    monkeypatch.setattr(cleaning, "save_pipeline_state", fail)
    with pytest.raises(OSError, match="state write failed"):
        commit_folder_items(root, state, tmp_path / "state.json", folder)
    assert (root / relative).is_file()
    assert (root / "delete" / "manual" / "20240101/EC01/b.arc_data").is_file()
    assert not (root / "delete" / "folder_delete" / "20240101/EC01/a.arc_data").exists()
    assert state.cleaning.committed_paths == set()


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


def test_cleaning_page_selects_and_archives_folder_between_restore_sections(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ramanv2.ui.pages import cleaning

    root = tmp_path / "cosmic_data"
    for name in ("EC01", "EC02"):
        write_spectrum(root / "20240101" / name / "a.arc_data", 1)
    write_spectrum(root / "delete/manual/20240101/EC02/b.arc_data", 2)
    state = PipelineState()
    state.cleaning.folder_catalog = {"20240101": {"EC01", "EC02"}}
    state.cleaning.completed_folder_keys = {"20240101/EC01", "20240101/EC02"}
    state.init_generated_dates = {"20240101"}
    state.dataset_builds["full_init"] = {"status": "ready"}
    state_file = tmp_path / ".raman_pipeline/state.json"
    save_pipeline_state(state_file, state)
    outputs = [tmp_path / name / "existing.arc_data" for name in ("shift_data", "init", "GN")]
    for path in outputs:
        write_spectrum(path, 3)
    before = {path: path.read_bytes() for path in outputs}
    monkeypatch.setattr(cleaning, "_deleted_records", lambda *args: ())
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.ui.pages.cleaning import render_cleaning_page
root = Path({str(tmp_path)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", state_file=root / ".raman_pipeline/state.json", shift_root=root / "shift_data", init_root=root / "init", classification_workbook=root / "mapping.xlsx", dataset_mapping_file=root / "mapping.json")
render_cleaning_page(context)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert [section.label for section in app.expander] == ["整夹删除恢复", "文件夹移除", "删除数据恢复"]
    app.selectbox(key="cleaning_remove_source_folder").select("20240101/EC02").run()
    app.button(key="cleaning_remove_folder_confirm").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert not (root / "20240101/EC02").exists()
    assert (root / "20240101/EC01/a.arc_data").is_file()
    archive = root / "delete/folder_delete/20240101/EC02"
    assert {path.name for path in archive.glob("*.arc_data")} == {"a.arc_data", "b.arc_data"}
    assert "20240101/EC02" not in app.selectbox(key="cleaning_remove_source_folder").options
    assert {path: path.read_bytes() for path in outputs} == before
    saved = load_pipeline_state(state_file)
    assert "20240101" not in saved.init_generated_dates
    assert saved.dataset_builds["full_init"]["status"] != "ready"
