from pathlib import Path

from ramanv2.pipeline.cleaning import (
    ReviewFolder,
    commit_date_items,
    commit_paths,
    scan_review_folders,
)
from ramanv2.pipeline.lineage import load_folder_lineage
from ramanv2.pipeline.state import PipelineState, write_pipeline_state


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


def test_commit_paths_and_date_items_record_removal_reasons(tmp_path: Path) -> None:
    root = tmp_path / "cosmic_data"
    first = "20240101/EC01/cell1.arc_data"
    second = "20240101/EC02/cell1.arc_data"
    write_spectrum(root / first, 1)
    write_spectrum(root / second, 2)
    state = PipelineState()
    state_path = tmp_path / ".raman_pipeline" / "state.json"

    # 指定路径移动：只移动请求的文件。
    result = commit_paths(root, state, state_path, [first], "folder_remove")
    assert result.moved_paths == (first,)
    assert (root / "delete" / "manual" / first).is_file()
    assert (root / second).is_file()

    # 小球谱直接永久删除，不进入人工删除区，也不写可恢复记录。
    bead = "20240101/小球/bead1.arc_data"
    normal = "20240102/EC01/cell1.arc_data"
    write_spectrum(root / bead, 1)
    write_spectrum(root / normal, 2)
    result = commit_paths(root, state, state_path, [bead, normal], "manual_selection")
    assert set(result.moved_paths) == {bead, normal}
    assert not (root / bead).exists()
    assert not (root / "20240101" / "小球").exists()
    assert not (root / "delete" / "manual" / bead).exists()
    assert (root / "delete" / "manual" / normal).is_file()
    assert state.cleaning.committed_paths == {first, normal}

    # 整日删除应写入 date_remove，而不是逐谱的 manual_selection。
    date_root = tmp_path / "date_cosmic_data"
    for path in (first, second):
        write_spectrum(date_root / path, 1)
    folders = [
        ReviewFolder("20240101/EC01", "20240101", "EC01", (first,)),
        ReviewFolder("20240101/EC02", "20240101", "EC02", (second,)),
    ]

    commit_date_items(date_root, PipelineState(), state_path, folders)

    lineage = load_folder_lineage(date_root)
    for key in ("20240101/EC01", "20240101/EC02"):
        record = lineage["folders"][key]
        assert record["folder_status"] == "folder_deleted"
        assert record["history"][-1]["operation"] == "date_remove"
        assert record["files"]["cell1.arc_data"]["reason"] == "date_remove"


def test_save_state_retries_transient_windows_replace_lock(tmp_path: Path, monkeypatch) -> None:
    import ramanv2.common.file_ops as publishing_module

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
    write_pipeline_state(state_path, PipelineState())

    assert attempts == 3
    assert state_path.is_file()
