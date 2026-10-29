from __future__ import annotations

import json
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import (
    initialize_folder_lineage,
    lineage_path,
    list_lineage_folders,
    load_folder_lineage,
    remove_source_folder,
    save_folder_lineage,
    sync_pipeline_state_from_lineage,
)


def _spectrum(path: Path, start: float = 100.0) -> None:
    write_arc_data(path, [start, start + 1.0], [1.0, 2.0])


def _mapping() -> DatasetMapping:
    return DatasetMapping.from_payload({"genera": {"Enterobacter": ["EAB"]}, "datasets": {"GN": ["Enterobacter"]}})






def test_source_folder_delete_archives_and_preserves_downstream(tmp_path: Path) -> None:
    """整夹移除：源谱归档、原 manual 目录并入、下游产物与审核报告保持原样。"""
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    source = cosmic / "20240101" / "EAB01"
    manual = cosmic / "delete" / "manual" / "20240101" / "EAB01"
    _spectrum(source / "remaining.arc_data")
    _spectrum(manual / "already_deleted.arc_data", 110.0)
    initialize_folder_lineage(cosmic, _mapping())
    output = tmp_path / "init" / "Enterobacter" / "EAB20240101_01"
    output.mkdir(parents=True)
    (output / "a.arc_data").write_text("100 1\n101 2\n", encoding="utf-8")
    payload = load_folder_lineage(cosmic)
    payload["folders"]["20240101/EAB01"]["outputs"]["init"] = str(output)
    save_folder_lineage(cosmic, payload)
    run_dir = shift / "audit_runs" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "applied", "folders": ["20240101/EAB01"], "items": []}),
        encoding="utf-8",
    )
    (shift / "20240101" / "shift").mkdir(parents=True)
    (shift / "20240101" / "shift" / "affine.json").write_text("{}", encoding="utf-8")
    _spectrum(shift / "20240101" / "EAB01" / "remaining.arc_data", 110.0)

    result = remove_source_folder(cosmic, "20240101/EAB01", shift_root=shift)

    assert result["status"] == "ready"
    archived = cosmic / "delete" / "folder_delete" / "20240101" / "EAB01"
    assert (archived / "remaining.arc_data").is_file()
    assert (archived / "already_deleted.arc_data").is_file()
    assert not source.exists() and not manual.exists()
    record = load_folder_lineage(cosmic)["folders"]["20240101/EAB01"]
    assert record["folder_status"] == "folder_deleted"
    assert {item["reason"] for item in record["files"].values()} == {"folder_remove"}
    assert list_lineage_folders(cosmic) == []
    # 下游产物与审核报告一并清理；该日期即便已无活动源谱，shift 参数也保留。
    assert output.exists()
    assert run_dir.exists()
    assert (shift / "20240101").exists()








def test_save_lineage_retries_transient_windows_replace_lock(tmp_path: Path, monkeypatch) -> None:
    """目标文件被扫描/另一会话短暂占用时，os.replace 会抛 WinError 5，须退避重试且不留临时文件。"""
    import ramanv2.common.publishing as publishing_module

    root = tmp_path / "cosmic_data"
    root.mkdir()
    original_replace = publishing_module.os.replace
    attempts = 0

    def replace_with_transient_lock(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "access denied")
        original_replace(source, target)

    monkeypatch.setattr(publishing_module.os, "replace", replace_with_transient_lock)
    save_folder_lineage(root, {"version": 1, "source_root": str(root), "folders": {}})

    assert attempts == 3
    assert lineage_path(root).is_file()
    assert not list(lineage_path(root).parent.glob("*.tmp"))
