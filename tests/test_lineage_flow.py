from __future__ import annotations

import json
import shutil
from pathlib import Path

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import (
    initialize_folder_lineage,
    list_lineage_folders,
    load_folder_lineage,
    remove_source_folder,
    save_folder_lineage,
    sync_pipeline_state_from_lineage,
)
from ramanv2.pipeline.state import PipelineState


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


def test_sync_ignores_records_whose_data_was_permanently_removed(tmp_path: Path) -> None:
    """走向记录仍在但磁盘已无数据时，日期不能继续出现在仿射摘要中。"""
    cosmic = tmp_path / "cosmic_data"
    _spectrum(cosmic / "20240101" / "EAB01" / "a.arc_data")
    _spectrum(cosmic / "20240102" / "EAB02" / "b.arc_data")
    initialize_folder_lineage(cosmic, _mapping())
    archived = cosmic / "delete" / "manual" / "20240102" / "EAB02"
    archived.parent.mkdir(parents=True)
    shutil.move(str(cosmic / "20240102" / "EAB02"), str(archived))
    payload = load_folder_lineage(cosmic)
    payload["folders"]["20240102/EAB02"]["folder_status"] = "manual_deleted"
    for record in payload["folders"].values():
        record["affine"] = {"status": "approved"}
    save_folder_lineage(cosmic, payload)
    state = PipelineState()

    sync_pipeline_state_from_lineage(cosmic, state)
    assert state.affine_completed_dates == {"20240101", "20240102"}

    shutil.rmtree(archived)
    sync_pipeline_state_from_lineage(cosmic, state)
    assert state.affine_completed_dates == {"20240101"}
