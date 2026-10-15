"""按当前磁盘重建流水线记录。"""

from __future__ import annotations

import json
from pathlib import Path

from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import load_folder_lineage
from ramanv2.pipeline.records import rebuild_pipeline_records
from ramanv2.pipeline.state import load_pipeline_state


def _spectrum(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def _approved_report(shift_root: Path, date_name: str) -> None:
    directory = shift_root / date_name / "shift"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "affine.json").write_text(
        json.dumps(
            {
                "status": "manual_approved",
                "scale": 0.99,
                "offset": 13.5,
                "approved_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


def _mapping() -> DatasetMapping:
    return DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    cosmic = tmp_path / "dataset" / "cosmic_data"
    shift = tmp_path / "dataset" / "shift_data"
    state_path = tmp_path / "dataset" / ".raman_pipeline" / "state.json"
    _spectrum(cosmic / "20240101" / "EC01" / "cell1.arc_data")
    _spectrum(cosmic / "20240101" / "EC01" / "cell2.arc_data")
    _spectrum(cosmic / "20240101" / "小球" / "bead1.arc_data")
    _spectrum(cosmic / "delete" / "manual" / "20240101" / "EC01" / "cell3.arc_data")
    _spectrum(cosmic / "delete" / "folder_delete" / "20240102" / "EC02" / "cell1.arc_data")
    _approved_report(shift, "20240101")
    _approved_report(shift, "20240102")
    return cosmic, shift, state_path


def test_rebuild_records_records_deletions_and_affine_from_disk(tmp_path: Path) -> None:
    """删除归档与 affine 报告是唯一事实来源：重建后走向树与页面状态都与磁盘一致。"""
    cosmic, shift, state_path = _layout(tmp_path)

    summary = rebuild_pipeline_records(cosmic, shift, _mapping(), state_path)

    assert summary["status_counts"] == {"manual_deleted": 1, "folder_deleted": 1}
    assert summary["approved_affine_dates"] == 2
    assert summary["affine_backfilled_folders"] == 2
    assert summary["committed_paths"] == 1

    lineage = load_folder_lineage(cosmic)["folders"]
    assert lineage["20240101/EC01"]["folder_status"] == "manual_deleted"
    assert lineage["20240102/EC02"]["folder_status"] == "folder_deleted"
    # 人工删除区的谱逐条记录，且带导入标记。
    deleted = lineage["20240101/EC01"]["files"]["cell3.arc_data"]
    assert deleted["status"] == "manual_deleted"
    assert deleted["reason"] == "imported_history"
    assert deleted["deleted_path"] == "delete/manual/20240101/EC01/cell3.arc_data"
    # 已确认的日期级参数回填到该日期所有文件夹，供下游门禁直接读取。
    assert lineage["20240101/EC01"]["affine"]["status"] == "approved"
    assert lineage["20240102/EC02"]["affine"]["scale"] == 0.99

    state = load_pipeline_state(state_path)
    # 整夹删除的文件夹不再进入清洗目录清单，人工删除区进入提交记录。
    assert {key: sorted(value) for key, value in state.cleaning.folder_catalog.items()} == {
        "20240101": ["EC01"]
    }
    assert state.cleaning.committed_paths == {"20240101/EC01/cell3.arc_data"}
    assert state.cleaning.completed_folder_keys == {"20240101/EC01"}
    assert state.affine_completed_dates == {"20240101", "20240102"}
    # 审核没有 applied 报告可依，只能是待审核。


def test_rebuild_records_clears_affine_when_report_disappears(tmp_path: Path) -> None:
    """重建是磁盘的函数：报告被删后残留的 approved 必须清掉，否则下游误以为参数可用。"""
    cosmic, shift, state_path = _layout(tmp_path)
    rebuild_pipeline_records(cosmic, shift, _mapping(), state_path)
    assert load_pipeline_state(state_path).affine_completed_dates == {"20240101", "20240102"}

    (shift / "20240101" / "shift" / "affine.json").unlink()
    summary = rebuild_pipeline_records(cosmic, shift, _mapping(), state_path)

    lineage = load_folder_lineage(cosmic)["folders"]
    assert lineage["20240101/EC01"]["affine"] == {
        "status": "pending",
        "version": None,
        "scale": None,
        "offset": None,
    }
    assert summary["affine_backfilled_folders"] == 1
    assert load_pipeline_state(state_path).affine_completed_dates == {"20240102"}


def test_rebuild_records_is_idempotent(tmp_path: Path) -> None:
    """重复重建不改变结果，也不引入临时文件残留。"""
    cosmic, shift, state_path = _layout(tmp_path)

    first = rebuild_pipeline_records(cosmic, shift, _mapping(), state_path)
    second = rebuild_pipeline_records(cosmic, shift, _mapping(), state_path)

    for key in ("status_counts", "date_count", "committed_paths", "affine_backfilled_folders"):
        assert first[key] == second[key]
    assert load_pipeline_state(state_path).cleaning.folder_catalog == {
        "20240101": {"EC01"}
    }
    assert not list(state_path.parent.glob("*.tmp"))
    assert not list(state_path.parent.glob(".*.tmp"))
