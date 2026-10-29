"""按当前磁盘重建流水线记录。"""

from __future__ import annotations

import json
from pathlib import Path

from ramanv2.data.catalog import DatasetMapping
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


    # 审核没有 applied 报告可依，只能是待审核。




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
