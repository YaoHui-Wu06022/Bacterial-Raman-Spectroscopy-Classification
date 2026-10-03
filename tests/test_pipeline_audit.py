from pathlib import Path

import json

from ramanv2.pipeline.audit import reconcile_audit_completed_dates, run_shift_folder_audit
from ramanv2.pipeline.state import PipelineState


def test_shift_audit_skips_completed_date(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "shift_data"
    (source / "20240101").mkdir(parents=True)
    state = PipelineState(audit_completed_dates={"20240101"})
    called = []

    monkeypatch.setattr(
        "ramanv2.pipeline.audit.run_folder_audit",
        lambda *args, **kwargs: called.append(args) or {},
    )

    reports = run_shift_folder_audit(source, ["20240101"], state)

    assert reports == [
        {
            "date": "20240101",
            "status": "skipped",
            "reason": "already_completed",
            "message": "该日期已经完成文件夹审核，跳过重复审核",
        }
    ]
    assert called == []


def test_reconcile_audit_state_from_applied_reports(tmp_path: Path) -> None:
    shift_root = tmp_path / "shift_data"
    folder_a = shift_root / "20240101" / "EC01"
    folder_b = shift_root / "20240101" / "EC02"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)
    (folder_a / "cell1.arc_data").write_text("a", encoding="utf-8")
    (folder_b / "cell1.arc_data").write_text("b", encoding="utf-8")
    run_dir = shift_root / "audit_runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "applied", "folders": ["20240101/EC01", "20240101/EC02"]}),
        encoding="utf-8",
    )

    state = PipelineState(affine_completed_dates={"20240101"})

    assert reconcile_audit_completed_dates(shift_root, state)
    assert "20240101" in state.audit_completed_dates


def test_reconcile_does_not_mark_partial_reports(tmp_path: Path) -> None:
    shift_root = tmp_path / "shift_data"
    folder_a = shift_root / "20240101" / "EC01"
    folder_b = shift_root / "20240101" / "EC02"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)
    (folder_a / "cell1.arc_data").write_text("a", encoding="utf-8")
    (folder_b / "cell1.arc_data").write_text("b", encoding="utf-8")
    run_dir = shift_root / "audit_runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "applied", "folders": ["20240101/EC01"]}),
        encoding="utf-8",
    )

    state = PipelineState(affine_completed_dates={"20240101"})

    assert not reconcile_audit_completed_dates(shift_root, state)
    assert "20240101" not in state.audit_completed_dates
