from pathlib import Path

import ramanv2.pipeline.calibration as pipeline
from ramanv2.pipeline.calibration import rebuild_shift_dates, run_affine_calibration


def test_pipeline_skips_dates_without_bead_data_and_applies_ready_dates(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "cosmic_data"
    output = tmp_path / "shift_data"
    (source / "20240101" / "小球").mkdir(parents=True)
    (source / "20240102").mkdir(parents=True)
    (source / "20240101" / "小球" / "bead.arc_data").write_text("100 1\n101 2\n", encoding="utf-8")
    calls: list[str] = []

    def analyze(date_dir, output_root):
        calls.append(f"analyze:{Path(date_dir).name}")
        return {"status": "ready", "date": Path(date_dir).name, "shift_dir": str(Path(output_root) / Path(date_dir).name / "shift")}

    def approve(date_dir, output_root):
        calls.append(f"approve:{Path(date_dir).name}")
        return {"status": "manual_approved", "date": Path(date_dir).name}

    monkeypatch.setattr(pipeline, "analyze_date", analyze)
    monkeypatch.setattr(pipeline, "approve_and_apply_date", approve)
    reports = run_affine_calibration(source, ["20240101", "20240102"], output)
    assert calls == ["analyze:20240101", "approve:20240101"]
    assert [item["status"] for item in reports] == ["manual_approved", "skipped"]


def test_pipeline_does_not_apply_needs_review_date(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "cosmic_data"
    (source / "20240101" / "小球").mkdir(parents=True)
    (source / "20240101" / "小球" / "bead.arc_data").write_text("100 1\n101 2\n", encoding="utf-8")
    approved = []
    monkeypatch.setattr(pipeline, "analyze_date", lambda *_: {"status": "needs_review", "errors": ["bad fit"]})
    monkeypatch.setattr(pipeline, "approve_and_apply_date", lambda *args: approved.append(args))
    reports = run_affine_calibration(source, ["20240101"], tmp_path / "shifted")
    assert reports[0]["status"] == "needs_review"
    assert approved == []


def test_pipeline_skips_existing_manual_approved_date(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "cosmic_data"
    output = tmp_path / "shifted"
    bead_dir = source / "20240101" / "小球"
    bead_dir.mkdir(parents=True)
    (bead_dir / "bead.arc_data").write_text("100 1\n101 2\n", encoding="utf-8")
    report_path = output / "20240101" / "shift" / "affine.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"status": "manual_approved", "scale": 1.0, "offset": 0.0}', encoding="utf-8")

    monkeypatch.setattr(pipeline, "analyze_date", lambda *_: (_ for _ in ()).throw(AssertionError("不应重复分析")))
    reports = run_affine_calibration(source, ["20240101"], output)

    assert reports[0]["status"] == "manual_approved"
    assert reports[0]["skipped"] is True


def test_rebuild_shift_does_not_pass_folder_compensation_to_shift_generation(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "cosmic_data"
    (source / "20240101" / "EC01").mkdir(parents=True)
    output = tmp_path / "shifted"
    report_path = output / "20240101" / "shift" / "affine.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        '{"status":"manual_approved","folder_compensations":{"EC01":1.25}}',
        encoding="utf-8",
    )
    received: list[tuple[Path, Path]] = []

    monkeypatch.setattr(pipeline, "analyze_date", lambda *_: {"status": "ready", "date": "20240101"})

    def approve(date_dir, output_root):
        received.append((Path(date_dir), Path(output_root)))
        (Path(output_root) / "20240101" / "shift").mkdir(parents=True)
        return {"status": "manual_approved", "date": "20240101"}

    monkeypatch.setattr(pipeline, "approve_and_apply_date", approve)
    monkeypatch.setattr(pipeline, "publish_directory", lambda *_: None)

    reports = rebuild_shift_dates(source, output, ["20240101"])

    assert reports[0]["status"] == "manual_approved"
    assert len(received) == 1
