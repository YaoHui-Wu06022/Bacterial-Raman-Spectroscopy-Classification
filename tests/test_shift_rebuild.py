from __future__ import annotations

import json
from pathlib import Path

import pytest

from ramanv2.common.arc_data import read_arc_data
from ramanv2.pipeline.shift import load_date_affine_parameters, rebuild_shift_data


DATE = "20240101"


def _write_affine(shift_root: Path, date_name: str = DATE, scale: float = 1.0, offset: float = 10.0) -> None:
    affine_dir = shift_root / date_name / "shift"
    affine_dir.mkdir(parents=True, exist_ok=True)
    (affine_dir / "affine.json").write_text(
        json.dumps({"status": "manual_approved", "base_scale": scale, "base_offset": offset}),
        encoding="utf-8",
    )


def _write_spectrum(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def test_rebuild_requires_confirmed_parameters(tmp_path: Path) -> None:
    """未确认仿射时不写任何派生谱，也不触发参数计算。"""
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    _write_spectrum(data_root / DATE / "EC01" / "cell1.arc_data", "100 1\n")

    assert load_date_affine_parameters(shift_root, DATE) is None
    reports = rebuild_shift_data(data_root, shift_root, [DATE])

    assert [item["status"] for item in reports] == ["needs_calibration"]
    assert not (shift_root / DATE / "EC01").exists()

    _write_affine(shift_root, scale=2.0, offset=5.0)
    assert load_date_affine_parameters(shift_root, DATE) == (2.0, 5.0)


def test_rebuild_replaces_old_derived_spectra_after_generation(tmp_path: Path) -> None:
    """整夹重建：删掉该日期的旧派生谱、失效文件夹与审核归档，再从 cosmic 全量生成。"""
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    _write_affine(shift_root)
    _write_spectrum(data_root / DATE / "EC01" / "cell1.arc_data", "100 1\n200 2\n")
    # 旧派生谱、失效文件夹与审核归档都必须在重建时被整夹清掉。
    _write_spectrum(shift_root / DATE / "EC01" / "cell1.arc_data", "999 1\n")
    _write_spectrum(shift_root / DATE / "EC01" / "stale.arc_data", "999 1\n")
    _write_spectrum(shift_root / DATE / "EC99" / "cell1.arc_data", "110 1\n")
    _write_spectrum(shift_root / "delete" / DATE / "EC01" / "old.arc_data", "110 1\n")

    reports = rebuild_shift_data(data_root, shift_root, [DATE])

    assert [item["status"] for item in reports] == ["rebuilt"]
    assert reports[0]["spectrum_count"] == 1
    assert reports[0]["folders"] == ["EC01"]
    assert read_arc_data(shift_root / DATE / "EC01" / "cell1.arc_data")[0].tolist() == [110.0, 210.0]
    assert not (shift_root / DATE / "EC01" / "stale.arc_data").exists()
    assert not (shift_root / DATE / "EC99").exists()
    # 历史 shift 归档不自动丢弃，也不参与新派生；新删除只发生在 cosmic。
    assert (shift_root / "delete" / DATE / "EC01" / "old.arc_data").is_file()
    # 参数文件与校准图所在目录保留。
    assert (shift_root / DATE / "shift" / "affine.json").is_file()


def test_rebuild_reports_every_date_without_audit(tmp_path: Path) -> None:
    data_root = tmp_path / "cosmic_data"
    shift_root = tmp_path / "shift_data"
    for date_name in ("20240101", "20240102"):
        _write_affine(shift_root, date_name, scale=1.0, offset=0.0)
        _write_spectrum(data_root / date_name / "EC01" / "cell1.arc_data", "100 1\n")
    run_dir = shift_root / "audit_runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "applied", "folders": [f"{DATE}/EC01"]}),
        encoding="utf-8",
    )

    reports = rebuild_shift_data(data_root, shift_root, ["20240102", DATE])

    assert [item["date"] for item in reports] == ["20240101", "20240102"]
    assert all(item["status"] == "rebuilt" for item in reports)
    # 重建重写了派生谱，旧审核结论作废；报告本身保留以便追溯。
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["status"] == "applied"
    assert (run_dir / "summary.json").is_file()


@pytest.mark.parametrize("failure", ["read", "publish"])
def test_failed_rebuild_preserves_previous_date(tmp_path, monkeypatch, failure):
    from ramanv2.pipeline import shift

    cosmic, output = tmp_path / "cosmic_data", tmp_path / "shift_data"
    _write_affine(output)
    _write_spectrum(output / DATE / "EC01" / "old.arc_data", "110 1\n")
    _write_spectrum(cosmic / DATE / "EC01" / "new.arc_data", "100 1\n")
    before = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    if failure == "read":
        _write_spectrum(cosmic / DATE / "EC01" / "new.arc_data", "broken spectrum\n")
    else:
        def fail(*args):
            raise OSError("publish failed")
        monkeypatch.setattr(shift, "publish_directory", fail)
    assert rebuild_shift_data(cosmic, output, [DATE])[0]["status"] == "error"
    assert {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


def test_publish_lock_on_old_directory_never_removes_old_output(tmp_path, monkeypatch):
    from ramanv2.common import publishing

    target, temporary = tmp_path / "output", tmp_path / "temporary"
    _write_spectrum(target / "old.arc_data", "100 1\n")
    _write_spectrum(temporary / "new.arc_data", "200 1\n")

    def locked(*args):
        raise PermissionError("locked")

    monkeypatch.setattr(publishing, "rename_directory", locked)
    with pytest.raises(PermissionError):
        publishing.publish_directory(temporary, target)
    assert (target / "old.arc_data").read_text(encoding="utf-8") == "100 1\n"


def test_readonly_snapshot_cleanup_leaves_no_tmp(tmp_path):
    import stat
    from ramanv2.common.publishing import preserve_paths

    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    original = nested / "a.arc_data"
    original.write_text("100 1\n", encoding="utf-8")
    for path in (original, nested, source):
        path.chmod(stat.S_IREAD)
    try:
        with preserve_paths([source]):
            assert any(tmp_path.glob(".*.tmp"))
        assert original.read_text(encoding="utf-8") == "100 1\n"
        assert not list(tmp_path.glob(".*.tmp"))
    finally:
        for path in (source, nested, original):
            path.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
