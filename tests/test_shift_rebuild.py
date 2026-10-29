from __future__ import annotations

import json
from pathlib import Path


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








