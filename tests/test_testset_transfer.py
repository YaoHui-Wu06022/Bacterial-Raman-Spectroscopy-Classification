"""init_test 单组直接移回 init，不重算光谱，保留路由且可回滚。"""

import json
from pathlib import Path

import pytest

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.builders.init import build_full_init, build_init_datasets
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline import dataset
from ramanv2.pipeline.lineage import load_folder_lineage, sync_pipeline_state_from_lineage
from ramanv2.pipeline.shift import rebuild_shift_data
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


DATE = "20240101"
# 移入 init_test 的两个普通来源；init_test 目录名与 init 同规则：{prefix}{date}_{序号}。
SOURCE = f"{DATE}/EC02"
SECOND = f"{DATE}/EC03"
# 测试组按前缀在日期内递增编号：EC02 是第一个 EC 测试组，EC03 是第二个。
_TEST_FOLDER_INDEX = {"EC02": 1, "EC03": 2}


def _test_folder_name(source_name: str) -> str:
    """普通来源移入 init_test 后的目录名：EC20240101_01。"""
    return f"EC{DATE}_{_TEST_FOLDER_INDEX[source_name]:02d}"


def _snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def layout(tmp_path):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    for folder in ("EC01", "EC02", "EC03"):
        write_arc_data(cosmic / DATE / folder / "a.arc_data", [400, 1000, 2000], [10, 20, 30])
    affine = shift / DATE / "shift/affine.json"
    affine.parent.mkdir(parents=True)
    affine.write_text(json.dumps({"status": "manual_approved", "scale": 1, "offset": 0}), encoding="utf-8")
    rebuild_shift_data(cosmic, shift, [DATE])
    report = build_full_init(shift, init, mapping, [DATE], selected_test_folders=[SOURCE, SECOND])
    state = PipelineState(affine_completed_dates={DATE}, init_generated_dates={DATE}, dataset_builds={
        "full_init": report, "profiles": {"GN": {"status": "ready"}},
    })
    state_file = tmp_path / ".raman_pipeline/state.json"
    save_pipeline_state(state_file, state)
    write_arc_data(tmp_path / "GN/init/old.arc_data", [600, 700], [1, 2])
    plot = tmp_path / "fig_init/Escherichia/EC.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"old plot")
    return cosmic, init, mapping, state, state_file


def _fake_plot(init, output, groups):
    assert groups == {("Escherichia", "EC")}
    if not any((Path(init) / "Escherichia").rglob("*.arc_data")):
        return []
    target = Path(output) / "Escherichia/EC.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"new plot")
    return [target]


def test_test_return_moves_only_selected_group_and_updates_future_route(layout, monkeypatch):
    cosmic, init, mapping, state, state_file = layout
    root = cosmic.parent
    source = root / "init_test" / _test_folder_name("EC02")
    original = (source / "a.arc_data").read_bytes()
    before = _snapshot(root)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    monkeypatch.setattr(dataset, "build_spectrum_folder", lambda *args: pytest.fail("must not rebuild"))
    assert dataset.collect_test_return_sources(init, state) == [SOURCE, SECOND]
    result = dataset.run_test_to_init(*layout, SOURCE)
    target = init / "Escherichia" / f"EC{DATE}_02"
    assert result["target_dir"] == str(target)
    assert result["spectrum_count"] == 1
    assert (target / "a.arc_data").read_bytes() == original
    assert not source.exists()
    report = state.dataset_builds["full_init"]
    assert report["selected_test_folders"] == [SECOND]
    assert [entry["source"] for entry in report["test_exports"]] == [SECOND]
    assert report["count"]["file_count"] == 2
    assert report["test_output_files"] == 1
    assert report["output_files"] == 3
    assert report["date_stats"][DATE]["test_files"] == 1
    record = load_folder_lineage(cosmic)["folders"][SOURCE]
    assert record["outputs"]["init"] == str(target)
    assert not record["outputs"].get("test")
    sync_pipeline_state_from_lineage(cosmic, state)
    assert DATE in state.init_generated_dates
    assert dataset.collect_test_return_sources(init, state) == [SECOND]
    allowed = {".raman_pipeline/state.json", ".raman_pipeline/folder_lineage.json",
               "fig_init/Escherichia/EC.png", f"init_test/{_test_folder_name('EC02')}/a.arc_data",
               f"init/Escherichia/EC{DATE}_02/a.arc_data"}
    after = _snapshot(root)
    assert {k: v for k, v in before.items() if k not in allowed} == {
        k: v for k, v in after.items() if k not in allowed
    }
    assert state.dataset_builds["profiles"] == {"GN": {"status": "ready"}}




























