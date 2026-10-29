"""从 shift 按原走向更新单组，验证恢复、隔离、状态及失败回滚。"""

import json
from pathlib import Path

import pytest

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.builders.init import build_full_init
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline import dataset
from ramanv2.pipeline.lineage import load_folder_lineage, record_restore
from ramanv2.pipeline.shift import rebuild_shift_data
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


DATE = "20240101"
OTHER_DATE = "20240102"


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def layout(tmp_path):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    for date, cs_name in ((DATE, "EC03"), (OTHER_DATE, "EC04")):
        for folder in ("EC01", "EC02", cs_name, "EC05"):
            write_arc_data(cosmic / date / folder / "a.arc_data", [400, 1000, 2000], [10, 20, 30])
        report = shift / date / "shift/affine.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"status": "manual_approved", "scale": 1, "offset": 0}), encoding="utf-8")
    rebuild_shift_data(cosmic, shift, [DATE, OTHER_DATE])
    report = build_full_init(shift, init, mapping, [DATE, OTHER_DATE], selected_test_folders=[f"{DATE}/EC03", f"{OTHER_DATE}/EC04"])
    state = PipelineState(affine_completed_dates={DATE, OTHER_DATE}, init_generated_dates={DATE, OTHER_DATE}, dataset_builds={"full_init": report})
    state_file = tmp_path / ".raman_pipeline/state.json"
    save_pipeline_state(state_file, state)
    write_arc_data(tmp_path / "GN/init/old.arc_data", [600, 700], [1, 2])
    figure = tmp_path / "fig_init/Escherichia/EC.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"old plot")
    (figure.parent / "OTHER.png").write_bytes(b"unrelated plot")
    return cosmic, shift, init, state, state_file


def _fake_plot(init_root, figure_root, groups):
    assert groups == {("Escherichia", "EC")}
    target = Path(figure_root) / "Escherichia/EC.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"updated plot")
    return [target]


@pytest.mark.parametrize("folder,is_test", [("EC02", False), ("EC03", True), ("EC05", False)])
def test_folder_update_preserves_route_numbering_and_unrelated_files(layout, monkeypatch, folder, is_test):
    cosmic, shift, init, state, state_file = layout
    source_id = f"{DATE}/{folder}"
    record = load_folder_lineage(cosmic)["folders"][source_id]
    target = Path(record["outputs"]["test" if is_test else "init"])
    write_arc_data(cosmic / source_id / "restored.arc_data", [400, 1000, 2000], [11, 21, 31])
    write_arc_data(shift / source_id / "restored.arc_data", [402, 1002, 2002], [11, 21, 31])
    # 旧 shift 中仍残留的删除谱不能重新进入输出。
    write_arc_data(shift / source_id / "deleted.arc_data", [400, 1000, 2000], [90, 90, 90])
    record_restore(cosmic, [f"{source_id}/restored.arc_data"])
    record_restore(cosmic, [f"{DATE}/EC01/a.arc_data"])
    record_restore(cosmic, [f"{OTHER_DATE}/EC01/a.arc_data"])
    state.dataset_builds["full_init"].update(status="pending", pending_dates=[DATE, OTHER_DATE])
    before = _snapshot(cosmic.parent)
    plot_calls = []

    def plot(*args):
        plot_calls.append(args)
        return _fake_plot(*args)

    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", plot)
    result = dataset.run_folder_dataset_update(*layout, source_id)

    assert result["target_dir"] == str(target)
    assert result["test_updated"] == is_test
    assert result["spectrum_count"] == 2
    assert (target / "restored.arc_data").read_bytes() == (shift / source_id / "restored.arc_data").read_bytes()
    assert not (target / "deleted.arc_data").exists()
    assert len(plot_calls) == (0 if is_test else 1)
    allowed = {state_file, cosmic.parent / ".raman_pipeline/folder_lineage.json", *target.rglob("*")}
    if not is_test:
        allowed.add(cosmic.parent / "fig_init/Escherichia/EC.png")
    after = _snapshot(cosmic.parent)
    assert {key: value for key, value in before.items() if cosmic.parent / key not in allowed} == {
        key: value for key, value in after.items() if cosmic.parent / key not in allowed
    }
    lineage = load_folder_lineage(cosmic)["folders"]
    assert not lineage[source_id]["init_pending"]
    assert lineage[f"{DATE}/EC01"]["init_pending"]
    assert state.dataset_builds["full_init"]["pending_dates"] == [DATE, OTHER_DATE]
    report = state.dataset_builds["full_init"]
    assert report["output_files"] == len(list(init.rglob("*.arc_data"))) + len(list((cosmic.parent / "init_test").rglob("*.arc_data")))






















