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
    for date, cs_name in ((DATE, "CS01EC"), (OTHER_DATE, "CS02EC")):
        for folder in ("EC01", "EC02", cs_name, "CS03EC"):
            write_arc_data(cosmic / date / folder / "a.arc_data", [400, 1000, 2000], [10, 20, 30])
        report = shift / date / "shift/affine.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"status": "manual_approved", "scale": 1, "offset": 0}), encoding="utf-8")
    rebuild_shift_data(cosmic, shift, [DATE, OTHER_DATE])
    report = build_full_init(shift, init, mapping, [DATE, OTHER_DATE], selected_cs_folders=[f"{DATE}/CS01EC", f"{OTHER_DATE}/CS02EC"])
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


@pytest.mark.parametrize("folder,is_cs", [("EC02", False), ("CS01EC", True), ("CS03EC", False)])
def test_folder_update_preserves_route_numbering_and_unrelated_files(layout, monkeypatch, folder, is_cs):
    cosmic, shift, init, state, state_file = layout
    source_id = f"{DATE}/{folder}"
    record = load_folder_lineage(cosmic)["folders"][source_id]
    target = Path(record["outputs"]["csdata" if is_cs else "init"])
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
    assert result["csdata_updated"] == is_cs
    assert result["spectrum_count"] == 2
    assert (target / "restored.arc_data").read_bytes() == (shift / source_id / "restored.arc_data").read_bytes()
    assert not (target / "deleted.arc_data").exists()
    assert len(plot_calls) == (0 if is_cs else 1)
    allowed = {state_file, cosmic.parent / ".raman_pipeline/folder_lineage.json", *target.rglob("*")}
    if not is_cs:
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
    assert report["output_files"] == len(list(init.rglob("*.arc_data"))) + len(list((cosmic.parent / "CSdata").rglob("*.arc_data")))


def test_last_pending_folder_clears_only_its_date(layout, monkeypatch):
    cosmic, shift, init, state, state_file = layout
    record_restore(cosmic, [f"{DATE}/EC02/a.arc_data"])
    state.dataset_builds["full_init"].update(status="pending", pending_dates=[DATE])
    state.init_generated_dates.discard(DATE)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    dataset.run_folder_dataset_update(*layout, f"{DATE}/EC02")
    assert state.dataset_builds["full_init"]["status"] == "ready"
    assert state.dataset_builds["full_init"]["pending_dates"] == []
    assert DATE in state.init_generated_dates


def test_restored_spectrum_missing_from_shift_blocks_update(layout):
    cosmic, shift, init, state, state_file = layout
    write_arc_data(cosmic / DATE / "EC02/restored.arc_data", [400, 1000, 2000], [1, 2, 3])
    before = _snapshot(cosmic.parent)
    with pytest.raises(ValueError, match="shift"):
        dataset.run_folder_dataset_update(*layout, f"{DATE}/EC02")
    assert _snapshot(cosmic.parent) == before


def test_folder_update_builds_real_plot_and_keeps_other_plot(layout):
    cosmic, shift, init, state, state_file = layout
    other_plot = cosmic.parent / "fig_init/Escherichia/OTHER.png"
    before = (other_plot.read_bytes(), other_plot.stat().st_mtime_ns)
    dataset.run_folder_dataset_update(*layout, f"{DATE}/EC02")
    assert (cosmic.parent / "fig_init/Escherichia/EC.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (other_plot.read_bytes(), other_plot.stat().st_mtime_ns) == before


@pytest.mark.parametrize("invalid", ["unapproved", "missing_route", "duplicate_target", "outside_root"])
def test_invalid_route_or_calibration_leaves_outputs_unchanged(layout, invalid):
    from ramanv2.pipeline.lineage import save_folder_lineage

    cosmic, shift, init, state, state_file = layout
    payload = load_folder_lineage(cosmic)
    record = payload["folders"][f"{DATE}/EC02"]
    if invalid == "unapproved":
        state.affine_completed_dates.discard(DATE)
    elif invalid == "missing_route":
        record["outputs"]["init"] = None
    elif invalid == "duplicate_target":
        record["outputs"]["init"] = payload["folders"][f"{DATE}/EC01"]["outputs"]["init"]
    else:
        record["outputs"]["init"] = str(cosmic / DATE / "EC02")
    save_folder_lineage(cosmic, payload)
    before = _snapshot(cosmic.parent)
    with pytest.raises(ValueError):
        dataset.run_folder_dataset_update(*layout, f"{DATE}/EC02")
    assert _snapshot(cosmic.parent) == before


@pytest.mark.parametrize("failure", ["copy", "publish", "plot", "state"])
def test_folder_update_failure_restores_output_plot_and_state(layout, monkeypatch, failure):
    from ramanv2.data.builders import init

    cosmic, shift, init_root, state, state_file = layout
    write_arc_data(shift / DATE / "EC02/a.arc_data", [402, 1002, 2002], [11, 21, 31])
    before = _snapshot(cosmic.parent)
    state_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("injected folder update failure")

    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    if failure == "copy":
        monkeypatch.setattr(init, "_copy_source_file", fail)
    elif failure == "publish":
        monkeypatch.setattr(init, "publish_directory", fail)
    elif failure == "plot":
        monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", fail)
    else:
        monkeypatch.setattr(dataset, "save_pipeline_state", fail)
    with pytest.raises(OSError, match="injected folder update failure"):
        dataset.run_folder_dataset_update(*layout, f"{DATE}/EC02")
    assert _snapshot(cosmic.parent) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == state_before


@pytest.mark.parametrize("folder", ["EC02", "CS01EC"])
@pytest.mark.parametrize("is_deleted", [False, True])
def test_dataset_page_can_update_selected_folder_while_pending(layout, monkeypatch, folder, is_deleted):
    from streamlit.testing.v1 import AppTest
    from ramanv2.ui.pages import dataset_acquisition

    cosmic, shift, init, state, state_file = layout
    if is_deleted:
        _delete_source_folder(layout, folder)
    else:
        record_restore(cosmic, [f"{DATE}/{folder}/a.arc_data"])
    state.dataset_builds["full_init"].update(status="pending", pending_dates=[DATE])
    save_pipeline_state(state_file, state)
    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    monkeypatch.setattr(dataset_acquisition, "_load_catalog", lambda *args: mapping)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.ui.pages.dataset_acquisition import render_dataset_acquisition_page
root = Path({str(cosmic.parent)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", shift_root=root / "shift_data", init_root=root / "init", state_file=root / ".raman_pipeline/state.json", classification_workbook=root / "mapping.xlsx", dataset_mapping_file=root / "mapping.json")
render_dataset_acquisition_page(context)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    app.selectbox(key="dataset_update_source").select(f"{DATE}/{folder}").run()
    app.button(key="dataset_update_confirm").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any(folder in message.value and ("CSdata" if folder.startswith("CS") else "init") in message.value for message in app.success)
    if is_deleted:
        assert any("已同步整夹删除" in message.value for message in app.success)


def _delete_source_folder(layout, folder):
    from ramanv2.pipeline.cleaning import ReviewFolder, commit_folder_items

    cosmic, shift, init, state, state_file = layout
    source_id = f"{DATE}/{folder}"
    record = load_folder_lineage(cosmic)["folders"][source_id]
    target = Path(record["outputs"]["csdata" if folder == "CS01EC" else "init"])
    commit_folder_items(cosmic, state, state_file, ReviewFolder(source_id, DATE, folder, ()))
    assert rebuild_shift_data(cosmic, shift, [DATE])[0]["status"] == "rebuilt"
    assert not (cosmic / source_id).exists()
    assert not (shift / source_id).exists()
    return source_id, target


@pytest.mark.parametrize("folder,is_cs", [("EC02", False), ("CS01EC", True)])
def test_deleted_folder_can_sync_removal_of_its_old_output(layout, monkeypatch, folder, is_cs):
    cosmic, shift, init, state, state_file = layout
    source_id, target = _delete_source_folder(layout, folder)
    assert source_id in dataset.collect_dataset_source_folders(cosmic, shift, state)
    before = _snapshot(cosmic.parent)
    removed_paths = set(target.rglob("*"))
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    result = dataset.run_folder_dataset_update(*layout, source_id)
    assert result["deleted"]
    assert result["removed_file_count"] == 1
    assert result["csdata_updated"] == is_cs
    assert not target.exists()
    assert source_id not in dataset.collect_dataset_source_folders(cosmic, shift, state)
    assert (cosmic / "delete/folder_delete" / source_id / "a.arc_data").is_file()
    allowed = {state_file, cosmic.parent / ".raman_pipeline/folder_lineage.json", *removed_paths}
    if not is_cs:
        allowed.add(cosmic.parent / "fig_init/Escherichia/EC.png")
    after = _snapshot(cosmic.parent)
    assert {key: value for key, value in before.items() if cosmic.parent / key not in allowed} == {
        key: value for key, value in after.items() if cosmic.parent / key not in allowed
    }
    report = state.dataset_builds["full_init"]
    assert report["output_files"] == len(list(init.rglob("*.arc_data"))) + len(list((cosmic.parent / "CSdata").rglob("*.arc_data")))
    assert report["status"] == "ready"
    assert report["pending_dates"] == []


@pytest.mark.parametrize("failure", ["remove", "plot", "state"])
def test_deleted_folder_sync_failure_restores_output_and_pending_state(layout, monkeypatch, failure):
    source_id, target = _delete_source_folder(layout, "EC02")
    cosmic, shift, init, state, state_file = layout
    before = _snapshot(cosmic.parent)
    state_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("injected deletion sync failure")

    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    monkeypatch.setattr(dataset, {"remove": "_remove_path", "plot": "build_prefix_plots_for_groups", "state": "save_pipeline_state"}[failure], fail)
    with pytest.raises(OSError, match="injected deletion sync failure"):
        dataset.run_folder_dataset_update(*layout, source_id)
    assert _snapshot(cosmic.parent) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == state_before


def test_removing_last_init_group_removes_obsolete_plot_and_updates_summary(layout, monkeypatch):
    import shutil

    cosmic, shift, init, state, state_file = layout
    source_id, target = _delete_source_folder(layout, "EC02")
    for folder in init.glob("*/*"):
        if folder != target:
            shutil.rmtree(folder)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", lambda *args: [])
    dataset.run_folder_dataset_update(*layout, source_id)
    assert state.dataset_builds["full_init"]["count"] == {"exists": True, "genus_count": 0, "folder_count": 0, "file_count": 0}
    assert not (cosmic.parent / "fig_init/Escherichia/EC.png").exists()
    assert (cosmic.parent / "fig_init/Escherichia/OTHER.png").read_bytes() == b"unrelated plot"


def test_restored_deleted_cs_folder_keeps_its_original_export_route(layout):
    from ramanv2.pipeline.restoration import restore_deleted_folder
    from ramanv2.pipeline.state import load_pipeline_state

    cosmic, shift, init, state, state_file = layout
    source_id, target = _delete_source_folder(layout, "CS01EC")
    dataset.run_folder_dataset_update(*layout, source_id)
    assert not target.exists()
    restore_deleted_folder(cosmic, shift, init, state_file, None, source_id)
    state = load_pipeline_state(state_file)
    assert rebuild_shift_data(cosmic, shift, [DATE])[0]["status"] == "rebuilt"
    result = dataset.run_folder_dataset_update(cosmic, shift, init, state, state_file, source_id)
    assert result["csdata_updated"]
    assert result["target_dir"] == str(target)
    assert (target / "a.arc_data").is_file()
