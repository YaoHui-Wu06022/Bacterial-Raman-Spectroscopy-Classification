"""CSdata 单组直接移回 init，不重算光谱，保留路由且可回滚。"""

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
SOURCE = f"{DATE}/CS01EC"


def _snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def layout(tmp_path):
    cosmic, shift, init = (tmp_path / name for name in ("cosmic_data", "shift_data", "init"))
    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    for folder in ("EC01", "CS01EC", "CS02EC"):
        write_arc_data(cosmic / DATE / folder / "a.arc_data", [400, 1000, 2000], [10, 20, 30])
    affine = shift / DATE / "shift/affine.json"
    affine.parent.mkdir(parents=True)
    affine.write_text(json.dumps({"status": "manual_approved", "scale": 1, "offset": 0}), encoding="utf-8")
    rebuild_shift_data(cosmic, shift, [DATE])
    report = build_full_init(shift, init, mapping, [DATE], selected_cs_folders=[SOURCE, f"{DATE}/CS02EC"])
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


def test_cs_return_moves_only_selected_group_and_updates_future_route(layout, monkeypatch):
    cosmic, init, mapping, state, state_file = layout
    root = cosmic.parent
    source = root / "CSdata/CS01EC"
    original = (source / "a.arc_data").read_bytes()
    before = _snapshot(root)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    monkeypatch.setattr(dataset, "build_spectrum_folder", lambda *args: pytest.fail("must not rebuild"))
    assert dataset.collect_cs_return_sources(init, state) == [SOURCE, f"{DATE}/CS02EC"]
    result = dataset.run_cs_to_init(*layout, SOURCE)
    target = init / "Escherichia" / f"EC{DATE}_02"
    assert result["target_dir"] == str(target)
    assert result["spectrum_count"] == 1
    assert (target / "a.arc_data").read_bytes() == original
    assert not source.exists()
    report = state.dataset_builds["full_init"]
    assert report["selected_cs_folders"] == [f"{DATE}/CS02EC"]
    assert [entry["source"] for entry in report["cs_exports"]] == [f"{DATE}/CS02EC"]
    assert report["count"]["file_count"] == 2
    assert report["cs_output_files"] == 1
    assert report["output_files"] == 3
    assert report["date_stats"][DATE]["test_files"] == 1
    record = load_folder_lineage(cosmic)["folders"][SOURCE]
    assert record["outputs"]["init"] == str(target)
    assert not record["outputs"].get("csdata")
    sync_pipeline_state_from_lineage(cosmic, state)
    assert DATE in state.init_generated_dates
    assert dataset.collect_cs_return_sources(init, state) == [f"{DATE}/CS02EC"]
    allowed = {".raman_pipeline/state.json", ".raman_pipeline/folder_lineage.json",
               "fig_init/Escherichia/EC.png", "CSdata/CS01EC/a.arc_data",
               f"init/Escherichia/EC{DATE}_02/a.arc_data"}
    after = _snapshot(root)
    assert {k: v for k, v in before.items() if k not in allowed} == {
        k: v for k, v in after.items() if k not in allowed
    }
    assert state.dataset_builds["profiles"] == {"GN": {"status": "ready"}}


@pytest.mark.parametrize("failure", ["move", "plot", "lineage", "state"])
def test_cs_return_failure_restores_folders_state_and_plot(layout, monkeypatch, failure):
    cosmic, init, mapping, state, state_file = layout
    before = _snapshot(cosmic.parent)
    memory_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("injected return failure")

    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    attribute = {"move": "rename_directory", "plot": "build_prefix_plots_for_groups",
                 "lineage": "save_folder_lineage", "state": "save_pipeline_state"}[failure]
    monkeypatch.setattr(dataset, attribute, fail)
    with pytest.raises(OSError, match="injected return failure"):
        dataset.run_cs_to_init(*layout, SOURCE)
    assert _snapshot(cosmic.parent) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == memory_before
    assert not list(cosmic.parent.rglob("*.tmp"))


def test_cs_return_page_uses_direct_transfer_without_rebuilding(layout, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ramanv2.ui.pages import dataset_acquisition

    cosmic, init, mapping, state, state_file = layout
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)

    def unexpected(*args, **kwargs):
        pytest.fail("must not rebuild datasets")

    for name in ("build_full_init", "build_init_datasets", "build_profile_dataset"):
        monkeypatch.setattr(dataset_acquisition, name, unexpected)
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.state import load_pipeline_state
from ramanv2.ui.pages.dataset_acquisition import _render_cs_return
root = Path({str(cosmic.parent)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", init_root=root / "init", state_file=root / ".raman_pipeline/state.json")
mapping = DatasetMapping(genera={{"Escherichia": ("EC",)}}, datasets={{"GN": ("Escherichia",)}})
_render_cs_return(context, load_pipeline_state(context.state_file), mapping)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    app.selectbox(key="cs_return_source").select(SOURCE).run()
    app.button(key="cs_return_confirm").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any("移回 init" in message.value for message in app.success)
    assert not (cosmic.parent / "CSdata/CS01EC").exists()


@pytest.mark.parametrize("invalid", ["pending", "missing_active", "unknown_species", "duplicate_export", "outside_root", "existing_init"])
def test_cs_return_preflight_preserves_every_file(layout, invalid):
    from ramanv2.pipeline.lineage import save_folder_lineage

    cosmic, init, mapping, state, state_file = layout
    payload = load_folder_lineage(cosmic)
    record = payload["folders"][SOURCE]
    if invalid == "pending":
        record["init_pending"] = True
    elif invalid == "missing_active":
        (cosmic / SOURCE / "a.arc_data").unlink()
    elif invalid == "unknown_species":
        mapping = DatasetMapping(genera={"Escherichia": ("OTHER",)}, datasets={"GN": ("Escherichia",)})
    elif invalid == "duplicate_export":
        state.dataset_builds["full_init"]["cs_exports"].append(dict(state.dataset_builds["full_init"]["cs_exports"][0]))
    elif invalid == "outside_root":
        state.dataset_builds["full_init"]["cs_exports"][0]["target"] = "../init"
    else:
        record["outputs"]["init"] = str(init / "Escherichia" / f"EC{DATE}_01")
    save_folder_lineage(cosmic, payload)
    before = _snapshot(cosmic.parent)
    with pytest.raises(ValueError):
        dataset.run_cs_to_init(cosmic, init, mapping, state, state_file, SOURCE)
    assert _snapshot(cosmic.parent) == before


def test_returned_group_keeps_init_route_during_folder_update(layout, monkeypatch):
    cosmic, init, mapping, state, state_file = layout
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    moved = dataset.run_cs_to_init(*layout, SOURCE)
    with pytest.raises(ValueError, match="不能重复移回"):
        dataset.run_cs_to_init(*layout, SOURCE)
    result = dataset.run_folder_dataset_update(cosmic, cosmic.parent / "shift_data", init, state, state_file, SOURCE)
    assert not result["csdata_updated"]
    assert result["target_dir"] == moved["target_dir"]
    assert not (cosmic.parent / "CSdata/CS01EC").exists()


def test_cs_return_does_not_reuse_reserved_deleted_folder_number(layout, monkeypatch):
    from ramanv2.pipeline.lineage import save_folder_lineage

    cosmic, init, mapping, state, state_file = layout
    payload = load_folder_lineage(cosmic)
    payload["folders"][f"{DATE}/EC_deleted"] = {
        "date": DATE, "folder_status": "folder_deleted",
        "outputs": {"init": str(init / "Escherichia" / f"EC{DATE}_09")},
    }
    save_folder_lineage(cosmic, payload)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    result = dataset.run_cs_to_init(*layout, SOURCE)
    assert Path(result["target_dir"]).name == f"EC{DATE}_10"


@pytest.mark.parametrize("to_csdata_enable", [False, True])
def test_full_page_synchronizes_cs_selection_after_transfer(layout, monkeypatch, to_csdata_enable):
    from streamlit.testing.v1 import AppTest
    from ramanv2.ui.pages import dataset_acquisition

    cosmic, init, mapping, state, state_file = layout
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
    app.multiselect(key="full_init_cs_selection").set_value([SOURCE, f"{DATE}/CS02EC"]).run()
    if to_csdata_enable:
        app.selectbox(key="init_export_source").select_index(0).run()
        app.button(key="init_export_confirm").click().run(timeout=30)
    else:
        app.selectbox(key="cs_return_source").select(SOURCE).run()
        app.button(key="cs_return_confirm").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    expected = {SOURCE, f"{DATE}/CS02EC", f"{DATE}/EC01"} if to_csdata_enable else {f"{DATE}/CS02EC"}
    assert set(app.multiselect(key="full_init_cs_selection").value) == expected
    if to_csdata_enable:
        reopened = AppTest.from_string(script).run(timeout=30)
        assert not reopened.exception
        assert set(reopened.multiselect(key="full_init_cs_selection").value) == expected


def test_cs_return_can_build_real_affected_plot(layout):
    cosmic, init, mapping, state, state_file = layout
    dataset.run_cs_to_init(*layout, SOURCE)
    assert (cosmic.parent / "fig_init/Escherichia/EC.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cs_return_counts_existing_exports_and_keeps_missing_route_for_restore(layout, monkeypatch):
    cosmic, init, mapping, state, state_file = layout
    missing = cosmic.parent / "CSdata/CS02EC"
    (missing / "a.arc_data").unlink()
    missing.rmdir()
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    dataset.run_cs_to_init(*layout, SOURCE)
    report = state.dataset_builds["full_init"]
    assert report["cs_output_files"] == 0
    assert report["cs_folder_count"] == 0
    assert report["cs_exports"][0]["source"] == f"{DATE}/CS02EC"
    assert report["cs_exports"][0]["file_count"] == 0


@pytest.mark.parametrize("source_id", [f"{DATE}/EC01", SOURCE])
def test_init_export_preserves_spectra_and_future_cs_route(layout, monkeypatch, source_id):
    from ramanv2.inference.labels import build_test_input_selection

    cosmic, init, mapping, state, state_file = layout
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    if source_id == SOURCE:
        dataset.run_cs_to_init(*layout, source_id)
    sources = dataset.collect_init_export_sources(cosmic, init, state)
    assert source_id in sources
    source = Path(load_folder_lineage(cosmic)["folders"][source_id]["outputs"]["init"])
    original = (source / "a.arc_data").read_bytes()
    unchanged = {name: _snapshot(cosmic.parent / name) for name in ("cosmic_data", "shift_data", "GN")}
    result = dataset.run_init_to_cs(*layout, source_id)
    target = Path(result["target_dir"])
    assert target.parent == cosmic.parent / "CSdata"
    assert target.name == ("CS01EC" if source_id == SOURCE else f"EC{DATE}_01")
    assert not source.exists()
    assert (target / "a.arc_data").read_bytes() == original
    record = load_folder_lineage(cosmic)["folders"][source_id]
    assert not record["outputs"].get("init")
    assert record["outputs"]["csdata"] == str(target)
    assert source_id in state.dataset_builds["full_init"]["selected_cs_folders"]
    assert source_id not in dataset.collect_init_export_sources(cosmic, init, state)
    assert source_id in dataset.collect_cs_return_sources(init, state)
    selected, _ = build_test_input_selection([target.name], {"class_names_by_level": {
        "level_1": ["Escherichia"], "level_2": ["Escherichia/EC"],
    }}, "level_2", ["Escherichia/EC"])
    assert selected == {target.name}
    updated = dataset.run_folder_dataset_update(cosmic, cosmic.parent / "shift_data", init, state, state_file, source_id)
    assert updated["csdata_updated"]
    assert updated["target_dir"] == str(target)
    assert {name: _snapshot(cosmic.parent / name) for name in unchanged} == unchanged


@pytest.mark.parametrize("mode", ["full", "replace_dates"])
def test_ordinary_init_export_is_retained_during_explicit_rebuild(layout, monkeypatch, mode):
    cosmic, init, mapping, state, state_file = layout
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    moved = dataset.run_init_to_cs(*layout, f"{DATE}/EC01")
    target = Path(moved["target_dir"])
    report = build_init_datasets(cosmic.parent / "shift_data", init, {}, mapping, [DATE],
                                 mode=mode, date_names=[DATE], test_root=cosmic.parent / "CSdata",
                                 selected_test_folders=state.dataset_builds["full_init"]["selected_cs_folders"])
    assert not list(init.rglob("*.arc_data"))
    assert (target / "a.arc_data").is_file()
    assert next(entry for entry in report["cs_exports"] if entry["source"] == f"{DATE}/EC01")["target"] == target.name


def test_export_last_init_group_removes_only_its_obsolete_plot(layout):
    cosmic, init, mapping, state, state_file = layout
    other = cosmic.parent / "fig_init/Escherichia/OTHER.png"
    other.write_bytes(b"other")
    dataset.run_init_to_cs(*layout, f"{DATE}/EC01")
    assert not (cosmic.parent / "fig_init/Escherichia/EC.png").exists()
    assert other.read_bytes() == b"other"
    assert state.dataset_builds["full_init"]["count"]["file_count"] == 0
    assert state.dataset_builds["full_init"]["count"]["genus_count"] == 0


def test_init_export_state_failure_restores_original_folder_and_plot(layout, monkeypatch):
    cosmic, init, mapping, state, state_file = layout
    before = _snapshot(cosmic.parent)
    memory_before = json.dumps(state.to_payload(), sort_keys=True)

    def fail(*args, **kwargs):
        raise OSError("injected export failure")

    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    monkeypatch.setattr(dataset, "save_pipeline_state", fail)
    with pytest.raises(OSError, match="injected export failure"):
        dataset.run_init_to_cs(*layout, f"{DATE}/EC01")
    assert _snapshot(cosmic.parent) == before
    assert json.dumps(state.to_payload(), sort_keys=True) == memory_before


def test_init_export_refuses_existing_csdata_target(layout):
    cosmic, init, mapping, state, state_file = layout
    target = cosmic.parent / "CSdata" / f"EC{DATE}_01"
    target.mkdir()
    (target / "other.arc_data").write_text("600 1\n1000 2\n", encoding="utf-8")
    before = _snapshot(cosmic.parent)
    with pytest.raises(FileExistsError):
        dataset.run_init_to_cs(*layout, f"{DATE}/EC01")
    assert _snapshot(cosmic.parent) == before


def test_ordinary_source_can_move_both_directions_without_recalculation(layout, monkeypatch):
    cosmic, init, mapping, state, state_file = layout
    original = _snapshot(init)
    monkeypatch.setattr(dataset, "build_prefix_plots_for_groups", _fake_plot)
    moved = dataset.run_init_to_cs(*layout, f"{DATE}/EC01")
    returned = dataset.run_cs_to_init(*layout, f"{DATE}/EC01")
    assert not Path(moved["target_dir"]).exists()
    assert _snapshot(init) == original
    assert returned["spectrum_count"] == 1
