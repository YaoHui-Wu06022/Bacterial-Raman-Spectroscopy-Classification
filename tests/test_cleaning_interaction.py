"""选择光谱只更新标记，不扫描全库、改写走向或重复重跑页面。"""

import json

from streamlit.testing.v1 import AppTest

from ramanv2.pipeline.state import PipelineState, load_pipeline_state, save_pipeline_state


def test_spectrum_selection_is_saved_with_one_rerun_and_no_lineage_scan(tmp_path, monkeypatch):
    from ramanv2.pipeline import lineage
    from ramanv2.ui.page_views import cleaning

    root = tmp_path / "cosmic_data"
    relative = "20240101/EC01/cell3.arc_data"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text("600 1\n1000 2\n1800 1\n", encoding="utf-8")
    state = PipelineState()
    state.cleaning.folder_catalog = {"20240101": {"EC01"}}
    state.cleaning.seeded_folder_keys = {"20240101/EC01"}
    state_file = tmp_path / ".raman_pipeline/state.json"
    save_pipeline_state(state_file, state)
    raw_before = source.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("光谱选择页面不应扫描全部源谱或重写走向记录")

    monkeypatch.setattr(lineage, "initialize_folder_lineage", unexpected)
    monkeypatch.setattr(lineage, "save_folder_lineage", unexpected)
    monkeypatch.setattr(cleaning, "_deleted_records", lambda *args: ())
    monkeypatch.setattr(cleaning, "build_spectrum_image", lambda *args: "data:image/png;base64,")
    script = f'''
from pathlib import Path
from types import SimpleNamespace
import streamlit as st
from ramanv2.ui.page_views.cleaning import render_cleaning_page
root = Path({str(tmp_path)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", state_file=root / ".raman_pipeline/state.json", shift_root=root / "shift_data", init_root=root / "init", classification_workbook=root / "mapping.xlsx", dataset_mapping_file=root / "mapping.json")
st.session_state["page_runs"] = st.session_state.get("page_runs", 0) + 1
render_cleaning_page(context)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert app.session_state["page_runs"] == 1
    for index, selected in enumerate(([relative], [], ["20240102/EC01/foreign.arc_data"])):
        grid = next(item for item in app.get("component_instance") if item.key.startswith("pipeline-grid-"))
        widgets = app._tree.get_widget_states()
        widget = widgets.widgets.add()
        widget.id = grid.proto.id
        widget.json_value = json.dumps({"event": "selection", "event_id": str(index), "selected_paths": selected})
        app._run(widgets, timeout=30)
        assert not app.exception
        assert app.session_state["page_runs"] == index + 2
        expected = {relative} if index == 0 else set()
        assert load_pipeline_state(state_file).cleaning.marked_paths == expected
        pending = next(metric for metric in app.metric if metric.label == "当前夹待移除")
        assert pending.value == f"{len(expected)} 条"
        remove = next(button for button in app.button if button.label == "移除选中光谱")
        assert remove.disabled == (not expected)
    assert source.read_bytes() == raw_before
    assert not (tmp_path / ".raman_pipeline/folder_lineage.json").exists()
