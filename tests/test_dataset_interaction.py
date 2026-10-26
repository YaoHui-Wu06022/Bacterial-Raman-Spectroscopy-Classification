"""数据获取页的普通交互只读，不初始化全库或保存清洗状态。"""

import base64
import json

from streamlit.testing.v1 import AppTest

from ramanv2.pipeline.state import PipelineState, save_pipeline_state


def test_dataset_controls_do_not_scan_or_write_lineage_and_datasets(tmp_path, monkeypatch):
    from ramanv2.data.catalog import DatasetMapping
    from ramanv2.pipeline import lineage
    from ramanv2.ui.page_views import dataset_acquisition

    dataset = tmp_path / "dataset"
    cosmic, shift, init = (dataset / name for name in ("cosmic_data", "shift_data", "init"))
    date = "20240101"
    records = {}
    for index, name in enumerate(("EC01", "EC02", "CS01EC"), start=1):
        source_id = f"{date}/{name}"
        for root in (cosmic, shift):
            path = root / source_id / "a.arc_data"
            path.parent.mkdir(parents=True)
            path.write_text("600 1\n1000 2\n1800 1\n", encoding="utf-8")
        target = dataset / "CSdata" / name if name.startswith("CS") else init / "Escherichia" / f"EC{date}_{index:02d}"
        target.mkdir(parents=True)
        (target / "a.arc_data").write_text("600 1\n1000 2\n1800 1\n", encoding="utf-8")
        records[source_id] = {"date": date, "folder": name, "folder_status": "active", "affine": {"status": "approved"}, "outputs": {"csdata" if name.startswith("CS") else "init": str(target)}}
    report = {"status": "ready", "source_dates": [date], "cs_exports": [{"source": f"{date}/CS01EC", "date": date, "target": "CS01EC", "file_count": 1}]}
    state = PipelineState(affine_completed_dates={date}, init_generated_dates={date}, dataset_builds={"full_init": report})
    state.cleaning.folder_catalog = {date: {"EC01", "EC02", "CS01EC"}}
    state_file = dataset / ".raman_pipeline/state.json"
    save_pipeline_state(state_file, state)
    (state_file.parent / "folder_lineage.json").write_text(json.dumps({"version": 1, "folders": records}), encoding="utf-8")
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=")
    for relative in ("Escherichia/EC.png", "Escherichia/OTHER.png", "Bacillus/BA.png"):
        path = dataset / "fig_init" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)

    def unexpected(*args, **kwargs):
        raise AssertionError("普通交互不应扫描初始化、构建产物或保存状态")

    mapping = DatasetMapping(genera={"Escherichia": ("EC",)}, datasets={"GN": ("Escherichia",)})
    monkeypatch.setattr(dataset_acquisition, "_load_catalog", lambda *args: mapping)
    monkeypatch.setattr(dataset_acquisition, "initialize_folder_lineage", unexpected, raising=False)
    monkeypatch.setattr(lineage, "initialize_folder_lineage", unexpected)
    monkeypatch.setattr(lineage, "save_folder_lineage", unexpected)
    for name in ("_save_state", "build_full_init", "build_init_datasets", "build_profile_dataset", "build_prefix_plots", "run_folder_offset", "run_folder_dataset_update"):
        monkeypatch.setattr(dataset_acquisition, name, unexpected)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in dataset.rglob("*") if path.is_file()}
    script = f'''
from pathlib import Path
from types import SimpleNamespace
from ramanv2.ui.page_views.dataset_acquisition import render_dataset_acquisition_page
root = Path({str(dataset)!r})
context = SimpleNamespace(data_root=root / "cosmic_data", shift_root=root / "shift_data", init_root=root / "init", state_file=root / ".raman_pipeline/state.json", classification_workbook=root / "mapping.xlsx", dataset_mapping_file=root / "mapping.json")
render_dataset_acquisition_page(context)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert app.dataframe[0].value.to_dict(orient="records") == [
        {"属": "Escherichia", "种数": 1, "来源文件夹数": 2, "光谱数": 2}
    ]
    assert app.dataframe[1].value.to_dict(orient="records") == [
        {"属": "Escherichia", "种前缀": "EC", "来源文件夹数": 2, "光谱数": 2}
    ]
    app.selectbox(key="folder_offset_source").select(f"{date}/EC02").run()
    app.number_input(key="folder_offset_delta").set_value(4.5).run()
    assert app.number_input(key="folder_offset_delta").value == 4.5
    app.selectbox(key="dataset_update_source").select(f"{date}/EC02").run()
    app.selectbox(key="dataset_profile_name").select("GN").run()
    app.multiselect(key="full_init_cs_selection").select(f"{date}/CS01EC").run()
    app.button(key="full_init_view").click().run()
    app.selectbox(key="full_init_genus").select("Escherichia").run()
    app.selectbox(key="full_init_prefix").select("OTHER").run()
    assert not app.exception
    assert not app.error
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in dataset.rglob("*") if path.is_file()}
    assert after == before
