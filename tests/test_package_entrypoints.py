import ast
import importlib
import json
from graphlib import TopologicalSorter
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ramanv2.cli.main import main
from ramanv2.data.arc_data import read_arc_data, write_arc_data
from ramanv2.data.datasets import write_dataset_config
from ramanv2.packaging.platform_bundle import build_platform_bundle, restore_data_archive
from ramanv2.spectra.affine import evaluate_pseudo_voigt
from ramanv2.spectra.cosmic import FolderCosmicOptions, remove_folder_cosmic_rays
from ramanv2.ui import app, page_context


ROOT = Path(__file__).resolve().parents[1]


def test_package_dependencies_and_notebook_imports():
    dependencies = {}
    package = ROOT / "ramanv2"
    for source in package.rglob("*.py"):
        owner = source.relative_to(package).parts[0]
        if not owner.endswith(".py"):
            neighbors = dependencies.setdefault(owner, set())
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                modules = (
                    (node.module,) if isinstance(node, ast.ImportFrom)
                    else tuple(item.name for item in node.names) if isinstance(node, ast.Import)
                    else ()
                )
                for module in modules:
                    if module and module.startswith("ramanv2."):
                        imported = module.split(".")[1]
                        if imported != owner:
                            neighbors.add(imported)
    assert "training" not in dependencies["data"] | dependencies["evaluation"]
    assert "evaluation" not in dependencies["analysis"]
    assert "inference" not in dependencies["modeling"]
    assert not ({"training", "inference", "evaluation", "analysis"} & dependencies["experiments"])
    assert set(TopologicalSorter(dependencies).static_order()) == set(dependencies) | {"common", "spectra"}

    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            source = "".join(line for line in cell["source"] if not line.lstrip().startswith(("%", "!")))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ramanv2."):
                    module = importlib.import_module(node.module)
                    assert all(hasattr(module, item.name) for item in node.names), (notebook, node.module)
                elif isinstance(node, ast.Import):
                    for item in node.names:
                        if item.name.startswith("ramanv2."):
                            importlib.import_module(item.name)


def test_spectrum_kernels_match_reference():
    rng = np.random.default_rng(42)
    values = np.tile(np.linspace(1, 4, 64), (10, 1)) + rng.normal(0, 0.01, (10, 64))
    values[0, 30] += 15
    result = remove_folder_cosmic_rays(values, FolderCosmicOptions(enable=True))
    assert result.status == "applied"
    assert result.stats[0].removed_points == 1
    assert result.cleaned[0, 30] == pytest.approx(2.42720183161803)
    assert result.cleaned.sum() == pytest.approx(1599.7820010153791)
    peaks = evaluate_pseudo_voigt(np.array([600., 610., 620.]), 1., 0., 10., 610., 5., 0.5)
    np.testing.assert_allclose(peaks, [1.29419394, 11., 1.29419394])


def test_archive_round_trip_with_unicode_data_root(tmp_path):
    bundle_root = tmp_path / "bundle_source"
    source_package = bundle_root / "ramanv2"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text('"""fixture"""\n', encoding="utf-8")
    dataset = bundle_root / "dataset" / "\u5149\u8c31"
    write_dataset_config(dataset, "Fixture", [], dataset_id="PB", is_default=True)
    axis = np.arange(4, dtype=np.float32)
    for stage in ("init", "init_test"):
        write_arc_data(dataset / stage / "A" / "cell1.arc_data", axis, axis + 1)

    bundle = build_platform_bundle("PB", tmp_path / "archives", bundle_root)
    with ZipFile(bundle["package_archive"]["output_path"]) as archive:
        assert archive.namelist() == ["ramanv2/__init__.py"]
    restore_root = tmp_path / "restored"
    report = restore_data_archive(bundle["data_archive"]["output_path"], restore_root)
    assert report["sample_counts"] == {"init": 1, "init_test": 1}
    restored = restore_root / dataset.relative_to(bundle_root)
    assert (restored / "dataset.json").read_bytes() == (dataset / "dataset.json").read_bytes()
    for stage in ("init", "init_test"):
        data = read_arc_data(restored / stage / "A" / "cell1.arc_data")
        np.testing.assert_array_equal(data, np.stack([axis, axis + 1]))


def test_cli_handlers_and_streamlit_pages(tmp_path, monkeypatch, capsys):
    dataset_root = tmp_path / "dataset"
    dataset = dataset_root / "\u5149\u8c31"
    write_dataset_config(dataset, "Fixture", [], dataset_id="PB", is_default=True)
    build_context = page_context.build_page_context
    monkeypatch.setattr(app, "DATASET_ROOT", dataset_root)
    monkeypatch.setattr(
        app, "build_page_context",
        lambda dataset_id, registry: build_context(dataset_id, registry, tmp_path),
    )
    from ramanv2.data import profiles

    monkeypatch.setattr(profiles, "_load_registry", lambda root: app.load_dataset_registry(dataset_root))
    assert main(["pipeline", "clean-status", "--dataset", "PB"]) == 0
    assert json.loads(capsys.readouterr().out)["affine_completed_dates"] == []
    for command in (["data", "split"], ["infer", "test"], ["pipeline", "upgrade-db"]):
        with pytest.raises(SystemExit) as help_exit:
            main([*command, "--help"])
        assert help_exit.value.code == 0

    for key in ("\u6570\u636e\u6e05\u6d17", "\u4eff\u5c04\u6821\u51c6", "\u6570\u636e\u96c6\u83b7\u53d6", "\u7ed3\u679c\u5206\u6790", "\u6837\u672c\u63a8\u7406"):
        page = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
        page.session_state["active_page"] = key
        page.run()
        assert not page.exception, key


def test_training_controls_on_windows(monkeypatch):
    from ramanv2.core.config import build_config
    from ramanv2.ui.page_views import model_training

    monkeypatch.setattr(model_training, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        model_training, "build_config",
        lambda values: build_config({**values, "val_loader_num_workers": 3}),
    )
    page = AppTest.from_string(
        "from types import SimpleNamespace\n"
        "from ramanv2.ui.page_views.model_training import _render_config_controls\n"
        "context = SimpleNamespace(dataset_id='PB', bad_bands=())\n"
        "hierarchy = SimpleNamespace(label_schema='hierarchy_v1')\n"
        "_render_config_controls('default', context, hierarchy)\n"
    ).run()
    assert not page.exception
    assert page.selectbox(key="training_workers").value == 0
    assert page.selectbox(key="validation_workers").value == 3


def test_preflight_scans_on_request_and_reconfirms_changed_plan(tmp_path, monkeypatch):
    from ramanv2.data.dataset_mapping import DatasetMapping
    from ramanv2.data.datasets import DatasetRegistry, DatasetSpec
    from ramanv2.data.hierarchy_spec import HierarchyLevel, HierarchySpec
    from ramanv2.pipeline.state import PipelineState
    from ramanv2.pipeline.supplement import SupplementPlan
    from ramanv2.ui.page_views import dataset_acquisition

    hierarchy = HierarchySpec(
        (HierarchyLevel("level_1", "group", True), HierarchyLevel("level_2", "prefix")),
        {},
        explicit=True,
    )
    spec = DatasetSpec("PB", "Fixture", tmp_path, tmp_path / "fixture", (), hierarchy)
    registry = DatasetRegistry((spec,), spec.dataset_id)
    context = page_context.build_page_context(spec.dataset_id, registry, tmp_path)
    mapping = DatasetMapping(("prefix",), "prefix", "prefix", {"A": {"prefix": "A"}})
    monkeypatch.setattr(page_context.PageContext, "dataset_mapping", lambda self: mapping)
    monkeypatch.setattr(dataset_acquisition, "load_page_state", lambda _: PipelineState())
    current_plan = SupplementPlan((), (), ())
    scanned = []
    submitted = []

    def scan_plan(*args):
        scanned.append(current_plan)
        return current_plan

    def build(*args, plan, **kwargs):
        submitted.append(plan)
        return {"supplement": {"folder_count": 0, "file_count": 0}}

    monkeypatch.setattr(dataset_acquisition, "prepare_supplement_plan", scan_plan)
    monkeypatch.setattr(dataset_acquisition, "run_full_init", build)
    page = AppTest.from_string(
        "import streamlit as st\n"
        "from ramanv2.ui.page_views.dataset_acquisition import render_dataset_acquisition_page\n"
        "render_dataset_acquisition_page(st.session_state.context)\n"
    )
    page.session_state["context"] = context
    page.run()
    assert not page.exception
    assert not scanned
    assert page.button(key="rebuild_full_init").proto.disabled

    page.button(key="check_supplement_plan").click().run()
    assert scanned == [current_plan]
    assert not page.checkbox(key="full_init_supplement_confirm").proto.disabled
    page.checkbox(key="full_init_supplement_confirm").check().run()
    assert len(scanned) == 1
    assert not page.button(key="rebuild_full_init").proto.disabled

    current_plan = SupplementPlan((), (), ("G/extra",))
    page.button(key="rebuild_full_init").click().run()
    assert not page.exception
    assert scanned[-1] == current_plan
    assert not submitted
    assert not page.checkbox(key="full_init_supplement_confirm").value
    assert page.button(key="rebuild_full_init").proto.disabled

    page.checkbox(key="full_init_supplement_confirm").check().run()
    page.button(key="rebuild_full_init").click().run()
    assert not page.exception
    assert submitted == [current_plan]


def test_sample_inference_loads_only_selected_folder_csv(tmp_path, monkeypatch):
    from ramanv2.ui.page_views import sample_inference

    date_dir = tmp_path / "20240101"
    result = tmp_path / "result" / date_dir.name
    for folder in ("CS01", "CS02"):
        (result / folder).mkdir(parents=True)
        (result / folder / "predictions.csv").write_text("file,label\ncell1,A\n", encoding="utf-8")
    summary = {"folders": [{"folder": name, "predicted_label": "A"} for name in ("CS01", "CS02")]}
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    calls = []
    read_csv = sample_inference.pd.read_csv

    def count_reads(path, *args, **kwargs):
        calls.append(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(sample_inference.pd, "read_csv", count_reads)
    page = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from ramanv2.ui.page_views.sample_inference import _render_unlabeled_result\n"
        "_render_unlabeled_result(Path(st.session_state.date), {})\n"
    )
    page.session_state["date"] = str(date_dir)
    page.run()
    assert not page.exception
    assert calls == []
    page.selectbox[0].select("CS02").run()
    assert not page.exception
    assert calls == [result / "CS02" / "predictions.csv"]
    page.selectbox[0].set_value(None).run()
    assert len(calls) == 1


def test_level_summary_reads_only_selected_result(tmp_path, monkeypatch):
    from ramanv2.ui.page_views import result_analysis

    mode_root = tmp_path / "level_2" / "level_summary" / "parent_known"
    for kind in ("val_result", "test_result"):
        (mode_root / kind).mkdir(parents=True)
        (mode_root / kind / "metrics.json").write_text("{}", encoding="utf-8")
    loaded = []
    monkeypatch.setattr(result_analysis, "_render_confusion", lambda _, path: loaded.append(path.name))
    page = AppTest.from_string(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from ramanv2.ui.page_views.result_analysis import _render_level_summary\n"
        "_render_level_summary(Path(st.session_state.experiment), 'level_2')\n"
    )
    page.session_state["experiment"] = str(tmp_path)
    page.run()
    assert not page.exception
    assert loaded == ["val_result"]
    page.radio(key="level_summary_result_level_2").set_value("test_result").run()
    assert not page.exception
    assert loaded == ["val_result", "test_result"]
