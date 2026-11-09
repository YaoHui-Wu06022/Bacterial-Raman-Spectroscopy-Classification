"""页面渲染与清洗分页行为的回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ramanv2.ui.page_views.cleaning import _apply_page_selection, _page_bounds


PAGE_TARGETS = {
    "cleaning": ("ramanv2.ui.page_views.cleaning", "render_cleaning_page"),
    "affine": ("ramanv2.ui.page_views.affine_calibration", "render_affine_calibration_page"),
    "dataset_acquisition": (
        "ramanv2.ui.page_views.dataset_acquisition",
        "render_dataset_acquisition_page",
    ),
    "model_training": ("ramanv2.ui.page_views.model_training", "render_training_page"),
    "result_analysis": (
        "ramanv2.ui.page_views.result_analysis",
        "render_result_analysis_page",
    ),
    "sample_inference": (
        "ramanv2.ui.page_views.sample_inference",
        "render_sample_inference_page",
    ),
    "dataset_config": ("ramanv2.ui.page_views.dataset_config", "render_dataset_config_page"),
}


def _make_project(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset" / "食源菌"
    (dataset_dir / "cosmic_data").mkdir(parents=True)
    (dataset_dir / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_id": "FBB",
                "display_name": "食源菌",
                "bad_bands": [],
                "aliases": [],
                "default": True,
                "profiles": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "output").mkdir()
    return tmp_path


def _page_script(project_root: str, module_name: str, function_name: str) -> None:
    import importlib
    from pathlib import Path

    from ramanv2.data.datasets import load_dataset_registry
    from ramanv2.ui.page_context import build_page_context

    root = Path(project_root)
    registry = load_dataset_registry(root / "dataset")
    context = build_page_context("FBB", registry, root)
    renderer = getattr(importlib.import_module(module_name), function_name)
    renderer(context)


def _run_page(project_root: Path, module_name: str, function_name: str) -> AppTest:
    app = AppTest.from_function(
        _page_script,
        args=(str(project_root), module_name, function_name),
        default_timeout=30,
    )
    app.run()
    return app


def _write_spectrum(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"100 {value}\n101 {value + 1}\n", encoding="utf-8")


def test_all_pages_render_without_exception(tmp_path: Path) -> None:
    project_root = _make_project(tmp_path)
    for page_name, (module_name, function_name) in PAGE_TARGETS.items():
        app = _run_page(project_root, module_name, function_name)
        assert not app.exception, f"{page_name}: {app.exception}"


def test_cleaning_page_renders_paginated_grid(tmp_path: Path) -> None:
    project_root = _make_project(tmp_path)
    folder = project_root / "dataset" / "食源菌" / "cosmic_data" / "20260101" / "Ecoli01"
    for index in range(10):
        _write_spectrum(folder / f"cell{index}.arc_data", index)

    app = _run_page(
        project_root,
        "ramanv2.ui.page_views.cleaning",
        "render_cleaning_page",
    )

    assert not app.exception
    labels = [button.label for button in app.button]
    assert "上一页" in labels
    assert "下一页" in labels


def test_cleaning_selection_helpers() -> None:
    assert _page_bounds(0, 3) == (1, 0)
    assert _page_bounds(10, 0) == (2, 0)
    assert _page_bounds(10, 5) == (2, 1)

    removed: set[str] = set()
    added: set[str] = set()
    changed = _apply_page_selection(
        {"a", "b"},
        {"b", "c"},
        {"a", "c"},
        remove=removed.update,
        add=added.update,
    )
    assert changed
    assert removed == {"b"}
    assert added == {"a"}
