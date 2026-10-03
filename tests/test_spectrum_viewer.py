from pathlib import Path

import numpy as np
import pytest

from ramanv2.common.arc_data import write_arc_data
from ramanv2.core.config import InputConfig
from ramanv2.data.config import DataBuildConfig


def _load_viewer():
    import json
    import sys
    import types

    path = Path(__file__).parents[1] / "notebooks" / "spectrum_viewer.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    module = types.ModuleType("spectrum_viewer")
    sys.modules[module.__name__] = module
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"][:3]
        if cell.get("cell_type") == "code"
    )
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def _write_spectrum(path: Path, count: int = 40) -> None:
    wavenumbers = np.linspace(100.0, 139.0, count)
    intensity = 10.0 + 0.2 * wavenumbers + np.sin(wavenumbers / 4.0)
    write_arc_data(path, wavenumbers, intensity)


def test_viewer_import_has_no_plot_side_effect_and_collects_natural_order(tmp_path: Path) -> None:
    viewer = _load_viewer()
    folder = tmp_path / "spectra"
    folder.mkdir()
    _write_spectrum(folder / "cell10.arc_data")
    _write_spectrum(folder / "cell2.arc_data")
    (folder / "ignore.txt").write_text("ignore", encoding="utf-8")

    paths = viewer.collect_folder_spectra(folder)

    assert [path.name for path in paths] == ["cell2.arc_data", "cell10.arc_data"]


def test_find_sample_path_is_stable_and_supports_explicit_path(tmp_path: Path) -> None:
    viewer = _load_viewer()
    folder = tmp_path / "samples"
    folder.mkdir()
    _write_spectrum(folder / "cell1.arc_data")
    _write_spectrum(folder / "cell2.arc_data")

    first = viewer.find_sample_path("GN", None, folder, 46, tmp_path)
    second = viewer.find_sample_path("GN", None, folder, 46, tmp_path)
    explicit = viewer.find_sample_path("GN", folder / "cell1.arc_data", None, None, tmp_path)

    assert first == second
    assert explicit.name == "cell1.arc_data"


def test_prepare_single_spectrum_builds_baseline_and_three_channels(tmp_path: Path) -> None:
    viewer = _load_viewer()
    path = tmp_path / "sample.arc_data"
    _write_spectrum(path, count=40)
    input_config = InputConfig(
        cut_min=100.0,
        cut_max=139.0,
        target_points=40,
        norm_method="snv",
        smooth_use=False,
        d1_use=False,
        bad_bands=(),
    )
    build_values = DataBuildConfig(
        baseline_method="asls",
        baseline_fit_min=100.0,
        baseline_fit_max=139.0,
        baseline_max_iter=3,
        cosmic_ray_profile_ids=(),
    )

    prepared = viewer.prepare_single_spectrum(path, input_config, build_values, "GN")

    assert prepared.corrected_intensity.shape == (40,)
    assert prepared.model_channels.shape == (3, 40)
    assert prepared.cosmic_ray_replaced == 0
    assert not np.allclose(prepared.corrected_intensity, prepared.raw_intensity)


def test_empty_folder_and_missing_spectrum_raise_clear_errors(tmp_path: Path) -> None:
    viewer = _load_viewer()
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match=r"未找到 \.arc_data"):
        viewer.collect_folder_spectra(empty)
    with pytest.raises(FileNotFoundError, match=r"未找到 \.arc_data"):
        viewer.find_sample_path("GN", None, empty, 1, tmp_path)


def test_old_notebooks_removed_and_new_script_present() -> None:
    root = Path(__file__).parents[1] / "notebooks"
    assert (root / "spectrum_viewer.ipynb").is_file()
    assert not (root / "spectrum_viewer.py").exists()
    notebook_text = (root / "spectrum_viewer.ipynb").read_text(encoding="utf-8")
    assert "FOLDER_PATH = SAMPLE_FOLDER" in notebook_text
    assert "plot_raw_spectrum(prepared)" in notebook_text
    assert "plot_baseline_comparison(prepared, input_config.bad_bands)" in notebook_text
    assert "plot_model_input(prepared)" in notebook_text
    assert "plot_folder_spectra(folder)" in notebook_text
    for name in (
        "raw_spectrum_viewer.ipynb",
        "model_input_channel_viewer.ipynb",
        "Cosmic_Ray_and_Baseline_Correction.ipynb",
    ):
        assert not (root / name).exists()
