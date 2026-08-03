from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_temp.core.config import build_config
from raman_temp.core.input_spec import build_input_spec
from raman_temp.inference.labels import build_expected_label_lookup, build_folder_summary
from raman_temp.inference.spectra import build_inference_preprocessor, preprocess_spectrum_path


def test_expected_lookup_projects_deep_labels_to_target_level() -> None:
    meta = {
        "class_names_by_level": {
            "level_1": ["GenusA"],
            "level_2": ["GenusA/AA", "GenusA/AB"],
        }
    }
    assert build_expected_label_lookup(meta, "level_1") == {
        "GENUSA": "GenusA",
        "AA": "GenusA",
        "AB": "GenusA",
    }
    assert build_expected_label_lookup(meta, "level_2") == {
        "AA": "GenusA/AA",
        "AB": "GenusA/AB",
    }


def test_folder_summary_uses_majority_prediction() -> None:
    rows = [
        {"top1_label": "GenusA"},
        {"top1_label": "GenusA"},
        {"top1_label": "GenusB"},
    ]
    summary = build_folder_summary("CS01AA", "GenusA", ["GenusA", "GenusB"], rows)
    assert summary["predicted_label"] == "GenusA"
    assert summary["correct_count"] == 2
    assert summary["folder_correct"] is True


def test_preprocess_spectrum_path_accepts_bad_band_filtered_length(tmp_path: Path) -> None:
    config = build_config()
    input_spec = build_input_spec(config.input)
    preprocessor = build_inference_preprocessor(input_spec, "cpu")
    full_axis = np.linspace(
        config.input.cut_min,
        config.input.cut_max,
        config.input.target_points,
        dtype=np.float32,
    )
    mask = ~((full_axis >= 890.0) & (full_axis <= 950.0))
    path = tmp_path / "sample.arc_data"
    np.savetxt(path, np.column_stack([full_axis[mask], np.sin(full_axis[mask])]))
    inputs = preprocess_spectrum_path(path, preprocessor, config.input.bad_bands)
    assert tuple(inputs.shape) == (1, input_spec.in_channels, input_spec.point_count)
