from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramanv2.core.config import build_config
from ramanv2.core.input_spec import build_input_spec
from ramanv2.inference.labels import (
    build_expected_label_lookup,
    build_folder_summary,
    build_test_input_selection,
)
from ramanv2.inference.predictor import Prediction, Predictor, _resolve_input_entry
from ramanv2.inference.spectra import build_inference_preprocessor, preprocess_spectrum_path


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


def test_test_input_selection_excludes_transferred_and_outside_model_labels() -> None:
    meta = {
        "class_names_by_level": {
            "level_1": ["GenusA", "GenusB"],
            "level_2": ["GenusA/AA", "GenusB/BB"],
        }
    }

    selected, rows = build_test_input_selection(
        ["CS01AA", "CS02BB", "CS03CC", "CS04AA"],
        meta,
        "level_1",
        ["GenusA"],
        {"CS04AA"},
    )

    assert selected == {"CS01AA"}
    assert [row["reason"] for row in rows] == [
        "selected",
        "outside_model_label_space",
        "unmapped_species_prefix",
        "transferred_to_alldata",
    ]


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


def test_predict_tensor_directly_inherits_single_child_branch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    meta = {
        "class_names_by_level": {
            "level_1": ["GenusA"],
            "level_2": ["GenusA/SpeciesA"],
        },
        "level_models": {"level_1": {"model_path": "level_1/model.pt"}},
        "parent_models": {
            "level_2": {
                "0": {
                    "model_path": None,
                    "child_ids": [0],
                    "status": "skipped_single_child",
                }
            }
        },
    }
    predictor = Predictor(
        tmp_path,
        None,
        meta,
        "GN",
        None,
        None,
        torch.device("cpu"),
        "level_2",
        ("level_1", "level_2"),
    )
    monkeypatch.setattr(
        predictor,
        "_predict_entry",
        lambda *args: [Prediction("GenusA", 1.0, 0)],
    )

    predictions = predictor.predict_tensor(torch.zeros((1, 1, 16)), top_k=1)

    assert predictions == [Prediction("GenusA/SpeciesA", 1.0, 0)]


def test_resolve_input_entry_uses_available_parent_model(tmp_path: Path) -> None:
    entry = {"model_path": "level_2/parent_0/model.pt"}
    meta = {
        "head_names": ["level_1", "level_2"],
        "level_models": {},
        "parent_models": {"level_2": {"0": entry}},
    }

    result = _resolve_input_entry(meta, tmp_path, None, "level_2")

    assert result is entry


def test_resolve_input_entry_accepts_unregistered_global_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "level_1" / "run_history"
    run_dir.mkdir(parents=True)
    (run_dir / "level_1_model.pt").touch()
    (run_dir / "model_config.yaml").touch()
    (run_dir / "resolved_config.yaml").touch()
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "run.log").touch()
    meta = {
        "head_names": ["level_1"],
        "level_models": {
            "level_1": {
                "run_dir": "level_1/current_run",
                "model_path": "level_1/current_model.pt",
            }
        },
        "parent_models": {},
    }

    entry = _resolve_input_entry(meta, tmp_path, run_dir, "level_1")

    assert entry["run_dir"] == "level_1/run_history"
    assert entry["model_path"] == "level_1/run_history/level_1_model.pt"
