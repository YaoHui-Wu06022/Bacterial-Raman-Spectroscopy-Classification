from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ramanv2.inference.labels import (
    build_expected_label_lookup,
    build_folder_summary,
    build_test_input_selection,
)
from ramanv2.inference.predictor import Prediction, Predictor, _resolve_input_entry
from ramanv2.inference.routing import predict_routed_labels






def test_test_input_selection_uses_current_model_labels() -> None:
    meta = {
        "class_names_by_level": {
            "level_1": ["GenusA", "GenusB"],
            "level_2": ["GenusA/AA", "GenusB/BB"],
        }
    }

    selected, rows = build_test_input_selection(
        ["AA01", "BB01", "CC01"],
        meta,
        "level_1",
        ["GenusA"],
    )

    assert selected == {"AA01"}
    assert [row["reason"] for row in rows] == [
        "selected",
        "outside_model_label_space",
        "unmapped_species_prefix",
    ]




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


def test_routed_labels_skip_invalid_and_shortcut_single_child() -> None:
    loaded: list[int] = []

    def load_parent(parent_id: int):
        loaded.append(parent_id)
        return object()

    results = predict_routed_labels(
        range(5),
        [1, 2, 3, -1, 1],
        [0, 1, 1, 0, 0],
        {"0": [1], "1": [2, 3]},
        {},
        load_parent,
        lambda predictor, sample_index: 7,
    )

    assert results == [(0, 1, 1), (1, 2, 7), (2, 3, 7), (4, 1, 1)]
    assert loaded == [1]




