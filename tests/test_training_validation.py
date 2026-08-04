from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman.metrics import compute_classification_metrics as reference_compute_classification_metrics
from raman.runtime import mask_logits_by_parent as reference_mask_logits_by_parent
from raman.runtime import resolve_allowed_indices as reference_resolve_allowed_indices
from raman.runtime import select_level_targets as reference_select_level_targets
from ramanv2.common.metrics import compute_classification_metrics
from ramanv2.core.hierarchy import mask_logits_by_parent, resolve_allowed_indices, select_level_targets
from ramanv2.modeling.layers import SEBlock1D
from ramanv2.training.validation import evaluate_validation_loader


class ValidationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.se = SEBlock1D(2, 1, True, torch.nn.ReLU)
        self.head = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0, -0.5], [-0.2, 0.8]]))
            self.head.bias.copy_(torch.tensor([0.1, -0.1]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.se(inputs).mean(dim=2))


def test_hierarchy_tools_and_metrics_match_reference() -> None:
    labels = torch.tensor([[0, 1], [1, 2], [-1, 0]])
    torch.testing.assert_close(select_level_targets(labels, 1), reference_select_level_targets(labels, 1))
    logits = torch.tensor([[1.0, 0.5, -0.2], [0.3, 0.8, 0.2], [0.1, 0.4, 0.9]])
    parent_labels = torch.tensor([0, 1, -1])
    parent_to_children = {0: [0, 1], "1": [1, 2]}
    expected_logits, expected_mask = reference_mask_logits_by_parent(
        logits,
        parent_labels,
        parent_to_children,
    )
    actual_logits, actual_mask = mask_logits_by_parent(logits, parent_labels, parent_to_children)
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_mask, expected_mask)
    assert resolve_allowed_indices(["A", "B", "C"], ["C", 0, "missing"]) == reference_resolve_allowed_indices(
        ["A", "B", "C"],
        ["C", 0, "missing"],
    )

    targets = np.array([0, 1, 2, 1])
    predictions = np.array([0, 2, 2, 1])
    assert compute_classification_metrics(targets, predictions, range(3)) == reference_compute_classification_metrics(
        targets,
        predictions,
        range(3),
    )


def test_validation_collects_metrics_and_se_stats() -> None:
    model = ValidationModel()
    inputs = torch.tensor(
        [
            [[1.0, 0.5, 0.2], [0.1, 0.3, 0.4]],
            [[0.2, 0.4, 0.1], [1.0, 0.8, 0.9]],
            [[0.5, 0.2, 0.3], [0.4, 0.1, 0.2]],
        ]
    )
    labels = torch.tensor([[0, 0], [0, 1], [-1, 0]])
    loader = [(inputs, labels, None)]
    loss, accuracy, metrics, se_stats = evaluate_validation_loader(
        model,
        loader,
        torch.device("cpu"),
        level_index=1,
        parent_level_index=0,
        parent_to_children={0: [0, 1]},
    )
    assert loss > 0.0
    assert 0.0 <= accuracy <= 1.0
    assert metrics["accuracy"] == accuracy
    assert set(metrics) == {"accuracy", "macro_f1", "macro_recall"}
    assert se_stats["se"]["sample_count"] == 2
    assert se_stats["se"]["channel_mean"].shape == (2,)


def test_validation_returns_zero_metrics_when_no_labels_are_valid() -> None:
    model = ValidationModel()
    inputs = torch.ones((2, 2, 3))
    labels = torch.full((2, 2), -1)
    loss, accuracy, metrics, se_stats = evaluate_validation_loader(
        model,
        [(inputs, labels, None)],
        torch.device("cpu"),
        level_index=1,
    )
    assert loss == 0.0
    assert accuracy == 0.0
    assert metrics == {"accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0}
    assert se_stats == {}
