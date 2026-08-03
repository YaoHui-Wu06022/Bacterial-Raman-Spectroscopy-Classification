from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_temp.analysis.integrated_gradients import compute_integrated_gradients
from raman_temp.analysis.layer_attribution import compute_layer_attribution
from raman_temp.analysis import runner


class AnalysisModel(torch.nn.Module):
    """用于归因组件测试的极小一维分类网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Conv1d(1, 2, kernel_size=1)
        self.head = torch.nn.Linear(2, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.layer1(values).mean(dim=2)
        return self.head(features)


def test_ig_and_layer_attribution_return_normalized_values() -> None:
    torch.manual_seed(3)
    model = AnalysisModel()
    inputs = torch.randn(4, 1, 8)
    labels = torch.tensor([0, 1, 0, 1])

    result = compute_integrated_gradients(
        model,
        inputs,
        labels,
        batch_size=2,
        ig_steps=4,
        max_per_class=2,
        class_count=2,
    )
    layers = compute_layer_attribution(model, inputs[:2], labels[:2])

    assert result.channel_importance.shape == (1,)
    assert np.isclose(result.channel_importance.sum(), 1.0)
    assert result.band_importance.shape == (2, 8)
    assert result.sample_counts.tolist() == [2, 2]
    assert result.mean_spectra.shape == (2, 8)
    assert set(layers) == {"layer1"}
    assert np.isclose(sum(layers.values()), 1.0)


def test_runner_writes_single_run_reports_without_umap_dependency(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnalysisModel()
    inputs = torch.randn(4, 1, 8)
    labels = torch.tensor([0, 1, 0, 1])
    task = SimpleNamespace(
        level_name="level_1",
        parent_id=None,
        class_ids=(0, 1),
        class_names=("A", "B"),
        run_dir=str(tmp_path / "run_demo"),
        entry={},
        train_indices=np.asarray([0, 1, 2, 3]),
        validation_indices=np.asarray([0, 1]),
    )
    context = SimpleNamespace(
        config=SimpleNamespace(
            analysis=SimpleNamespace(
                attribution_split="train",
                attribution_batch_count=2,
                ig_steps=4,
                max_per_class=2,
                row_norm="max",
                inherit_missing_levels_use=False,
            ),
            training=SimpleNamespace(batch_size=2),
            execution=SimpleNamespace(seed=42),
        )
    )

    class PredictorStub:
        def load_model(self, level_name, entry, parent_id):
            return model

    monkeypatch.setattr(runner, "load_predictor", lambda *args: PredictorStub())
    monkeypatch.setattr(runner, "collect_task_inputs", lambda *args: (inputs, labels))
    monkeypatch.setattr(runner, "_save_task_umap", lambda *args: None)

    output_dir = runner._run_tasks(context, [task], "run", "cpu")

    assert (output_dir / "logs" / "analysis_log.txt").is_file()
    assert (output_dir / "figures" / "band_importance_per_class_level_1.csv").is_file()
