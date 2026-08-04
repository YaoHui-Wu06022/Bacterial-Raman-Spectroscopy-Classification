from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramanv2.analysis.integrated_gradients import compute_integrated_gradients
from ramanv2.analysis.layer_attribution import compute_layer_attribution
from ramanv2.analysis import runner


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
                separate_class_plots_use=True,
                inherit_missing_levels_use=False,
            ),
            training=SimpleNamespace(batch_size=2),
            execution=SimpleNamespace(seed=42, use_gpu=False),
            input=SimpleNamespace(
                cut_min=600.0,
                cut_max=1800.0,
                target_points=8,
                bad_bands=(),
            ),
        ),
        input_spec=SimpleNamespace(
            norm_method="snv",
            smooth_enable=False,
            d1_enable=False,
        ),
    )

    class PredictorStub:
        def load_model(self, level_name, entry, parent_id):
            return model

    monkeypatch.setattr(runner, "load_predictor", lambda *args: PredictorStub())
    monkeypatch.setattr(runner, "collect_task_inputs", lambda *args: (inputs, labels))
    monkeypatch.setattr(runner, "_save_task_umap", lambda *args: None)

    output_dir = runner._run_tasks(context, [task], "run", "cpu")

    assert (output_dir / "logs" / "analysis_log.txt").is_file()
    assert (output_dir / "figures" / "band_importance_per_class.csv").is_file()
    assert (output_dir / "figures" / "band_importance_heatmap.png").is_file()
    assert (output_dir / "figures" / "band_importance_heatmap__A.png").is_file()
    log_text = (output_dir / "logs" / "analysis_log.txt").read_text(encoding="utf-8")
    assert "Analysis target: level_1" in log_text
    assert "=== Computing input channel importance and band importance ===" in log_text
    assert "=== Layer Importance (merged by stage) ===" in log_text
    assert "Saved band importance heatmap figures: 2" in log_text


def test_aggregate_analysis_log_uses_aggregate_output_names(tmp_path) -> None:
    tasks = [
        SimpleNamespace(parent_id=0),
        SimpleNamespace(parent_id=1),
    ]
    summaries = [
        {"class_names": ("A",)},
        {"class_names": ("B",)},
    ]
    output_path = tmp_path / "analysis_log.txt"
    figure_dir = tmp_path / "figures"

    runner._write_aggregate_analysis_log(
        summaries,
        output_path,
        tasks,
        "level_2",
        torch.device("cpu"),
        SimpleNamespace(use_gpu=False),
        figure_dir,
        True,
    )

    log_text = output_path.read_text(encoding="utf-8")
    assert "Aggregate analysis for level_2 over 2 parents." in log_text
    assert "layer_importance_aggregate.png" in log_text
    assert "band_importance_per_class_aggregate.csv" in log_text
