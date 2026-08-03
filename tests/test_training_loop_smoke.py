from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_temp.core.config import build_config
from raman_temp.training.loop import (
    TrainArtifacts,
    run_train_loop,
)
from raman_temp.training.optimizer import build_loader
from raman_temp.training.spec import build_execution_spec, build_training_spec
from raman_temp.training.diagnostics import write_nonfinite_diagnostic
from raman_temp.training.split import build_global_train_task


class LoopDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.inputs = torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4]],
                [[0.2, 0.2, 0.1, 0.1]],
                [[0.8, 0.9, 0.7, 0.8]],
                [[0.9, 0.7, 0.8, 0.9]],
            ]
        )
        self.labels = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index], f"sample_{index}.arc_data"


class _LoopIndex:
    """为循环冒烟测试提供最小的训练任务数据索引。"""

    def __init__(self, dataset: LoopDataset) -> None:
        self.level_labels = dataset.labels.numpy()
        self.head_names = ["level_1", "leaf"]
        self.head_name_to_idx = {"level_1": 0, "leaf": 1}
        self.num_classes_by_level = {"level_1": 2, "leaf": 2}

    def resolve_level_name(self, level_name: str) -> str:
        return level_name


class LoopModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 3)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor, return_embedding_enable: bool = False):
        embeddings = self.backbone(inputs.mean(dim=2))
        logits = self.head(torch.tanh(embeddings))
        return (logits, embeddings) if return_embedding_enable else logits


def test_train_loop_runs_and_publishes_best_model(tmp_path: Path) -> None:
    config = build_config(
        {
            "epochs": 1,
            "patience": 2,
            "batch_size": 2,
            "use_align_loss": False,
            "use_supcon_loss": False,
            "use_ema": False,
            "resume_training": False,
            "checkpoint_interval": 1,
            "use_amp": False,
            "train_loader_num_workers": 0,
            "val_loader_num_workers": 0,
        }
    )
    train_spec = build_training_spec(config.training, config.execution)
    runtime_spec = build_execution_spec(config.execution)
    dataset = LoopDataset()
    train_loader = build_loader(
        dataset,
        train_spec.train_loader,
        torch.device("cpu"),
    )
    val_loader = build_loader(
        dataset,
        train_spec.validation_loader,
        torch.device("cpu"),
    )
    messages: list[str] = []
    train_task = build_global_train_task(
        _LoopIndex(dataset),
        "leaf",
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
    )
    train_artifacts = TrainArtifacts(
        model_path=tmp_path / "level_model.pt",
        se_stats_path=tmp_path / "level_se.pt",
        checkpoint_path=tmp_path / "level_checkpoint.pt",
        diagnostic_path=tmp_path / "level_numerical_diagnostic.json",
    )
    mode_calls: list[bool] = []
    result = run_train_loop(
        LoopModel(),
        train_loader,
        val_loader,
        train_task,
        train_spec,
        runtime_spec,
        train_artifacts,
        messages.append,
        lambda model: mode_calls.append(model.training),
    )
    assert result.best_epoch == 1
    assert result.model_path.exists()
    assert not result.checkpoint_path.exists()
    assert any("Best model updated" in message for message in messages)
    assert mode_calls == [True]


def test_nonfinite_diagnostic_writes_indented_json(tmp_path: Path) -> None:
    diagnostic_path = tmp_path / "level_numerical_diagnostic.json"
    written_enable = write_nonfinite_diagnostic(
        False,
        diagnostic_path,
        "input",
        1,
        0,
        torch.tensor([[float("nan")]]),
        torch.tensor([[0, 1]]),
        ["sample.arc_data"],
    )
    assert written_enable is True
    text = diagnostic_path.read_text(encoding="utf-8")
    assert "\n  \"stage\":" in text
    assert json.loads(text)["sample_paths"] == ["sample.arc_data"]
