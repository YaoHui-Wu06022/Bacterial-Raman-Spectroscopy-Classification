from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramanv2.core.config import build_config
from ramanv2.core.config_file import read_yaml_dict
from ramanv2.training.workflow import TrainRequest, run_training


class WorkflowModel(torch.nn.Module):
    """用于 workflow 冒烟测试的轻量分类模型。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 3)
        self.head = torch.nn.Linear(3, num_classes)

    def forward(self, inputs: torch.Tensor, return_embedding_enable: bool = False):
        embeddings = self.backbone(inputs.mean(dim=2))
        logits = self.head(torch.tanh(embeddings))
        return (logits, embeddings) if return_embedding_enable else logits


def test_workflow_writes_global_run_and_uses_plugin_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "dataset" / "train"
    _write_spectra(train_dir / "GenusA" / "SpeciesA", "A", 1.0)
    _write_spectra(train_dir / "GenusB" / "SpeciesB", "B", 3.0)
    config = _build_workflow_config()
    monkeypatch.setattr(
        "ramanv2.training.workflow.build_model",
        lambda num_classes, model_spec: WorkflowModel(num_classes),
    )
    monkeypatch.setattr(
        "ramanv2.training.workflow.get_dataset_dir",
        lambda profile, project_root: tmp_path / "dataset",
    )
    initialized_tasks: list[str] = []
    mode_calls: list[bool] = []
    meta = run_training(
        TrainRequest(
            config,
            "level_1",
            train_per_parent_enable=False,
            experiment_dir=tmp_path / "output",
            initialize_model=lambda model, task: initialized_tasks.append(task.model_tag),
            apply_training_mode=lambda model: mode_calls.append(model.training),
        )
    )

    entry = meta["level_models"]["level_1"]
    output_dir = tmp_path / "output"
    assert initialized_tasks == ["level_1"]
    assert mode_calls == [True]
    assert (output_dir / entry["model_path"]).is_file()
    assert (output_dir / entry["config_path"]).is_file()
    assert (output_dir / entry["resolved_config_path"]).is_file()
    assert entry["train_split_path"] == "train_split.json"
    assert entry["val_split_path"] == "val_split.json"
    assert entry["split_hash"]
    assert "leaf" in meta["class_names_by_level"]
    assert not (output_dir / "class_names.json").exists()


def test_workflow_writes_parent_model_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_dir = tmp_path / "dataset" / "train" / "GenusA"
    _write_spectra(train_dir / "SpeciesA", "A", 1.0)
    _write_spectra(train_dir / "SpeciesB", "B", 3.0)
    config = _build_workflow_config()
    monkeypatch.setattr(
        "ramanv2.training.workflow.build_model",
        lambda num_classes, model_spec: WorkflowModel(num_classes),
    )
    monkeypatch.setattr(
        "ramanv2.training.workflow.get_dataset_dir",
        lambda profile, project_root: tmp_path / "dataset",
    )
    output_dir = tmp_path / "output"
    meta = run_training(
        TrainRequest(
            config,
            "level_2",
            only_parent_name="GenusA",
            experiment_dir=output_dir,
        )
    )

    entry = meta["parent_models"]["level_2"]["0"]
    assert entry["status"] == "trained"
    assert entry["child_ids"] == [0, 1]
    assert (output_dir / entry["model_path"]).is_file()
    model_config = read_yaml_dict(output_dir / entry["config_path"])
    assert model_config["only_parent"] == 0


def test_workflow_accepts_explicit_train_dir_without_a_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "stanford_train"
    _write_spectra(train_dir / "SA01", "S", 1.0)
    _write_spectra(train_dir / "SB01", "T", 3.0)
    config = _build_workflow_config(profile_id="Stanford")
    monkeypatch.setattr(
        "ramanv2.training.workflow.build_model",
        lambda num_classes, model_spec: WorkflowModel(num_classes),
    )

    result = run_training(
        TrainRequest(
            config,
            "level_1",
            train_per_parent_enable=False,
            experiment_dir=tmp_path / "output",
            train_dir=train_dir,
        )
    )

    assert result["level_models"]["level_1"]["status"] == "trained"


def _write_spectra(directory: Path, prefix: str, value: float) -> None:
    """写入一组可切分为 train 与 validation 的测试光谱。"""
    directory.mkdir(parents=True, exist_ok=True)
    wavenumbers = np.arange(8, dtype=np.float32)
    for index in range(2):
        intensities = np.linspace(value + index, value + index + 1.0, 8)
        np.savetxt(
            directory / f"{prefix}{index:02d}.arc_data",
            np.column_stack([wavenumbers, intensities]),
        )


def _build_workflow_config(profile_id: str = "GN"):
    """创建禁用随机增强的最小 CPU 训练配置。"""
    return build_config(
        {
            "profile_id": profile_id,
            "target_points": 8,
            "smooth_use": False,
            "d1_use": False,
            "epochs": 1,
            "patience": 2,
            "batch_size": 2,
            "use_gpu": False,
            "use_amp": False,
            "use_align_loss": False,
            "use_supcon_loss": False,
            "use_ema": False,
            "resume_training": False,
            "train_loader_num_workers": 0,
            "val_loader_num_workers": 0,
            "p_piecewise_gain": 0.0,
            "p_noise": 0.0,
            "p_axis": 0.0,
            "p_baseline_weak": 0.0,
            "p_baseline_strong": 0.0,
            "p_cut": 0.0,
        }
    )
