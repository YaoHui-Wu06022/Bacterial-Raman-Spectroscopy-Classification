from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramanv2.core.config import build_config
from ramanv2.core.hierarchy_meta import LABEL_SCHEMA
from ramanv2.evaluation.context import load_evaluation_context, resolve_run_entry
from ramanv2.evaluation.model_eval import _predict_run
from ramanv2.inference.predictor import load_predictor
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
        "ramanv2.training.workflow.resolve_training_dir",
        lambda profile_key, project_root: tmp_path / "dataset" / "train",
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
    run_dir = output_dir / entry["run_dir"]
    manifest = json.loads((run_dir / "train_data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == len(manifest["samples"])
    assert all((run_dir / "train_data" / sample["path"]).is_file() for sample in manifest["samples"])
    assert "leaf" in meta["class_names_by_level"]
    assert not (output_dir / "class_names.json").exists()






def test_species_run_can_train_reload_and_evaluate_across_dates(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    for folder in ("GenusA/AB20231212_01", "GenusA/ab20231214_01", "GenusA/AJO20231214_01"):
        _write_spectra(train_dir / folder, "cell", 1.0)
    config = _build_workflow_config()
    monkeypatch.setattr("ramanv2.training.workflow.build_model", lambda n, spec: WorkflowModel(n))
    monkeypatch.setattr("ramanv2.inference.predictor.build_model", lambda n, spec: WorkflowModel(n))
    monkeypatch.setattr("ramanv2.evaluation.context.resolve_training_dir", lambda profile: train_dir)
    output = tmp_path / "experiment"
    meta = run_training(TrainRequest(
        config, "level_2", train_per_parent_enable=False,
        train_dir=train_dir, experiment_dir=output,
    ))
    assert meta["label_schema"] == LABEL_SCHEMA
    assert meta["class_names_by_level"]["level_2"] == ["GenusA/AB", "GenusA/AJO"]
    assert len(meta["class_names_by_level"]["leaf"]) == 3
    entry = meta["level_models"]["level_2"]
    run = output / entry["run_dir"]
    snapshot = json.loads((run / "train_data/manifest.json").read_text(encoding="utf-8"))
    assert {row["target_class"] for row in snapshot["samples"]} == {"GenusA/AB", "GenusA/AJO"}
    assert all("2023" in row["path"] for row in snapshot["samples"])
    context = load_evaluation_context(run)
    predictor = load_predictor(run, "cpu", "level_2")
    names, paths, labels, predictions = _predict_run(context, predictor, resolve_run_entry(context))
    assert names == ["GenusA/AB", "GenusA/AJO"]
    assert len(paths) == len(labels) == len(predictions) == len(context.validation_indices)

    # 同一实验可补训练另一个层，已经训练的种级条目不得丢失。
    combined = run_training(TrainRequest(
        config, "level_1", train_per_parent_enable=False,
        train_dir=train_dir, experiment_dir=output,
    ))
    assert combined["level_models"]["level_2"] == entry
    assert "level_1" in combined["level_models"]






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
