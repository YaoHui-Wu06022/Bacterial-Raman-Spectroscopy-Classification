from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramanv2.core.config import build_config
from ramanv2.inference.runner import evaluate_test_set
from ramanv2.training.workflow import TrainRequest, run_training


class SmokeModel(torch.nn.Module):
    """测试用轻量分类模型：只依赖输入均值，保证 CPU 上秒级收敛。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 3)
        self.head = torch.nn.Linear(3, num_classes)

    def forward(self, inputs: torch.Tensor, return_embedding_enable: bool = False):
        embeddings = self.backbone(inputs.mean(dim=2))
        logits = self.head(torch.tanh(embeddings))
        return (logits, embeddings) if return_embedding_enable else logits


def _write_spectra(directory: Path, prefix: str, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # 波数轴必须落在统一插值网格上：推理侧会按坏段裁到 input_spec.point_count。
    wavenumbers = np.linspace(600.0, 1800.0, 8, dtype=np.float32)
    for index in range(2):
        intensities = np.linspace(value + index, value + index + 1.0, 8)
        np.savetxt(
            directory / f"{prefix}{index:02d}.arc_data",
            np.column_stack([wavenumbers, intensities]),
        )


def _build_config():
    """创建禁用随机增强的最小 CPU 训练配置。"""
    return build_config(
        {
            "profile_id": "GN",
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


def test_evaluate_test_set_writes_summary_for_trained_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "dataset" / "train"
    test_dir = tmp_path / "dataset" / "init_test"
    _write_spectra(train_dir / "GenusA" / "SpeciesA", "A", 1.0)
    _write_spectra(train_dir / "GenusB" / "SpeciesB", "B", 5.0)
    # 独立测试集是扁平结构：一层文件夹，属名只能从前缀反查。
    _write_spectra(test_dir / "SpeciesA20240101_01", "A", 1.0)
    _write_spectra(test_dir / "SpeciesB20240102_01", "B", 5.0)

    build_model = lambda num_classes, model_spec: SmokeModel(num_classes)
    monkeypatch.setattr("ramanv2.training.workflow.build_model", build_model)
    monkeypatch.setattr("ramanv2.inference.predictor.build_model", build_model)
    resolve_train_dir = lambda profile_key, project_root=None: train_dir
    monkeypatch.setattr("ramanv2.training.workflow.resolve_training_dir", resolve_train_dir)
    monkeypatch.setattr("ramanv2.evaluation.context.resolve_training_dir", resolve_train_dir)

    experiment_dir = tmp_path / "output"
    run_training(
        TrainRequest(
            _build_config(),
            "level_1",
            train_per_parent_enable=False,
            experiment_dir=experiment_dir,
        )
    )

    report = evaluate_test_set(experiment_dir, "cpu", test_dir)

    assert report["level_name"] == "level_1"

    assert report["sample_count"] == 4
    assert report["folder_count"] == 2
    assert Path(report["summary_path"]).is_file()
    records = report["records"]
    assert [record["mode"] for record in records] == ["model"]
    assert records[0]["level_name"] == "level_1"
    assert records[0]["support"] == 4
    assert 0.0 <= records[0]["spectrum_accuracy"] <= 1.0
    assert records[0]["spectrum_accuracy"] == records[0]["accuracy"]
    assert records[0]["folder_count"] == 2
    assert 0 <= records[0]["folder_correct"] <= 2
    assert 0.0 <= records[0]["folder_accuracy"] <= 1.0
    assert records[0]["covered_class_count"] == 2
    assert records[0]["class_count"] == 2
    assert Path(records[0]["result_dir"]).is_dir()
    assert (Path(records[0]["result_dir"]) / "test_eval_results.csv").is_file()

    summary_json = json.loads(
        (experiment_dir / "test_summary.json").read_text(encoding="utf-8")
    )
    assert summary_json["test_sample_count"] == 4
    assert summary_json["test_folder_count"] == 2
    assert summary_json["target_level"] == "level_1"
    assert summary_json["ignored_sample_counts"] == {"level_1": 0}
    assert summary_json["ignored_folders"] == {"level_1": []}
    summary_text = (experiment_dir / "test_summary.txt").read_text(encoding="utf-8")
    assert "独立测试集评估汇总" in summary_text
    assert "整体结果（部署口径）" in summary_text
    assert "Spectrum Accuracy" in summary_text
    assert "Folder Accuracy" in summary_text
    assert "level_1" in summary_text

