from pathlib import Path

import numpy as np
import pytest
import torch

from ramanv2.data.arc_data import write_arc_data
from ramanv2.experiments.run_manifest import build_hierarchy_entry, build_hierarchy_meta, write_hierarchy_meta
from ramanv2.core.config import build_config
from ramanv2.core.input_spec import build_input_spec
from ramanv2.experiments.run_context import open_experiment_context
from ramanv2.data.hierarchy_spec import HierarchyLevel, HierarchySpec
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.runtime.input import InputPreprocessor
from ramanv2.experiments.data_context import load_experiment_data_context
from ramanv2.experiments.run_manifest import resolve_slot_target
from ramanv2.inference.folder_labels import build_folder_summary
from ramanv2.modeling.factory import build_model
from ramanv2.modeling.model_spec import build_model_spec
from ramanv2.inference.predictor import load_predictor
from ramanv2.experiments.split import save_split_files
from ramanv2.training.split import build_split_indices


def build_experiment_fixture(tmp_path, hierarchical):
    train_dir = tmp_path / "\u5149\u8c31"
    experiment_dir = tmp_path / "experiment"
    hierarchy = HierarchySpec.flat()
    groups = ("A", "B", "C")
    if hierarchical:
        hierarchy = HierarchySpec(
            (HierarchyLevel("level_1", "group", True), HierarchyLevel("level_2", "prefix")),
            {},
            explicit=True,
        )
        groups = ("A/AA", "A/AB", "B/BA")
    axis = np.linspace(400, 1800, 32)
    intensity = np.arange(32, dtype=np.float32)
    for group in groups:
        for number in range(1, 5):
            write_arc_data(train_dir / group / f"cell{number}.arc_data", axis, intensity + number)

    index = DatasetIndex(train_dir, hierarchy=hierarchy)
    config = build_config(
        dict(label_schema=hierarchy.label_schema, target_points=32, bad_bands=[],
             smooth_use=False, backbone_type="direct", encoder_type="none", transformer_dim=8)
    )
    spec = build_model_spec(config.model, build_input_spec(config.input))
    open_experiment_context(experiment_dir, config)
    train, validation = build_split_indices(index, 0.5, 42)
    save_split_files(index, train, validation, experiment_dir)

    def write_slot(relative_dir):
        run_dir = experiment_dir / relative_dir
        run_dir.mkdir(parents=True)
        model = build_model(2 if hierarchical else 3, spec)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias[-1] = 1
        torch.save(model.state_dict(), run_dir / "model.pt")
        return build_hierarchy_entry(relative_dir, f"{relative_dir}/model.pt",
                                     child_ids=[0, 1] if "/" in relative_dir else None)

    first_entry = write_slot("level_1")
    parent_entries = {"level_2": {"0": write_slot("level_2/level_2_0")}} if hierarchical else {}
    meta = build_hierarchy_meta(
        **index.hierarchy_metadata,
        level_models={"level_1": first_entry},
        parent_models=parent_entries,
        label_schema=hierarchy.label_schema,
        hierarchy=index.hierarchy.to_payload(),
    )
    write_hierarchy_meta(experiment_dir / "hierarchy_meta.json", meta)
    return experiment_dir, train_dir, intensity, train, validation


@pytest.mark.parametrize("hierarchical", [False, True])
def test_saved_experiment_and_predictor(tmp_path, hierarchical):
    experiment, train_dir, intensity, train, validation = build_experiment_fixture(tmp_path, hierarchical)
    context = load_experiment_data_context(experiment, train_dir)
    assert context.dataset_index.get_class_names("level_1") == (["A", "B"] if hierarchical else ["A", "B", "C"])
    np.testing.assert_array_equal(context.train_indices, train)
    np.testing.assert_array_equal(context.validation_indices, validation)
    inputs = InputPreprocessor(context.input_spec, torch.device("cpu")).preprocess_intensity(intensity)
    assert inputs.shape == (1, 1, 32)
    assert inputs.mean().item() == pytest.approx(0, abs=1e-6)
    level = "level_2" if hierarchical else "level_1"
    predictor = load_predictor(experiment, "cpu", level)
    predictions = predictor.predict_tensor(inputs, top_k=1)
    assert predictions[0].class_id == 2
    assert predictions[0].probability == (1.0 if hierarchical else pytest.approx(0.5761169))
    if hierarchical:
        run_dir = experiment / "level_2" / "level_2_0"
        assert resolve_slot_target(run_dir) == ("level_2", 0)
        parent_predictor = load_predictor(run_dir, "cpu", "level_2")
        assert parent_predictor.predict_batch_class_ids(inputs) == [1]
        with pytest.raises(FileNotFoundError):
            resolve_slot_target(experiment / "level_2" / "level_2_9")


def test_experiment_readers_reject_retired_schema(tmp_path):
    from ramanv2.common.file_ops import write_json
    from ramanv2.experiments.run_manifest import load_hierarchy_meta
    from ramanv2.experiments.snapshot import load_run_snapshot, write_yaml_dict

    retired = {"label_schema": "genus_species_prefix_v1"}
    write_yaml_dict(tmp_path / "shared_config.yaml", retired)
    manifest_path = tmp_path / "hierarchy_meta.json"
    write_json(manifest_path, retired)

    with pytest.raises(ValueError, match="genus_species_prefix_v1"):
        load_run_snapshot(tmp_path)
    with pytest.raises(ValueError, match="genus_species_prefix_v1"):
        load_hierarchy_meta(manifest_path)


def test_test_folder_tie_keeps_first_label():
    summary = build_folder_summary(
        "sample", "B", ["A", "B"], [{"top1_label": "B"}, {"top1_label": "A"}]
    )
    assert summary["predicted_label"] == "B"
    assert summary["correct_ratio"] == 0.5
    assert summary["folder_correct"]

def test_independent_metrics_only_count_covered_classes(tmp_path):
    experiment, train_dir, intensity, _, _ = build_experiment_fixture(tmp_path, False)
    test_dir = tmp_path / "test"
    for name in ("C20240101_01", "Z20240101_01"):
        write_arc_data(test_dir / name / f"{name}_cell1.arc_data", np.linspace(400, 1800, 32), intensity)
    from ramanv2.evaluation.independent_test import evaluate_test_set
    from ramanv2.evaluation.model_eval import evaluate_model_run

    run_dir = experiment / "level_1"
    result = evaluate_test_set(run_dir, "cpu", test_dir, train_dir=train_dir)
    assert result["sample_count"] == 1
    assert result["folder_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["covered_class_count"] == 1
    assert result["class_count"] == 3
    validation = evaluate_model_run(run_dir, "level_1", "cpu", train_dir)
    assert (validation / "val_eval_results.csv").is_file()


def test_level_summary_writes_parent_known_and_cascade_reports(tmp_path, monkeypatch):
    from ramanv2.common.file_ops import read_json
    from ramanv2.data import profiles
    from ramanv2.data.datasets import DatasetRegistry, DatasetSpec
    from ramanv2.evaluation.level_summary import summarize_level_results
    from ramanv2.experiments.run_manifest import load_hierarchy_meta

    experiment, train_dir, intensity, _, _ = build_experiment_fixture(tmp_path, True)
    dataset_dir = tmp_path / "dataset_root"
    dataset_dir.mkdir()
    train_dir.rename(dataset_dir / "train")
    meta = load_hierarchy_meta(experiment / "hierarchy_meta.json")
    hierarchy = HierarchySpec.from_payload(meta["hierarchy"])
    spec = DatasetSpec("PB", "fixture", tmp_path, dataset_dir, (), hierarchy)
    registry = DatasetRegistry((spec,), spec.dataset_id)
    monkeypatch.setattr(profiles, "_load_registry", lambda root: registry)
    for prefix in ("AA", "AB", "BA"):
        name = f"{prefix}20240101_01"
        write_arc_data(
            spec.dataset_dir / "test" / name / f"{name}_cell1.arc_data", np.linspace(400, 1800, 32), intensity
        )

    from ramanv2.evaluation.independent_test import evaluate_test_set
    from ramanv2.evaluation.model_eval import evaluate_model_run

    evaluate_model_run(experiment / "level_1", "level_1", "cpu")
    evaluate_test_set(experiment / "level_1", "cpu")
    report = summarize_level_results(experiment, "level_2", "cpu")
    summary_path = experiment / "level_2" / "level_summary"
    assert Path(report["level_summary_dir"]) == summary_path
    for mode, expected_accuracy in (("parent_known", 2 / 3), ("cascade", 1 / 3)):
        test_result = summary_path / mode / "test_result"
        metrics = read_json(test_result / "metrics.json")
        assert metrics["summary"]["support"] == 3
        assert metrics["summary"]["accuracy"] == pytest.approx(expected_accuracy)
        assert metrics["folder"]["folder_count"] == 3
        assert (test_result / "summary.txt").is_file()
    inference_dir = summary_path / "cascade" / "test_result" / "inference_result"
    assert (inference_dir / "summary.json").is_file()
    assert not (inference_dir / "summary.txt").exists()


def test_training_checkpoint_round_trip(tmp_path):
    from ramanv2.training.checkpoint import restore_training_checkpoint, write_training_checkpoint

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    checkpoint = tmp_path / "checkpoint.pt"
    write_training_checkpoint(checkpoint, 3, model, optimizer, scheduler, 0.75, 2, 1, torch.ones(2))
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1)
    result = restore_training_checkpoint(
        checkpoint, restored, restored_optimizer, restored_scheduler, torch.device("cpu"), lambda _: None
    )
    assert result[:4] == (4, 0.75, 2, 1)
    torch.testing.assert_close(result[4], torch.ones(2))
    torch.testing.assert_close(restored.weight, model.weight)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
