from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman.training.losses import AlignLoss
from raman.training.losses import FocalLoss as ReferenceFocalLoss
from raman.training.losses import SupConLoss as ReferenceSupConLoss
from raman.training.losses import build_class_weights as reference_build_class_weights
from raman.training.losses import get_linear_weight as reference_get_linear_weight
from raman.training.split import apply_train_filter as reference_apply_train_filter
from raman.training.split import resolve_train_scope as reference_resolve_train_scope
from raman.training.split import split_by_lowest_level_ratio as reference_split_by_lowest_level_ratio
from raman.data.input import augment_norm_spectrum as reference_augment_normalized_spectrum
from raman.data.input import augment_raw_spectrum as reference_augment_raw_spectrum
from raman.data.input import build_model_input as reference_build_model_input
from raman.data.input import build_sg_kernels as reference_build_sg_kernels
from raman.config import make_config
from raman_temp.core.config import build_config
from raman_temp.core.input_spec import build_input_spec
from raman_temp.data.augmentation import build_augmentation_spec
from raman_temp.data.augmentation import augment_normalized_spectrum, augment_raw_spectrum
from raman_temp.data.input import build_model_input, build_sg_kernels
from raman_temp.training.optimizer import build_loader, build_optimizer, build_scheduler
from raman_temp.training.spec import build_execution_spec, build_training_spec
from raman_temp.training.checkpoint import restore_training_checkpoint, save_training_checkpoint
from raman_temp.training.losses import FocalLoss, SupConLoss, build_class_weights, compute_align_loss, compute_linear_weight
from raman_temp.training.split import (
    apply_train_filter,
    build_global_train_task,
    build_parent_train_task,
    build_train_scope,
    split_by_lowest_level_ratio,
)


class DemoDataset:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.samples = [
            root_dir / "G1" / "A01_a.csv",
            root_dir / "G1" / "A01_b.csv",
            root_dir / "G1" / "B02_a.csv",
            root_dir / "G1" / "B02_b.csv",
            root_dir / "G2" / "C03_a.csv",
            root_dir / "G2" / "C03t_a.csv",
        ]
        self.level_labels = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 1],
                [1, 2],
                [1, 2],
            ]
        )
        self.label_maps_by_level = [{"G1": 0, "G2": 1}, {"A": 0, "B": 1, "C": 2}]
        self.num_classes_by_level = {"genus": 2, "leaf": 3}
        self.head_names = ["genus", "leaf"]
        self.head_name_to_idx = {"genus": 0, "leaf": 1}
        self.parent_to_children = {"genus": {}, "leaf": {0: [0, 1], 1: [2]}}

    def __len__(self) -> int:
        return len(self.samples)

    def get_split_key(self, index: int, level_name: str) -> str:
        return self.get_level_key(index, level_name)

    def get_leaf_key(self, index: int) -> int:
        return int(self.level_labels[index, 1])

    def get_level_key(self, index: int, level_name: str) -> int:
        return int(self.level_labels[index, {"genus": 0, "leaf": 1}[level_name]])

    def get_parent_level(self, level_name: str) -> str | None:
        return "genus" if level_name == "leaf" else None

    def resolve_level_name(self, level_name: str) -> str:
        return level_name

    def _resolve_level_name(self, level_name: str) -> str:
        return self.resolve_level_name(level_name)


@dataclass
class DemoConfig:
    train_only_parent: int | None = None
    train_only_parent_name: str | None = "G1"
    train_filter_level: str | None = None
    train_filter_value: str | None = None


class DummyScaler:
    def __init__(self, state: dict[str, float]) -> None:
        self.state = state

    def state_dict(self) -> dict[str, float]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.state = dict(state)


class OptimizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem_branches = torch.nn.Linear(2, 2)
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)


def test_losses_match_reference_values_and_gradients() -> None:
    labels = np.array([0, 0, 1, -1, 2])
    np.testing.assert_allclose(build_class_weights(labels, 3), reference_build_class_weights(labels, 3))
    assert compute_linear_weight(3, 2, 6, 0.0, 0.4) == reference_get_linear_weight(3, 2, 6, 0.0, 0.4)

    logits = torch.tensor([[1.0, 0.2], [0.3, 1.4], [1.3, 0.6]], requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([0, 1, -1])
    weights = torch.tensor([0.7, 1.3])
    target_loss = FocalLoss(2.0, weights, -1)(logits, targets).mean()
    reference_loss = ReferenceFocalLoss(2.0, weights, -1)(reference_logits, targets).mean()
    target_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(target_loss, reference_loss)
    torch.testing.assert_close(logits.grad, reference_logits.grad)

    features = torch.tensor([[1.0, 0.2], [0.6, 0.8], [2.0, 1.0], [2.5, 0.4]], requires_grad=True)
    reference_features = features.detach().clone().requires_grad_(True)
    feature_labels = torch.tensor([0, 0, 1, 1])
    target_align = compute_align_loss(features, feature_labels)
    reference_align = AlignLoss(reference_features, feature_labels)
    target_align.backward()
    reference_align.backward()
    torch.testing.assert_close(target_align, reference_align)
    torch.testing.assert_close(features.grad, reference_features.grad)

    target_supcon = SupConLoss(0.2)(features.detach(), feature_labels)
    reference_supcon = ReferenceSupConLoss(0.2)(features.detach(), feature_labels)
    torch.testing.assert_close(target_supcon, reference_supcon)


def test_train_scope_does_not_mutate_config_and_filters_like_reference(tmp_path: Path) -> None:
    dataset = DemoDataset(tmp_path / "dataset")
    head_name_to_index = {"genus": 0, "leaf": 1}
    config = DemoConfig()
    target_scope = build_train_scope(
        dataset,
        "leaf",
        head_name_to_index,
        only_parent_name=config.train_only_parent_name,
    )
    assert config.train_only_parent is None
    assert config.train_filter_level is None
    assert config.train_filter_value is None
    assert target_scope.only_parent == 0
    assert target_scope.filter_level == "genus"
    assert target_scope.filter_values == ("G1",)

    reference_config = DemoConfig()
    reference_parent = reference_resolve_train_scope(dataset, reference_config, "leaf", head_name_to_index)
    train_indices = np.array([0, 1, 2, 3, 4, 5])
    val_indices = np.array([1, 3, 5])
    expected_train, expected_val = reference_apply_train_filter(
        dataset,
        train_indices,
        val_indices,
        reference_config,
        head_name_to_index,
    )
    actual_train, actual_val = apply_train_filter(
        dataset,
        train_indices,
        val_indices,
        target_scope,
        head_name_to_index,
    )
    assert reference_parent == target_scope.only_parent
    np.testing.assert_array_equal(actual_train, expected_train)
    np.testing.assert_array_equal(actual_val, expected_val)


def test_split_methods_match_reference(tmp_path: Path) -> None:
    dataset = DemoDataset(tmp_path / "dataset")
    for source_prefix_enable in (False, True):
        expected = reference_split_by_lowest_level_ratio(
            dataset,
            seed=17,
            split_by_source_prefix=source_prefix_enable,
        )
        actual = split_by_lowest_level_ratio(
            dataset,
            seed=17,
            split_by_source_prefix_enable=source_prefix_enable,
        )
        assert actual == expected


def test_train_tasks_preserve_global_and_parent_label_semantics(tmp_path: Path) -> None:
    dataset = DemoDataset(tmp_path / "dataset")
    train_indices = np.array([0, 1, 2, 3, 4])
    val_indices = np.array([5])
    global_task = build_global_train_task(dataset, "leaf", train_indices, val_indices)
    assert global_task.model_tag == "leaf"
    assert global_task.visible_class_ids == (0, 1, 2)
    assert global_task.label_map is None
    np.testing.assert_array_equal(global_task.weight_labels, [0, 0, 1, 1, 2])
    assert global_task.train_indices.flags.writeable is False

    parent_task = build_parent_train_task(dataset, "leaf", 0, train_indices, val_indices)
    assert parent_task is not None
    assert parent_task.model_tag == "leaf_0"
    assert parent_task.parent_id == 0
    assert parent_task.visible_class_ids == (0, 1)
    np.testing.assert_array_equal(parent_task.train_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(parent_task.val_indices, [])
    np.testing.assert_array_equal(parent_task.label_map, [0, 1, -1])
    np.testing.assert_array_equal(parent_task.weight_labels, [0, 0, 1, 1])
    assert parent_task.label_map.flags.writeable is False
    assert build_parent_train_task(dataset, "leaf", 1, train_indices, val_indices) is None


def test_checkpoint_restores_scaler_and_training_state(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = DummyScaler({"scale": 128.0})
    checkpoint_path = tmp_path / "resume_checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_score=0.75,
        best_epoch=2,
        patience_counter=1,
        ema_class_ce=torch.tensor([0.5, 1.5]),
        scaler=scaler,
    )

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=4)
    restored_scaler = DummyScaler({"scale": 1.0})
    messages: list[str] = []
    state = restore_training_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
        messages.append,
        restored_scaler,
    )
    assert state[:4] == (4, 0.75, 2, 1)
    torch.testing.assert_close(state[4], torch.tensor([0.5, 1.5]))
    assert restored_scaler.state == {"scale": 128.0}
    assert messages and "last_epoch=3" in messages[0]
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_train_and_runtime_specs_extract_config_without_retaining_it() -> None:
    config = build_config(
        {"epochs": 12, "scheduler_t_max": None, "use_align_loss": True, "use_amp": True}
    )
    train_spec = build_training_spec(config.training, config.execution)
    runtime_spec = build_execution_spec(config.execution)
    assert train_spec.epochs == 12
    assert train_spec.optimizer.scheduler_t_max == 12
    assert train_spec.loss.align_enable is True
    assert train_spec.loss.focal_gamma == config.training.gamma
    assert runtime_spec.amp_enable is True
    input_spec = build_input_spec(config.input)
    assert input_spec.in_channels == 1 + int(config.input.smooth_use) + int(config.input.d1_use)
    assert input_spec.point_count == 851
    assert input_spec.d1_enable is config.input.d1_use
    with pytest.raises(FrozenInstanceError):
        train_spec.epochs = 3


def test_loader_optimizer_and_scheduler_follow_train_spec() -> None:
    config = build_config(
        {"batch_size": 2, "learning_rate": 0.01, "train_loader_num_workers": 0, "val_loader_num_workers": 0}
    )
    train_spec = build_training_spec(config.training, config.execution)
    dataset = torch.utils.data.TensorDataset(torch.arange(6).view(3, 2).float())
    loader = build_loader(dataset, train_spec.train_loader, torch.device("cpu"))
    assert loader.batch_size == 2
    assert len(loader) == 2

    optimizer = build_optimizer(OptimizerModel(), train_spec.optimizer)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.006, 0.01, 0.011]
    )
    scheduler = build_scheduler(optimizer, train_spec.optimizer)
    assert scheduler.T_max == train_spec.optimizer.scheduler_t_max


def test_input_and_augmentation_match_reference_with_fixed_seed() -> None:
    reference_config = make_config()
    reference_config.d1_use = True
    config = build_config(reference_config.to_dict())
    values = np.linspace(1.0, 5.0, config.input.target_points, dtype=np.float32)
    input_spec = build_input_spec(config.input)
    augmentation_spec = build_augmentation_spec(config.training)
    reference_smooth_kernel, reference_d1_kernel = reference_build_sg_kernels(reference_config, "cpu")
    smooth_kernel, d1_kernel = build_sg_kernels(input_spec, "cpu")
    expected_input = reference_build_model_input(
        values,
        reference_config,
        reference_smooth_kernel,
        reference_d1_kernel,
        "cpu",
        augment=False,
    )
    actual_input = build_model_input(
        values,
        input_spec,
        smooth_kernel,
        d1_kernel,
        "cpu",
    )
    torch.testing.assert_close(actual_input, expected_input)

    np.random.seed(7)
    expected_raw = reference_augment_raw_spectrum(values, reference_config)
    np.random.seed(7)
    actual_raw = augment_raw_spectrum(values, augmentation_spec)
    np.testing.assert_allclose(actual_raw, expected_raw)

    normalized = np.linspace(-1.0, 1.0, config.input.target_points, dtype=np.float32)
    np.random.seed(17)
    expected_normalized = reference_augment_normalized_spectrum(normalized, reference_config)
    np.random.seed(17)
    actual_normalized = augment_normalized_spectrum(normalized, augmentation_spec)
    np.testing.assert_allclose(actual_normalized, expected_normalized)
