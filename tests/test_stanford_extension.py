from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramanv2.core.config import InputConfig
from ramanv2.data.build import build_train
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.io import read_arc_data, write_arc_data
from ramanv2.data.profiles import DatasetProfile
from ramanv2.extensions.stanford_finetune.dataset import build_stanford_train
from ramanv2.extensions.stanford_finetune.initializer import TransferInitializer


class TransferModel(torch.nn.Module):
    """用于迁移初始化测试的极小模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(3, 2)


def test_stanford_train_preserves_leaf_categories_and_native_values(tmp_path: Path) -> None:
    reference_axis = np.arange(600.0, 620.0)
    input_config = InputConfig(
        cut_min=600.0,
        cut_max=619.0,
        target_points=20,
        bad_bands=(),
    )
    source_values = np.arange(20, dtype=np.float32)
    write_arc_data(
        tmp_path / "init" / "CA01" / "sample.arc_data",
        reference_axis,
        source_values,
        fmt="%.10g",
    )

    train_dir = build_stanford_train(
        tmp_path,
        input_config=input_config,
        reference_wavenumbers=reference_axis,
    )

    output_axis, output_values = read_arc_data(train_dir / "CA01" / "sample.arc_data")
    assert (train_dir / "CA01").is_dir()
    assert np.allclose(output_axis, reference_axis)
    assert np.allclose(output_values, source_values)


def test_transfer_initializer_resets_head_and_freezes_unselected_modules(tmp_path: Path) -> None:
    source = TransferModel()
    with torch.no_grad():
        source.layer1.weight.fill_(2.0)
        source.head.weight.fill_(7.0)
    source_path = tmp_path / "source.pt"
    torch.save(source.state_dict(), source_path)
    target = TransferModel()
    initializer = TransferInitializer(source_path, ())
    task = SimpleNamespace(model_tag="level_1", num_classes=2)

    initializer.initialize(target, task)
    target.train()
    initializer.apply_training_mode(target)

    assert torch.equal(target.layer1.weight, source.layer1.weight)
    assert not torch.equal(target.head.weight, source.head.weight)
    assert target.head.weight.requires_grad
    assert not target.layer1.weight.requires_grad
    assert target.head.training
    assert not target.layer1.training
    assert initializer.reports["level_1"]["head_reset"] is True


def test_regular_builder_accepts_an_external_reference_axis(tmp_path: Path) -> None:
    reference_axis = np.arange(600.0, 620.0)
    input_config = InputConfig(
        cut_min=600.0,
        cut_max=619.0,
        target_points=20,
        bad_bands=(),
    )
    write_arc_data(
        tmp_path / "init" / "Genus" / "AB01" / "source.arc_data",
        reference_axis,
        np.linspace(0.0, 1.0, 20),
    )
    profile = DatasetProfile("temp", "temp")

    build_train(
        profile,
        tmp_path,
        config=DataBuildConfig(min_samples_per_class=1),
        input_config=input_config,
        reference_wavenumbers=reference_axis,
    )

    output_axis, _output_values = read_arc_data(
        tmp_path / "train" / "Genus" / "AB" / "AB01_source.arc_data"
    )
    assert np.allclose(output_axis, reference_axis)
