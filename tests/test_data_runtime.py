from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raman_temp.core.input_spec import InputSpec
from raman_temp.data.dataset import RamanDataset
from raman_temp.data.index import DatasetIndex
from raman_temp.data.input import InputPreprocessor


def test_dataset_scans_hierarchy_and_builds_input(tmp_path: Path) -> None:
    root_dir = tmp_path / "train"
    _write_spectrum(root_dir / "GenusA" / "SpeciesA" / "A01_a.arc_data", 1.0)
    _write_spectrum(root_dir / "GenusA" / "SpeciesA" / "A01_b.arc_data", 2.0)
    _write_spectrum(root_dir / "GenusA" / "SpeciesB" / "B01_a.arc_data", 3.0)
    _write_spectrum(root_dir / "GenusB" / "SpeciesC" / "C01_a.arc_data", 4.0)
    dataset_index = DatasetIndex(root_dir)
    dataset = RamanDataset(
        dataset_index,
        InputSpec(in_channels=1, point_count=8, norm_method="none"),
    )

    assert len(dataset) == 4
    assert dataset_index.head_names == ["level_1", "level_2", "leaf"]
    assert dataset_index.num_classes_by_level == {
        "level_1": 2,
        "level_2": 3,
        "leaf": 3,
    }
    assert dataset_index.parent_to_children["level_2"] == {0: [0, 1], 1: [2]}
    assert dataset_index.get_parent_level("level_2") == "level_1"
    assert dataset_index.get_split_key(0, "level_1/leaf") == (
        "GenusA",
        "GenusA/SpeciesA",
    )
    assert not dataset_index.get_raw_intensity(0).flags.writeable

    inputs, labels, path = dataset[0]
    assert inputs.shape == (1, 8)
    assert labels.tolist() == [0, 0, 0]
    assert path.endswith("A01_a.arc_data")

    preprocessor = InputPreprocessor(
        InputSpec(in_channels=1, point_count=8, norm_method="none"),
        "cpu",
    )
    preprocessed = preprocessor.preprocess_intensity(
        dataset_index.get_raw_intensity(0)
    )
    assert preprocessed.shape == (1, 1, 8)
    assert torch.equal(preprocessed[0], inputs)


def _write_spectrum(path: Path, value: float) -> None:
    """写入一条固定长度的双列测试光谱。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    wavenumbers = np.arange(8, dtype=np.float32)
    intensities = np.linspace(value, value + 1.0, 8, dtype=np.float32)
    np.savetxt(path, np.column_stack([wavenumbers, intensities]))
