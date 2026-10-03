from __future__ import annotations

from pathlib import Path

import numpy as np

from ramanv2.common.arc_data import read_arc_data, read_raw_arc_data, write_arc_data
from ramanv2.common.naming import build_natural_key
from ramanv2.core.paths import find_ancestor_dir
from ramanv2.core.runtime import resolve_device
from ramanv2.data.count import summarize_dataset
from ramanv2.data.profiles import resolve_training_dir
from ramanv2.evaluation.scope import resolve_run_class_ids, select_parent_indices


def test_arc_data_io_is_canonical_and_data_io_has_no_file_writer() -> None:
    import ramanv2.data.io as data_io

    assert not hasattr(data_io, "read_arc_data")
    assert not hasattr(data_io, "write_arc_data")
    assert read_arc_data is not None


def test_arc_data_readers_agree_on_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "spectrum.arc_data"
    path.write_text("600 1\ninvalid\n601 2 3\n601 2\n", encoding="utf-8")

    parsed_axis, parsed_intensity = read_arc_data(path)
    raw_axis, raw_intensity, malformed = read_raw_arc_data(path)

    assert np.array_equal(parsed_axis, raw_axis)
    assert np.array_equal(parsed_intensity, raw_intensity)
    assert malformed == 2


def test_shared_sorting_and_dataset_summary(tmp_path: Path) -> None:
    root = tmp_path / "init"
    write_arc_data(root / "Genus" / "cell10" / "b.arc_data", [1, 2], [3, 4])
    write_arc_data(root / "Genus" / "cell2" / "a.arc_data", [1, 2], [3, 4])

    assert sorted(["cell10", "cell2"], key=build_natural_key) == ["cell2", "cell10"]
    assert summarize_dataset(root) == {
        "exists": True,
        "genus_count": 1,
        "folder_count": 2,
        "file_count": 2,
    }


def test_training_dir_resolution_and_ancestor_lookup(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    train_dir = project_root / "dataset" / "GN" / "train"
    train_dir.mkdir(parents=True)
    (project_root / "shared_config.yaml").write_text("{}\n", encoding="utf-8")

    assert resolve_training_dir("GN", project_root) == train_dir.resolve()
    assert find_ancestor_dir(train_dir, "shared_config.yaml") == project_root.resolve()


def test_training_dir_falls_back_to_init_or_returns_missing_train(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    init_dir = project_root / "dataset" / "GN" / "init"
    init_dir.mkdir(parents=True)

    assert resolve_training_dir("GN", project_root) == init_dir.resolve()
    assert resolve_training_dir(
        "GN",
        project_root,
        fallback_to_init_enable=False,
    ) == (project_root / "dataset" / "GN" / "train").resolve()


def test_runtime_and_evaluation_scope_helpers() -> None:
    assert str(resolve_device("cpu")) == "cpu"

    labels = np.asarray([[0, 0], [0, 1], [1, 1], [1, -1]])
    selected = select_parent_indices(labels, np.arange(4), 0, 1, 0)
    assert selected.tolist() == [0, 1]

    dataset_index = type("DatasetIndexStub", (), {"num_classes_by_level": {"leaf": 3}})()
    global_entry = type("RunEntryStub", (), {"parent_id": None, "level_name": "leaf", "values": {}})()
    parent_entry = type(
        "RunEntryStub",
        (),
        {"parent_id": 1, "level_name": "leaf", "values": {"child_ids": [2, 4]}},
    )()
    assert resolve_run_class_ids(dataset_index, global_entry) == [0, 1, 2]
    assert resolve_run_class_ids(dataset_index, parent_entry) == [2, 4]


def test_analysis_context_wrapper_is_removed() -> None:
    assert not Path("ramanv2/analysis/context.py").exists()


def test_dataset_mapping_subset_and_old_data_modules_are_removed() -> None:
    from ramanv2.data.catalog import DatasetMapping

    mapping = DatasetMapping.from_payload(
        {
            "genera": {"Enterobacter": ["ECL"], "Klebsiella": ["KP"]},
            "datasets": {"GN": ["Enterobacter"], "GP": ["Klebsiella"]},
        }
    )
    subset = mapping.subset("GN")
    assert subset.datasets == {"GN": ("Enterobacter",)}
    assert subset.genera == mapping.genera
    assert not Path("ramanv2/data/profile_catalog.py").exists()
    assert not Path("ramanv2/data/init_builder.py").exists()
    assert not Path("ramanv2/data/colab_export.py").exists()
