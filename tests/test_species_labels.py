"""种标签跨采集合并，来源目录及已有切分保持独立。"""

from pathlib import Path

import numpy as np

from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.training.split import (
    build_global_train_task,
    build_parent_train_task,
    split_by_lowest_level_ratio,
)


def _write_group(root: Path, relative: str, count: int = 3) -> None:
    folder = root / relative
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (folder / f"cell{index}.arc_data").write_text("600 1\n1000 2\n", encoding="utf-8")


def test_species_merge_dates_and_case_without_merging_sources(tmp_path):
    for relative in (
        "Acinetobacter/AB20231212_01",
        "Acinetobacter/ab20231214_01",
        "Acinetobacter/AJO20231214_01",
        "OtherGenus/AB20231214_01",
    ):
        _write_group(tmp_path, relative)
    index = DatasetIndex(tmp_path, load_intensity_enable=False)
    assert index.num_classes_by_level == {"level_1": 2, "level_2": 3, "leaf": 4}
    assert set(index.get_class_names("level_2")) == {
        "Acinetobacter/AB", "Acinetobacter/AJO", "OtherGenus/AB",
    }
    ab_indices = [
        i for i in range(len(index))
        if index.get_level_key(i, "level_2") == "Acinetobacter/AB"
    ]
    assert len(ab_indices) == 6
    assert len({index.get_leaf_key(i) for i in ab_indices}) == 2
    assert len({index.level_labels[i, 1] for i in ab_indices}) == 1
    for i in range(len(index)):
        assert index.get_leaf_key(i) == Path(index.samples[i]).parent.relative_to(tmp_path).as_posix()

    indices = np.arange(len(index))
    task = build_global_train_task(index, "level_2", indices, indices)
    assert task.num_classes == 3
    genus_id = index.label_maps_by_level[0]["Acinetobacter"]
    parent_task = build_parent_train_task(index, "level_2", genus_id, indices, indices)
    assert parent_task.num_classes == 2
    assert sorted(np.bincount(parent_task.weight_labels).tolist()) == [3, 6]
    other_id = index.label_maps_by_level[0]["OtherGenus"]
    assert build_parent_train_task(index, "level_2", other_id, indices, indices) is None




