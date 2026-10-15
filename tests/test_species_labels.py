"""种标签跨采集合并，来源目录及已有切分保持独立。"""

from pathlib import Path

import numpy as np
import pytest

from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.inference.labels import build_expected_label_lookup
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


def test_species_merge_dates_case_and_transferred_samples_without_merging_sources(tmp_path):
    for relative in (
        "Acinetobacter/AB20231212_01",
        "Acinetobacter/ab20231214_01",
        "Acinetobacter/ABCS01",
        "Acinetobacter/AJO20231214_01",
        "OtherGenus/AB20231214_01",
    ):
        _write_group(tmp_path, relative)
    index = DatasetIndex(tmp_path, load_intensity_enable=False)
    assert index.num_classes_by_level == {"level_1": 2, "level_2": 3, "leaf": 5}
    assert set(index.get_class_names("level_2")) == {
        "Acinetobacter/AB", "Acinetobacter/AJO", "OtherGenus/AB",
    }
    ab_indices = [
        i for i in range(len(index))
        if index.get_level_key(i, "level_2") == "Acinetobacter/AB"
    ]
    assert len(ab_indices) == 9
    assert len({index.get_leaf_key(i) for i in ab_indices}) == 3
    assert len({index.level_labels[i, 1] for i in ab_indices}) == 1
    for i in range(len(index)):
        assert index.get_leaf_key(i) == Path(index.samples[i]).parent.relative_to(tmp_path).as_posix()

    indices = np.arange(len(index))
    task = build_global_train_task(index, "level_2", indices, indices)
    assert task.num_classes == 3
    genus_id = index.label_maps_by_level[0]["Acinetobacter"]
    parent_task = build_parent_train_task(index, "level_2", genus_id, indices, indices)
    assert parent_task.num_classes == 2
    assert sorted(np.bincount(parent_task.weight_labels).tolist()) == [3, 9]
    other_id = index.label_maps_by_level[0]["OtherGenus"]
    assert build_parent_train_task(index, "level_2", other_id, indices, indices) is None


def test_species_labels_preserve_deeper_levels_and_source_split(tmp_path):
    for relative in ("Genus/AB20231212_01/Area1", "Genus/ab20231214_01/Area1"):
        _write_group(tmp_path, relative, 5)
    index = DatasetIndex(tmp_path, load_intensity_enable=False)
    assert index.get_class_names("level_2") == ["Genus/AB"]
    assert index.get_class_names("level_3") == ["Genus/AB/Area1"]
    assert index.num_classes_by_level["leaf"] == 2

    # 直接用完整来源路径作为旧 leaf 分组键，核对切分文件归属不变。
    class SourceIndex:
        def __len__(self):
            return len(index)

        def get_leaf_key(self, number):
            return Path(index.samples[number]).parent.relative_to(tmp_path).as_posix()

    actual = split_by_lowest_level_ratio(index, seed=42)
    expected = split_by_lowest_level_ratio(SourceIndex(), seed=42)
    assert actual == expected


@pytest.mark.parametrize("level, expected", [
    ("level_1", {"ACINETOBACTER": "Acinetobacter", "AB": "Acinetobacter", "AJO": "Acinetobacter"}),
    ("level_2", {"AB": "Acinetobacter/AB", "AJO": "Acinetobacter/AJO"}),
])
def test_cs_expected_labels_ignore_raw_leaf_names(level, expected):
    meta = {"class_names_by_level": {
        "level_1": ["Acinetobacter"],
        "level_2": ["Acinetobacter/AB", "Acinetobacter/AJO"],
        "leaf": ["Acinetobacter/AB20231212_01", "Acinetobacter/AB20231214_01"],
    }}
    assert build_expected_label_lookup(meta, level) == expected
