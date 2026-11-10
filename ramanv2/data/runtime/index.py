"""训练数据目录的结构化索引。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ramanv2.common.hierarchy import parts_to_key
from ramanv2.common.naming import parse_folder_prefix
from ramanv2.data.dataset_io import iter_arc_dirs, load_arc_intensity
from ramanv2.data.hierarchy_config import HierarchySpec


class DatasetIndex:
    """一次扫描训练目录，保存层级标签和只读原始强度缓存。"""

    def __init__(
        self,
        train_dir: Path | str,
        hierarchy: HierarchySpec,
        load_intensity_enable: bool = True,
    ) -> None:
        self.root_dir = Path(train_dir).resolve()
        self.hierarchy = hierarchy
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"训练目录不存在：{self.root_dir}")

        self.samples: np.ndarray
        self.level_labels: np.ndarray
        self.hier_names: np.ndarray
        self.head_names: list[str] = []
        self.head_name_to_idx: dict[str, int] = {}
        self.label_maps_by_level: list[dict[str, int]] = []
        self.parent_to_children: dict[str, dict[int, list[int]]] = {}
        self._scan_samples()
        # 页面预检只需要目录层级和样本数量，避免无谓读取全部光谱数值。
        self._raw_intensities = (
            self._load_all_raw_intensities() if load_intensity_enable else {}
        )

    @property
    def level_names(self) -> list[str]:
        """返回业务层级名称。"""
        return list(self.head_names)

    def __len__(self) -> int:
        """返回索引内的光谱样本数。"""
        return len(self.samples)

    @property
    def class_names_by_level(self) -> list[list[str]]:
        """返回每个层级按标签索引排序的类别名称。"""
        return [list(label_map) for label_map in self.label_maps_by_level]

    @property
    def num_classes_by_level(self) -> dict[str, int]:
        """返回各层级类别数量。"""
        return {
            level_name: len(self.label_maps_by_level[level_index])
            for level_index, level_name in enumerate(self.head_names)
        }

    @property
    def parent_level_name(self) -> dict[str, str | None]:
        """返回每个层级标签的直接父层级。"""
        return {
            level_name: None if index == 0 else self.head_names[index - 1]
            for index, level_name in enumerate(self.head_names)
        }

    @property
    def hierarchy_metadata(self) -> dict:
        """导出训练与评估共同校验的类别顺序和父子关系。"""
        return {
            "head_names": list(self.head_names),
            "class_names_by_level": dict(zip(self.head_names, self.class_names_by_level)),
            "parent_to_children": {
                name: {str(parent): children for parent, children in mapping.items()}
                for name, mapping in self.parent_to_children.items()
            },
        }

    def get_raw_intensity(self, index: int) -> np.ndarray:
        """返回指定样本的只读原始强度数组。"""
        return self._raw_intensities[Path(self.samples[index])]

    def get_class_names(self, level_name: str) -> list[str]:
        """返回指定业务层级的类别名称。"""
        level_index = self.head_name_to_idx[self.resolve_level_name(level_name)]
        return list(self.class_names_by_level[level_index])

    def get_hierarchy(self, index: int) -> dict[str, str | None]:
        """返回样本的业务层级名称，不构建输入张量。"""
        return dict(self.hier_names[index])

    def get_level_key(self, index: int, level_name: str) -> str | None:
        """返回样本在指定业务层级的稳定类别键。"""
        return self.hier_names[index].get(self.resolve_level_name(level_name))

    def get_parent_level(self, level_name: str) -> str | None:
        """返回指定业务层级的直接父层级。"""
        return self.parent_level_name.get(self.resolve_level_name(level_name))

    def resolve_level_name(self, level_name: str) -> str:
        """校验业务层级名称位于当前数据集结构内。"""
        if not isinstance(level_name, str) or not level_name.startswith("level_"):
            raise ValueError(f"level_name 必须是业务层级名称，当前为：{level_name}")
        if level_name not in self.level_names:
            raise ValueError(
                f"未知业务层级：{level_name}；可选值：{self.level_names}"
            )
        return level_name

    def _scan_samples(self) -> None:
        """按显式层级配置解析训练目录；目录层级之外的分类值由前缀映射得到。"""
        spec = self.hierarchy
        depth = spec.directory_depth
        records: list[tuple[Path, tuple[str, ...]]] = []
        for leaf_dir, filenames in iter_arc_dirs(self.root_dir):
            parts = leaf_dir.relative_to(self.root_dir).parts
            if len(parts) < max(depth, 1):
                raise ValueError(f"训练目录层级不足：{leaf_dir}")
            records.extend((leaf_dir / filename, parts) for filename in filenames)
        if not records:
            raise RuntimeError(f"训练目录中没有 .arc_data 光谱：{self.root_dir}")

        level_names = list(spec.level_names)
        self.head_names = list(level_names)
        self.head_name_to_idx = {
            name: index for index, name in enumerate(self.head_names)
        }
        learned_mappings = {
            level_name: dict(mapping)
            for level_name, mapping in spec.mappings.items()
        }
        for level in spec.levels:
            learned_mappings.setdefault(level.name, {})
        level_maps = [dict() for _ in level_names]
        label_records: list[tuple[Path, tuple[str, ...]]] = []
        resolved_records: list[tuple[Path, tuple[str, ...], tuple[str, ...]]] = []
        for path, parts in records:
            group_name = parts[-1]
            raw_prefix = parse_folder_prefix(group_name).upper()
            values_list: list[str] = []
            for level_index, level in enumerate(spec.levels):
                if level.directory and level_index < len(parts):
                    value = parts[level_index]
                    learned_mappings[level.name][raw_prefix] = value
                else:
                    value = learned_mappings[level.name].get(raw_prefix)
                    if value is None:
                        value = parse_folder_prefix(group_name)
                values_list.append(value)
            values = tuple(values_list)
            for level_index in range(len(values)):
                level_key = parts_to_key(values[: level_index + 1])
                level_maps[level_index].setdefault(level_key, len(level_maps[level_index]))
            resolved_records.append((path, parts, values))
            label_records.append((path, values))

        self.label_maps_by_level = level_maps
        self.parent_to_children = _build_parent_to_children(
            label_records,
            level_maps,
            level_names,
        )
        samples: list[Path] = []
        labels: list[list[int]] = []
        hierarchies: list[dict[str, str | None]] = []
        for path, parts, values in resolved_records:
            row = [-1] * len(self.head_names)
            hierarchy = {name: None for name in level_names}
            for level_index, value in enumerate(values):
                level_key = parts_to_key(values[: level_index + 1])
                row[level_index] = level_maps[level_index][level_key]
                hierarchy[level_names[level_index]] = level_key
            samples.append(path)
            labels.append(row)
            hierarchies.append(hierarchy)
        self.samples = np.asarray(samples, dtype=object)
        self.level_labels = np.asarray(labels, dtype=np.int64)
        self.hier_names = np.asarray(hierarchies, dtype=object)
        self.hierarchy = HierarchySpec(
            levels=spec.levels,
            mappings=learned_mappings,
            explicit=spec.explicit,
        )

    def _load_all_raw_intensities(self) -> dict[Path, np.ndarray]:
        """一次性读取全部样本强度，并冻结数组以避免增强原地修改。"""
        intensities: dict[Path, np.ndarray] = {}
        for sample in self.samples:
            path = Path(sample)
            values = load_arc_intensity(path)
            values.setflags(write=False)
            intensities[path] = values
        return intensities


def _build_parent_to_children(
    records: list[tuple[Path, tuple[str, ...]]],
    level_maps: list[dict[str, int]],
    level_names: list[str],
) -> dict[str, dict[int, list[int]]]:
    """从目录树构建每个业务层级的父类到子类索引映射。"""
    mappings: dict[str, dict[int, list[int]]] = {}
    for level_index, level_name in enumerate(level_names):
        if level_index == 0:
            mappings[level_name] = {}
            continue
        children_by_parent: defaultdict[int, set[int]] = defaultdict(set)
        for _, parts in records:
            if len(parts) < level_index + 1:
                continue
            parent_key = parts_to_key(parts[:level_index])
            child_key = parts_to_key(parts[: level_index + 1])
            parent_id = level_maps[level_index - 1][parent_key]
            child_id = level_maps[level_index][child_key]
            children_by_parent[parent_id].add(child_id)
        mappings[level_name] = {
            parent_id: sorted(child_ids)
            for parent_id, child_ids in children_by_parent.items()
        }
    return mappings
