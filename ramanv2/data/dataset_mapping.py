"""数据集属、规范前缀和 Profile 的统一映射模型。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

from ramanv2.common.file_ops import create_temporary_path, publish_file, read_json, write_json


@dataclass(frozen=True)
class DatasetMapping:
    """描述规范种简称到属以及属到 Profile 的唯一映射。"""

    genera: dict[str, tuple[str, ...]]
    datasets: dict[str, tuple[str, ...]]

    @classmethod
    def from_payload(cls, payload: object) -> "DatasetMapping":
        if not isinstance(payload, dict):
            raise ValueError("dataset_mapping 必须是 JSON 对象")
        raw_genera = payload.get("genera")
        raw_datasets = payload.get("datasets")
        if not isinstance(raw_genera, dict) or not isinstance(raw_datasets, dict):
            raise ValueError("dataset_mapping 必须同时包含 genera 和 datasets")

        genera: dict[str, tuple[str, ...]] = {}
        prefix_owner: dict[str, str] = {}
        for raw_name, raw_prefixes in raw_genera.items():
            genus = str(raw_name).strip()
            if not genus or not isinstance(raw_prefixes, list) or not raw_prefixes:
                raise ValueError("每个属必须有非空前缀列表")
            prefixes = tuple(str(prefix).strip().upper() for prefix in raw_prefixes if str(prefix).strip())
            if not prefixes:
                raise ValueError(f"属 {raw_name} 必须有非空前缀列表")
            if len(prefixes) != len(set(prefixes)):
                raise ValueError(f"属 {genus} 存在重复前缀")
            if genus in genera:
                raise ValueError(f"属 {genus} 重复定义")
            for prefix in prefixes:
                if prefix in prefix_owner:
                    raise ValueError(f"前缀 {prefix} 同时属于多个属")
                prefix_owner[prefix] = genus
            genera[genus] = prefixes

        datasets: dict[str, tuple[str, ...]] = {}
        for raw_name, raw_genera in raw_datasets.items():
            dataset_name = str(raw_name).strip()
            if not dataset_name or not isinstance(raw_genera, list) or not raw_genera:
                raise ValueError(f"数据集 {raw_name} 必须有非空属列表")
            names = tuple(str(name).strip() for name in raw_genera if str(name).strip())
            if len(names) != len(set(names)):
                raise ValueError(f"数据集 {dataset_name} 存在重复属")
            unknown = sorted(set(names) - set(genera))
            if unknown:
                raise ValueError(f"数据集 {dataset_name} 包含未定义属：{unknown}")
            if dataset_name in datasets:
                raise ValueError(f"数据集 {dataset_name} 重复定义")
            datasets[dataset_name] = names
        return cls(genera=genera, datasets=datasets)

    @classmethod
    def from_json(cls, path: Path | str) -> "DatasetMapping":
        return cls.from_payload(read_json(path))

    @classmethod
    def from_workbook(
        cls,
        workbook_path: Path | str,
        columns: tuple[str, ...] | list[str],
        profile_rules: Mapping[str, object],
    ) -> "DatasetMapping":
        path = Path(workbook_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"分类表不存在：{path}")
        if len(columns) != 4:
            raise ValueError("分类表列配置必须依次为：革兰氏、科、属英文、规范种简称")
        gram_column, family_column, genus_column, prefix_column = columns
        gram_map = {
            str(key).strip(): str(value).strip()
            for key, value in dict(profile_rules.get("gram_map", {})).items()
        }
        profiles = tuple(str(name) for name in profile_rules.get("profiles", ()))
        micro_profile = str(profile_rules.get("micro_profile", "")).strip()
        if not gram_map or not profiles:
            raise ValueError("数据集未配置完整的 profile_rules")
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            headers = next(rows, None)
            if headers is None:
                raise ValueError("分类表为空")
            positions = {str(value).strip(): index for index, value in enumerate(headers) if value is not None}
            missing = [name for name in columns if name not in positions]
            if missing:
                raise ValueError(f"分类表缺少列：{missing}")

            genera_lists: dict[str, list[str]] = {}
            genus_profiles: dict[str, str] = {}
            prefix_owner: dict[str, str] = {}
            profile_genera: dict[str, set[str]] = {name: set() for name in profiles}
            for row in rows:
                gram = str(row[positions[gram_column]] or "").strip()
                family = str(row[positions[family_column]] or "").strip()
                genus = str(row[positions[genus_column]] or "").strip()
                prefix = str(row[positions[prefix_column]] or "").strip().upper()
                if not gram and not family and not genus and not prefix:
                    continue
                profile_name = gram_map.get(gram)
                if profile_name is None or profile_name not in profile_genera:
                    raise ValueError(f"分类表存在未配置的革兰氏分类：{gram!r}")
                if not genus or not prefix:
                    raise ValueError(f"分类表存在无效记录：{row}")
                previous_profile = genus_profiles.setdefault(genus, profile_name)
                if previous_profile != profile_name:
                    raise ValueError(f"属 {genus} 同时出现在多个分类：{previous_profile}、{profile_name}")
                owner = prefix_owner.get(prefix)
                if owner is not None and owner != genus:
                    raise ValueError(f"规范种简称 {prefix} 同时属于 {owner} 和 {genus}")
                prefix_owner[prefix] = genus
                if prefix not in genera_lists.setdefault(genus, []):
                    genera_lists[genus].append(prefix)
                profile_genera[profile_name].add(genus)
        finally:
            workbook.close()

        genera = {name: tuple(prefixes) for name, prefixes in sorted(genera_lists.items())}
        datasets = {
            profile_name: tuple(sorted(profile_genera[profile_name]))
            for profile_name in profiles
        }
        if micro_profile:
            datasets[micro_profile] = tuple(sorted(genera))
        return cls(genera=genera, datasets=datasets)

    def subset(self, profile_name: str) -> "DatasetMapping":
        """生成只包含一个 Profile 的映射，供单 Profile 构建使用。"""
        if profile_name not in self.datasets:
            raise ValueError(f"未知 Profile：{profile_name}")
        return DatasetMapping(genera=dict(self.genera), datasets={profile_name: self.datasets[profile_name]})

    def to_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "datasets": {name: list(genera) for name, genera in self.datasets.items()},
            "genera": {name: list(prefixes) for name, prefixes in self.genera.items()},
        }


def build_prefix_to_genus(mapping: DatasetMapping) -> dict[str, str]:
    """把属到前缀的映射展开为规范前缀到属的反查表。"""
    return {
        str(prefix).upper(): str(genus)
        for genus, prefixes in mapping.genera.items()
        for prefix in prefixes
    }


def load_dataset_mapping(path: Path | str) -> DatasetMapping:
    """读取 UTF-8 JSON 映射文件。"""
    return DatasetMapping.from_json(path)


def ensure_dataset_mapping(
    workbook_path: Path | str,
    mapping_path: Path | str,
    columns: tuple[str, ...] | list[str],
    profile_rules: Mapping[str, object],
) -> DatasetMapping:
    """按 Excel 修改时间生成或复用唯一的本地 JSON 映射。"""
    workbook = Path(workbook_path).resolve()
    target = Path(mapping_path).resolve()
    if target.is_file() and (not workbook.is_file() or target.stat().st_mtime_ns >= workbook.stat().st_mtime_ns):
        try:
            return load_dataset_mapping(target)
        except (OSError, ValueError, json.JSONDecodeError):
            # 配置可能来自旧页面或被中断写入；根据当前 Excel 原子重建。
            if not workbook.is_file():
                raise

    mapping = DatasetMapping.from_workbook(workbook, columns, profile_rules)
    temporary = create_temporary_path(target)
    try:
        write_json(temporary, mapping.to_payload())
        publish_file(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return mapping


def build_hierarchy_mapping(hierarchy: object) -> DatasetMapping:
    """用层级配置的一级前缀映射构造 lineage/删除扫描映射。"""
    mappings = getattr(hierarchy, "mappings", {}) or {}
    level_one = mappings.get("level_1", {}) if isinstance(mappings, dict) else {}
    genera: dict[str, list[str]] = {}
    for prefix, value in level_one.items():
        genera.setdefault(str(value), []).append(str(prefix).upper())
    return DatasetMapping(
        genera={name: tuple(prefixes) for name, prefixes in genera.items()},
        datasets={},
    )


__all__ = [
    "DatasetMapping",
    "build_hierarchy_mapping",
    "build_prefix_to_genus",
    "ensure_dataset_mapping",
    "load_dataset_mapping",
]
