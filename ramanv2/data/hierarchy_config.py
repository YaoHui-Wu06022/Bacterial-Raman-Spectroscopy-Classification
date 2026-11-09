"""数据集层级配置：dataset/<id>/hierarchy.json。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ramanv2.common.file_ops import read_json, write_json
from ramanv2.common.hierarchy import FLAT_LABEL_SCHEMA, HIERARCHY_LABEL_SCHEMA
from ramanv2.common.naming import parse_folder_prefix


HIERARCHY_FILE_NAME = "hierarchy.json"
HIERARCHY_VERSION = 1


@dataclass(frozen=True)
class HierarchyLevel:
    """一个业务层级；directory=True 的层级会生成 init 父目录。"""

    name: str
    display_name: str
    directory: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "HierarchyLevel":
        if not isinstance(payload, Mapping):
            raise ValueError("层级定义必须是 JSON 对象")
        name = str(payload.get("name", "")).strip()
        if not name.startswith("level_") or not name.removeprefix("level_").isdigit():
            raise ValueError(f"层级名称必须形如 level_1：{name!r}")
        display_name = str(payload.get("display_name", "")).strip() or name
        return cls(name=name, display_name=display_name, directory=bool(payload.get("directory", False)))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "directory": self.directory,
        }


@dataclass(frozen=True)
class HierarchySpec:
    """显式层级映射；缺失配置时通过 :meth:`flat` 退化为平展。"""

    levels: tuple[HierarchyLevel, ...]
    mappings: Mapping[str, Mapping[str, str]]
    version: int = HIERARCHY_VERSION
    explicit: bool = False

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("层级配置至少需要一个层级")
        names = [level.name for level in self.levels]
        if len(names) != len(set(names)):
            raise ValueError(f"层级名称重复：{names}")
        expected = [f"level_{index + 1}" for index in range(len(names))]
        if names != expected:
            raise ValueError(f"层级必须从 level_1 连续编号：{names}")
        directory_flags = [level.directory for level in self.levels]
        if any(directory_flags) and directory_flags != sorted(directory_flags, reverse=True):
            raise ValueError("directory 层级必须排在非 directory 层级之前")
        unknown = set(self.mappings) - set(names)
        if unknown:
            raise ValueError(f"映射包含未定义层级：{sorted(unknown)}")
        for level_name, mapping in self.mappings.items():
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{level_name} 的映射必须是对象")
            for prefix in mapping:
                if str(prefix) != str(prefix).upper():
                    raise ValueError(f"映射前缀必须大写：{prefix!r}")

    @classmethod
    def flat(cls) -> "HierarchySpec":
        """无 hierarchy.json 时的默认单层：按来源前缀生成父目录。"""
        return cls(
            levels=(HierarchyLevel("level_1", "类别", directory=True),),
            mappings={},
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "HierarchySpec":
        if not isinstance(payload, Mapping):
            raise ValueError("hierarchy.json 必须是 JSON 对象")
        version = int(payload.get("version", HIERARCHY_VERSION))
        if version != HIERARCHY_VERSION:
            raise ValueError(f"hierarchy.json 版本必须为 {HIERARCHY_VERSION}")
        raw_levels = payload.get("levels")
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError("hierarchy.json 必须包含非空 levels 列表")
        levels = tuple(HierarchyLevel.from_payload(item) for item in raw_levels)
        raw_mappings = payload.get("mappings", {})
        if not isinstance(raw_mappings, Mapping):
            raise ValueError("hierarchy.json.mappings 必须是对象")
        mappings = {
            str(level_name): {
                str(prefix).upper(): str(value)
                for prefix, value in mapping.items()
            }
            for level_name, mapping in raw_mappings.items()
            if isinstance(mapping, Mapping)
        }
        return cls(levels=levels, mappings=mappings, explicit=True)

    @classmethod
    def load(cls, path: Path | str | None) -> "HierarchySpec":
        if path is None:
            return cls.flat()
        target = Path(path)
        if not target.is_file():
            return cls.flat()
        return cls.from_payload(read_json(target))

    @property
    def is_flat(self) -> bool:
        return not self.explicit

    @property
    def label_schema(self) -> str:
        return FLAT_LABEL_SCHEMA if self.is_flat else HIERARCHY_LABEL_SCHEMA

    @property
    def directory_levels(self) -> tuple[HierarchyLevel, ...]:
        return tuple(level for level in self.levels if level.directory)

    @property
    def level_names(self) -> tuple[str, ...]:
        return tuple(level.name for level in self.levels)

    @property
    def directory_depth(self) -> int:
        return len(self.directory_levels)

    def resolve_values(self, group_name: str, *, strict_directory: bool = True) -> dict[str, str]:
        """按来源文件夹前缀解析各层级取值；显式目录层级缺失映射时默认报错。"""
        raw_prefix = parse_folder_prefix(group_name)
        prefix = raw_prefix.upper()
        values: dict[str, str] = {}
        for level in self.levels:
            value = self._lookup(level.name, prefix)
            if value is None:
                if level.directory and self.mappings and strict_directory:
                    raise ValueError(f"层级 {level.name} 未映射来源前缀：{group_name} ({prefix})")
                value = raw_prefix
            values[level.name] = value
        return values

    def init_relative_dir(self, group_name: str) -> Path:
        """init 中的父目录路径；组文件夹由调用方追加。"""
        values = self.resolve_values(group_name)
        parts = [values[level.name] for level in self.directory_levels]
        return Path(*parts) if parts else Path(".")

    def class_relative_path(self, group_name: str) -> Path:
        """train 中的类别路径：父目录值 + 非父目录层级值。"""
        values = self.resolve_values(group_name)
        parts = [values[level.name] for level in self.levels]
        return Path(*parts) if parts else Path(".")

    def init_class_relative(self, relative_dir: Path, leaf_name: str) -> Path:
        """从 init 叶子相对目录还原 train 类别路径。"""
        directory_parts = tuple(relative_dir.parts[: self.directory_depth])
        values = self.resolve_values(leaf_name)
        extra = [values[level.name] for level in self.levels if not level.directory]
        parts = [*directory_parts, *extra]
        return Path(*parts) if parts else Path(".")

    def expected_label(self, group_name: str, level_name: str) -> str:
        """返回测试文件夹在指定层级的预期标签。"""
        if level_name not in self.level_names:
            raise ValueError(f"未知层级：{level_name}")
        return self.resolve_values(group_name)[level_name]

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "levels": [level.to_payload() for level in self.levels],
            "mappings": {
                level_name: dict(sorted(mapping.items()))
                for level_name, mapping in sorted(self.mappings.items())
            },
        }

    def signature(self) -> str:
        text = json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _lookup(self, level_name: str, prefix: str) -> str | None:
        mapping = self.mappings.get(level_name, {})
        if prefix in mapping:
            return mapping[prefix]
        for key in sorted(mapping, key=len, reverse=True):
            if prefix.startswith(key):
                return mapping[key]
        return None


def write_hierarchy(path: Path | str, spec: HierarchySpec) -> None:
    write_json(path, spec.to_payload())
