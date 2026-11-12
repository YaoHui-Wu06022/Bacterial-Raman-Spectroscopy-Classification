"""数据集标签映射：从分类表或目录前缀构建通用层级表。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

from ramanv2.common.file_ops import create_temporary_path, publish_file, read_json, write_json
from ramanv2.common.naming import SPECIAL_FOLDER_NAMES, parse_folder_prefix


MAPPING_VERSION = 3
MAPPING_FILE_NAME = "dataset_mapping.json"


@dataclass(frozen=True)
class DatasetMapping:
    """前缀到各层级取值的只读映射。"""

    levels: tuple[str, ...]
    prefix_level: str
    grouping_key: str
    rows: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("映射至少需要一个层级")
        if len(self.levels) != len(set(self.levels)):
            raise ValueError(f"映射层级重复：{self.levels}")
        if self.prefix_level not in self.levels:
            raise ValueError(f"前缀层级不在层级列表中：{self.prefix_level}")
        if self.grouping_key not in self.levels:
            raise ValueError(f"分组层级不在层级列表中：{self.grouping_key}")
        if not self.rows:
            raise ValueError("映射没有任何前缀")
        for prefix, values in self.rows.items():
            if not prefix:
                raise ValueError("映射包含空前缀")
            missing = [name for name in self.levels if not str(values.get(name, "")).strip()]
            if missing:
                raise ValueError(f"前缀 {prefix} 缺少层级取值：{missing}")

    @property
    def genera(self) -> dict[str, tuple[str, ...]]:
        """按分组层级聚合前缀。"""
        grouped: dict[str, list[str]] = {}
        for prefix, values in self.rows.items():
            grouped.setdefault(values[self.grouping_key], []).append(prefix)
        return {name: tuple(prefixes) for name, prefixes in grouped.items()}

    def prefix_to_group(self) -> dict[str, str]:
        """返回前缀到分组层级的反查表。"""
        return {
            prefix: values[self.grouping_key]
            for prefix, values in self.rows.items()
        }

    def prefix_values(self, level_name: str) -> dict[str, str]:
        """返回指定层级的前缀到取值表。"""
        if level_name not in self.levels:
            raise ValueError(f"未知层级：{level_name}")
        return {
            prefix: values[level_name]
            for prefix, values in self.rows.items()
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "version": MAPPING_VERSION,
            "levels": list(self.levels),
            "prefix_level": self.prefix_level,
            "grouping_key": self.grouping_key,
            "rows": {prefix: dict(values) for prefix, values in self.rows.items()},
        }

    @classmethod
    def from_payload(cls, payload: object) -> "DatasetMapping":
        if not isinstance(payload, dict):
            raise ValueError("dataset_mapping 必须是 JSON 对象")
        if payload.get("version") != MAPPING_VERSION:
            raise ValueError(f"不支持的映射版本：{payload.get('version')}")
        levels = tuple(str(name) for name in payload.get("levels", ()))
        rows_payload = payload.get("rows")
        if not isinstance(rows_payload, dict):
            raise ValueError("dataset_mapping 必须包含 rows")
        rows = {
            str(prefix).strip().upper(): {
                str(name): str(value).strip()
                for name, value in dict(values).items()
            }
            for prefix, values in rows_payload.items()
        }
        return cls(
            levels=levels,
            prefix_level=str(payload.get("prefix_level", "")),
            grouping_key=str(payload.get("grouping_key", "")),
            rows=rows,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "DatasetMapping":
        return cls.from_payload(read_json(path))


def build_prefix_to_genus(mapping: DatasetMapping) -> dict[str, str]:
    """返回前缀到分组层级的反查表；名称沿用旧消费方。"""
    return mapping.prefix_to_group()


def from_workbook(
    workbook_path: Path | str,
    levels: Mapping[str, str],
    prefix_level: str,
    grouping_key: str,
) -> DatasetMapping:
    """按层级别名和列名读取分类表，构建通用前缀表。"""
    path = Path(workbook_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"分类表不存在：{path}")
    level_names = tuple(levels)
    if prefix_level not in levels:
        raise ValueError(f"前缀层级不在层级列表中：{prefix_level}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        headers = next(rows, None)
        if headers is None:
            raise ValueError("分类表为空")
        positions = {
            str(value).strip(): index
            for index, value in enumerate(headers)
            if value is not None
        }
        missing = [column for column in levels.values() if column not in positions]
        if missing:
            raise ValueError(f"分类表缺少列：{missing}")
        rows_by_prefix: dict[str, dict[str, str]] = {}
        for row in rows:
            values = {
                name: str(row[positions[column]] or "").strip()
                for name, column in levels.items()
            }
            prefix = values[prefix_level].upper()
            if not any(values.values()):
                continue
            missing_values = [name for name in level_names if not values[name]]
            if missing_values:
                raise ValueError(f"分类表存在空层级取值：{missing_values}，行：{row}")
            if prefix in rows_by_prefix:
                raise ValueError(f"规范前缀重复：{prefix}")
            rows_by_prefix[prefix] = values
    finally:
        workbook.close()
    if not rows_by_prefix:
        raise ValueError("分类表没有有效数据行")
    return DatasetMapping(
        levels=level_names,
        prefix_level=prefix_level,
        grouping_key=grouping_key,
        rows=rows_by_prefix,
    )


def from_folder_prefix(
    root: Path | str,
    level_name: str,
    *,
    label_root: Path | str | None = None,
) -> DatasetMapping:
    """从目录前缀构建单层映射；用于没有分类表的扁平数据集。"""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"前缀来源目录不存在：{root_path}")
    date_dirs = sorted(
        path
        for path in root_path.iterdir()
        if path.is_dir() and path.name.isdigit() and len(path.name) == 8
    )
    source_dirs: list[Path] = []
    if date_dirs:
        for date_dir in date_dirs:
            source_dirs.extend(
                folder
                for folder in sorted(date_dir.iterdir())
                if folder.is_dir()
            )
    else:
        source_dirs = sorted(path for path in root_path.iterdir() if path.is_dir())
    prefixes: list[str] = []
    for folder in source_dirs:
        if folder.name in SPECIAL_FOLDER_NAMES or folder.name.startswith("."):
            continue
        prefix = parse_folder_prefix(folder.name, uppercase_enable=True)
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    if not prefixes:
        raise RuntimeError(f"目录中没有可识别的前缀：{root_path}")
    label_by_prefix: dict[str, str] = {}
    if label_root is not None and Path(label_root).is_dir():
        label_by_prefix = {
            path.name.upper(): path.name
            for path in Path(label_root).iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
    rows = {
        prefix: {level_name: label_by_prefix.get(prefix, prefix)}
        for prefix in prefixes
    }
    return DatasetMapping(
        levels=(level_name,),
        prefix_level=level_name,
        grouping_key=level_name,
        rows=rows,
    )


def load_dataset_mapping(path: Path | str) -> DatasetMapping:
    """读取 UTF-8 JSON 映射文件。"""
    return DatasetMapping.from_json(path)


def ensure_dataset_mapping(
    label_source: Mapping[str, object],
    dataset_dir: Path | str,
    target_path: Path | str,
) -> DatasetMapping:
    """按 label_source 生成或复用本地映射缓存。"""
    dataset_root = Path(dataset_dir).resolve()
    target = Path(target_path).resolve()
    kind = str(label_source.get("kind", "")).strip().lower()
    source_root: Path | None = None
    workbook: Path | None = None
    source_missing = False
    if kind == "workbook":
        workbook = dataset_root / str(label_source.get("file", "")).strip()
        if not workbook.is_file():
            source_missing = True
    elif kind == "folder_prefix":
        source_root = dataset_root / str(label_source.get("root", "cosmic_data")).strip()
        if not source_root.is_dir():
            source_root = dataset_root / "init"
        if not source_root.is_dir():
            source_missing = True
    else:
        raise ValueError(f"未知 label_source.kind：{kind!r}")

    if source_missing:
        cached = _load_cached_mapping(target, label_source)
        if cached is not None:
            return cached
        raise FileNotFoundError(f"缺少映射来源，且没有可用缓存：{target}")

    source_mtime = (
        workbook.stat().st_mtime_ns
        if workbook is not None
        else max(
            [source_root.stat().st_mtime_ns]
            + [
                path.stat().st_mtime_ns
                for path in source_root.iterdir()
                if path.is_dir()
            ]
        )
    )
    if kind == "workbook" and target.is_file() and target.stat().st_mtime_ns >= source_mtime:
        cached = _load_cached_mapping(target, label_source)
        if cached is not None:
            return cached

    levels = label_source.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("label_source.levels 必须是对象")
    prefix_level = str(label_source.get("prefix_level", "")).strip()
    grouping_key = str(label_source.get("grouping_key", "")).strip()
    if kind == "workbook":
        mapping = from_workbook(
            workbook,
            {str(name): str(column) for name, column in levels.items()},
            prefix_level,
            grouping_key,
        )
    else:
        mapping = from_folder_prefix(
            source_root,
            prefix_level,
            label_root=dataset_root / "init",
        )
    temporary = create_temporary_path(target)
    try:
        payload = mapping.to_payload()
        payload["config"] = _mapping_config_payload(label_source)
        write_json(temporary, payload)
        publish_file(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return mapping


def _mapping_config_payload(
    label_source: Mapping[str, object],
) -> dict[str, object]:
    """生成映射缓存的配置签名，配置变化时强制重建。"""
    return {"label_source": dict(label_source)}


def _load_cached_mapping(
    target: Path,
    label_source: Mapping[str, object],
) -> DatasetMapping | None:
    """读取与当前配置一致的映射缓存；缺失或不兼容时返回 None。"""
    if not target.is_file():
        return None
    try:
        cached_payload = read_json(target)
        if cached_payload.get("config") == _mapping_config_payload(label_source):
            return DatasetMapping.from_payload(cached_payload)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


__all__ = [
    "DatasetMapping",
    "MAPPING_FILE_NAME",
    "MAPPING_VERSION",
    "build_prefix_to_genus",
    "ensure_dataset_mapping",
    "from_folder_prefix",
    "from_workbook",
    "load_dataset_mapping",
]
