"""数据集发现与 dataset.json 配置。

数据集就是 ``dataset/<id>/`` 下含 ``cosmic_data`` 的目录；层级由
``dataset.json`` 的 ``label_source`` 与层级投影配置派生。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ramanv2.common.defaults import DEFAULT_BAD_BANDS
from ramanv2.common.file_ops import read_json, write_json
from ramanv2.common.naming import ANALYSIS_DELETE_RELATIVE, PIPELINE_DIR_NAME
from ramanv2.common.paths import DATASET_ROOT
from ramanv2.data.dataset_mapping import MAPPING_FILE_NAME, ensure_dataset_mapping
from ramanv2.data.hierarchy_spec import (
    HIERARCHY_FILE_NAME,
    HierarchySpec,
    build_projected_hierarchy,
)


DATASET_CONFIG_FILE_NAME = "dataset.json"
DATASET_CONFIG_VERSION = 1
_RESERVED_DATASET_CHILDREN = {
    "cosmic_data",
    "shift_data",
    "init",
    "init_test",
    "fig_init",
    "extra_data",
    ANALYSIS_DELETE_RELATIVE.name,
    PIPELINE_DIR_NAME,
}


@dataclass(frozen=True)
class DatasetSpec:
    """一个可独立维护的数据集及其目录布局。"""

    dataset_id: str
    display_name: str
    base_dir: Path
    directory: Path
    bad_bands: tuple[tuple[float, float], ...]
    hierarchy: HierarchySpec
    is_default: bool = False
    classification_workbook: str | None = None
    supplement_dir: str | None = None
    label_source: Mapping[str, Any] = field(default_factory=dict)
    taxonomy_labels: Mapping[str, str] = field(default_factory=dict)
    hierarchy_config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        return self.hierarchy.is_flat

    @property
    def dataset_dir(self) -> Path:
        return self.directory.resolve()

    @property
    def cosmic_dir(self) -> Path:
        return self.dataset_dir / "cosmic_data"

    @property
    def shift_dir(self) -> Path:
        return self.dataset_dir / "shift_data"

    @property
    def init_dir(self) -> Path:
        return self.dataset_dir / "init"

    @property
    def init_test_dir(self) -> Path:
        return self.dataset_dir / "init_test"

    @property
    def fig_dir(self) -> Path:
        return self.dataset_dir / "fig_init"

    @property
    def metadata_dir(self) -> Path:
        return self.dataset_dir / PIPELINE_DIR_NAME

    @property
    def config_path(self) -> Path:
        return self.dataset_dir / DATASET_CONFIG_FILE_NAME

    @property
    def hierarchy_path(self) -> Path:
        return self.dataset_dir / HIERARCHY_FILE_NAME

    @property
    def workbook_path(self) -> Path | None:
        return self.resolve_subdir(self.classification_workbook)

    @property
    def supplement_path(self) -> Path | None:
        return self.resolve_subdir(self.supplement_dir)

    def resolve_subdir(self, relative: str | None) -> Path | None:
        if relative is None or not str(relative).strip():
            return None
        target = (self.dataset_dir / relative).resolve()
        if target != self.dataset_dir and self.dataset_dir not in target.parents:
            raise ValueError(f"数据集路径不能越界：{relative}")
        return target

    def hierarchy_signature(self) -> str:
        return self.hierarchy.signature()


@dataclass(frozen=True)
class DatasetRegistry:
    """扫描 ``dataset`` 目录得到的数据集集合。"""

    datasets: tuple[DatasetSpec, ...]
    default_dataset_id: str

    @classmethod
    def load(
        cls,
        dataset_root: Path | str | None = None,
    ) -> "DatasetRegistry":
        root = _resolve_dataset_root(dataset_root)
        specs: list[DatasetSpec] = []
        if root.is_dir():
            for directory in sorted(root.iterdir(), key=lambda path: path.name):
                if not directory.is_dir():
                    continue
                # 本地数据集以 cosmic_data 为锚；远程训练包只有配置与 init/init_test。
                if (directory / "cosmic_data").is_dir() or (directory / DATASET_CONFIG_FILE_NAME).is_file():
                    specs.append(_build_spec(directory, root))
        if not specs:
            raise FileNotFoundError(f"dataset 下没有包含 cosmic_data 的数据集：{root}")
        default_id = next(
            (spec.dataset_id for spec in specs if spec.is_default),
            specs[0].dataset_id,
        )
        registry = cls(datasets=tuple(specs), default_dataset_id=default_id)
        registry.validate()
        return registry

    def list_datasets(self) -> list[DatasetSpec]:
        return list(self.datasets)

    def resolve(self, dataset_id: str | None = None) -> DatasetSpec:
        key = self.default_dataset_id if dataset_id is None else str(dataset_id).strip()
        if key == "":
            key = self.default_dataset_id
        for spec in self.datasets:
            if spec.dataset_id == key or spec.directory.name == key:
                return spec
        raise KeyError(f"未知数据集：{key}")

    def resolve_for_profile(self, profile_id: str) -> DatasetSpec:
        return self.resolve(profile_id)

    def validate(self) -> None:
        if not self.datasets:
            raise ValueError("没有可用数据集")
        ids: set[str] = set()
        for spec in self.datasets:
            if spec.dataset_id in ids:
                raise ValueError(f"dataset_id 重复：{spec.dataset_id}")
            ids.add(spec.dataset_id)
        if self.default_dataset_id not in ids:
            raise ValueError(f"默认数据集未注册：{self.default_dataset_id}")


def load_dataset_registry(
    dataset_root: Path | str | None = None,
) -> DatasetRegistry:
    return DatasetRegistry.load(dataset_root)


def load_dataset_config(dataset_dir: Path | str) -> dict[str, Any]:
    """读取 dataset.json；不存在时返回目录名与默认坏区。"""
    directory = Path(dataset_dir).resolve()
    path = directory / DATASET_CONFIG_FILE_NAME
    if not path.is_file():
        return {
            "version": DATASET_CONFIG_VERSION,
            "display_name": directory.name,
            "bad_bands": [list(item) for item in DEFAULT_BAD_BANDS],
        }
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"dataset.json 必须是 JSON 对象：{path}")
    return dict(payload)


def write_dataset_config(
    dataset_dir: Path | str,
    display_name: str,
    bad_bands: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    dataset_id: str | None = None,
    label_source: Mapping[str, Any] | None = None,
    taxonomy_labels: Mapping[str, str] | None = None,
    hierarchy: Mapping[str, Any] | None = None,
    is_default: bool = False,
) -> Path:
    directory = Path(dataset_dir).resolve()
    payload = {
        "version": DATASET_CONFIG_VERSION,
        "dataset_id": str(dataset_id).strip() if dataset_id else directory.name,
        "display_name": str(display_name).strip() or directory.name,
        "bad_bands": [[float(start), float(end)] for start, end in bad_bands],
        "default": bool(is_default),
    }
    if label_source is not None:
        payload["label_source"] = dict(label_source)
    if taxonomy_labels is not None:
        payload["taxonomy_labels"] = dict(taxonomy_labels)
    if hierarchy is not None:
        payload["hierarchy"] = dict(hierarchy)
    _validate_bad_bands(payload["bad_bands"])
    path = directory / DATASET_CONFIG_FILE_NAME
    write_json(path, payload)
    return path


def annotate_dataset_build(report: dict[str, Any], spec: DatasetSpec) -> dict[str, Any]:
    """把层级与坏区签名写入构建报告，供 UI 判断是否待重建。"""
    report["hierarchy_signature"] = spec.hierarchy_signature()
    report["hierarchy"] = spec.hierarchy.to_payload()
    report["bad_bands"] = [list(item) for item in spec.bad_bands]
    return report


def resolve_project_root_for_dataset_dir(dataset_dir: Path | str) -> Path:
    """从数据集目录向上定位项目根；兼容旧根与 ``dataset/<id>`` 布局。"""
    dataset_root = Path(dataset_dir).resolve()
    if (dataset_root / "cosmic_data").is_dir() and dataset_root.parent.name == "dataset":
        return dataset_root.parent.parent
    return dataset_root.parent


def _build_spec(
    directory: Path,
    base_dir: Path,
) -> DatasetSpec:
    config = load_dataset_config(directory)
    dataset_id = str(config.get("dataset_id", "")).strip() or directory.name
    bad_bands = _read_bad_bands(config.get("bad_bands"))
    label_source_payload = config.get("label_source")
    label_source = (
        dict(label_source_payload)
        if isinstance(label_source_payload, Mapping)
        else {}
    )
    taxonomy_labels_payload = config.get("taxonomy_labels")
    taxonomy_labels = (
        {str(key): str(value) for key, value in taxonomy_labels_payload.items()}
        if isinstance(taxonomy_labels_payload, Mapping)
        else {}
    )
    hierarchy_config = config.get("hierarchy")
    hierarchy_config = (
        dict(hierarchy_config)
        if isinstance(hierarchy_config, Mapping)
        else {}
    )
    mapping = (
        ensure_dataset_mapping(
            label_source,
            directory,
            directory / PIPELINE_DIR_NAME / MAPPING_FILE_NAME,
        )
        if label_source
        else None
    )
    hierarchy = (
        build_projected_hierarchy(
            mapping,
            hierarchy_config.get("levels", ()),
            hierarchy_config.get("directories", ()),
            taxonomy_labels,
        )
        if mapping is not None and hierarchy_config
        else HierarchySpec.load(directory / HIERARCHY_FILE_NAME)
    )
    workbook_name = (
        str(label_source.get("file", "")).strip()
        if str(label_source.get("kind", "")).strip().lower() == "workbook"
        else ""
    )
    classification_workbook = (
        workbook_name
        if workbook_name and (directory / workbook_name).is_file()
        else _find_classification_workbook(
            directory,
            tuple(str(column) for column in dict(label_source.get("levels") or {}).values()),
        )
    )
    return DatasetSpec(
        dataset_id=dataset_id,
        display_name=str(config.get("display_name", "")).strip() or dataset_id,
        base_dir=base_dir.resolve(),
        directory=directory.resolve(),
        bad_bands=bad_bands,
        hierarchy=hierarchy,
        is_default=bool(config.get("default", False)),
        classification_workbook=classification_workbook,
        supplement_dir="extra_data" if (directory / "extra_data").is_dir() else None,
        label_source=label_source,
        taxonomy_labels=taxonomy_labels,
        hierarchy_config=hierarchy_config,
    )


def _read_bad_bands(value: Any) -> tuple[tuple[float, float], ...]:
    if value is None:
        return DEFAULT_BAD_BANDS
    if not isinstance(value, list):
        raise ValueError("dataset.json.bad_bands 必须是列表")
    bands = tuple((float(item[0]), float(item[1])) for item in value if isinstance(item, (list, tuple)) and len(item) == 2)
    if not bands:
        return DEFAULT_BAD_BANDS
    _validate_bad_bands([list(item) for item in bands])
    return bands


def _validate_bad_bands(bands: list[list[float]]) -> None:
    for start, end in bands:
        if not start < end:
            raise ValueError(f"坏区起点必须小于终点：{start}-{end}")


def _find_classification_workbook(
    directory: Path,
    required_columns: tuple[str, ...],
) -> str | None:
    """扫描数据集内 xlsx，返回含配置列名的相对文件名。"""
    if not required_columns:
        return None
    from openpyxl import load_workbook

    for path in sorted(directory.glob("*.xlsx")):
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
        except (OSError, ValueError):
            continue
        try:
            sheet = workbook.active
            header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
            names = {str(value).strip() for value in header if value is not None}
        finally:
            workbook.close()
        if all(column in names for column in required_columns):
            return path.name
    return None


def _resolve_dataset_root(dataset_root: Path | str | None) -> Path:
    root = DATASET_ROOT if dataset_root is None else Path(dataset_root)
    return root.resolve()
