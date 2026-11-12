"""数据集主数据源解析与阶段目录布局。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.data.datasets import DatasetRegistry, DatasetSpec
from ramanv2.data.hierarchy_spec import HierarchySpec


# 测试母本目录名；由数据构建阶段统一使用。
TEST_ROOT_NAME = "init_test"


@dataclass(frozen=True)
class DatasetProfile:
    """描述数据集的唯一主数据源及其阶段目录名称。"""

    profile_id: str
    dataset_name: str
    dataset_id: str = ""
    dataset_dir_name: str = ""
    root_init: str = "init"
    root_init_test: str = "init_test"
    root_train_clean: str = "train"
    root_test: str = "test"
    root_train_fig: str = "fig_train"


@dataclass(frozen=True)
class DatasetSelection:
    """一次解析得到的数据集、profile 与其阶段目录根。

    ``dataset_dir`` 是层级数据集的 profile 子目录，扁平数据集则是数据集根；
    训练、测试与打包都从这里派生，避免各入口重复扫描注册表或推断父目录。
    """

    spec: DatasetSpec
    profile: DatasetProfile
    dataset_dir: Path
    hierarchy: HierarchySpec

    @property
    def train_dir(self) -> Path:
        """优先返回构建后的 train；缺失时回退 init，再回退 dataset_dir。"""
        train_dir = self.dataset_dir / self.profile.root_train_clean
        if train_dir.is_dir():
            return train_dir
        init_dir = self.dataset_dir / self.profile.root_init
        return init_dir if init_dir.is_dir() else self.dataset_dir

    @property
    def test_root(self) -> Path:
        """返回数据集的 init_test 母本目录；profile 测试集位于 profile 根下。"""
        return self.spec.init_test_dir


def list_spec_profiles(spec: DatasetSpec) -> list[DatasetProfile]:
    """每个数据集只有一个主数据源。"""
    return [_build_profile(spec)]


def select_profile_id(spec: DatasetSpec, profile_key: str | None = None) -> str:
    """返回数据集主数据源标识；传入其他键视为错误。"""
    key = "" if profile_key is None else str(profile_key).strip()
    if key and key != spec.dataset_id:
        raise KeyError(f"数据集 {spec.dataset_id} 只有主数据源 {spec.dataset_id}")
    return spec.dataset_id


def resolve_spec_profile(spec: DatasetSpec, profile_key: str | None = None) -> DatasetProfile:
    """返回数据集的唯一主数据源。"""
    select_profile_id(spec, profile_key)
    return _build_profile(spec)


def list_profiles(
    dataset_id: str | None = None,
    project_root: Path | str | None = None,
) -> list[DatasetProfile]:
    """返回指定数据集的主数据源。"""
    return list_spec_profiles(_load_registry(project_root).resolve(dataset_id))


def resolve_selection(
    dataset_id: str | None = None,
    profile_id: str | None = None,
    project_root: Path | str | None = None,
) -> DatasetSelection:
    """唯一的 dataset 解析入口；profile 参数仅用于兼容等价的数据集标识。"""
    registry = _load_registry(project_root)
    dataset_key = "" if dataset_id is None else str(dataset_id).strip()
    profile_key = "" if profile_id is None else str(profile_id).strip()
    if dataset_key:
        spec = registry.resolve(dataset_key)
    elif profile_key:
        spec = registry.resolve(profile_key)
    else:
        spec = registry.resolve()
    profile = resolve_spec_profile(spec, profile_key)
    return DatasetSelection(
        spec=spec,
        profile=profile,
        dataset_dir=spec.dataset_dir,
        hierarchy=spec.hierarchy,
    )


def _build_profile(spec: DatasetSpec) -> DatasetProfile:
    return DatasetProfile(
        profile_id=spec.dataset_id,
        dataset_name=spec.dataset_id,
        dataset_id=spec.dataset_id,
        dataset_dir_name=spec.directory.name,
    )


def _load_registry(project_root: Path | str | None) -> DatasetRegistry:
    """项目根目录下的 ``dataset`` 子目录即注册表根目录。"""
    dataset_root = None if project_root is None else Path(project_root) / "dataset"
    return DatasetRegistry.load(dataset_root)
