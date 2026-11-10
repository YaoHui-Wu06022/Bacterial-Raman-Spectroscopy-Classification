"""常规数据集 profile 映射与按数据集解析的目录布局。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.data.datasets import (
    FLAT_PROFILE_ID,
    DatasetRegistry,
    DatasetSpec,
)


# 测试母本目录名；由数据构建阶段统一使用。
TEST_ROOT_NAME = "init_test"


@dataclass(frozen=True)
class DatasetProfile:
    """描述一个 profile 及其在所属数据集内的阶段目录名称。"""

    profile_id: str
    dataset_name: str
    dataset_id: str = ""
    dataset_dir_name: str = ""
    subdir: str = ""
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
    """不加载注册表，直接从数据集规格构造 profile 列表。"""
    if spec.is_flat:
        return [_build_flat_profile(spec)]
    return [_build_profile(profile_id, spec) for profile_id in spec.profiles]


def select_profile_id(spec: DatasetSpec, profile_key: str | None = None) -> str:
    """在数据集内选择 profile；空键取 dataset.json 默认项，再取第一个。"""
    key = "" if profile_key is None else str(profile_key).strip()
    if spec.is_flat:
        if key and key != FLAT_PROFILE_ID:
            raise KeyError(
                f"扁平数据集 {spec.dataset_id} 只有 profile {FLAT_PROFILE_ID}"
            )
        return FLAT_PROFILE_ID
    if key:
        if key not in spec.profiles:
            raise KeyError(f"数据集 {spec.dataset_id} 没有 profile：{key}")
        return key
    if not spec.profiles:
        raise KeyError(f"数据集 {spec.dataset_id} 没有可用 profile")
    preferred = spec.default_profile
    return preferred if preferred in spec.profiles else spec.profiles[0]


def resolve_spec_profile(spec: DatasetSpec, profile_key: str | None = None) -> DatasetProfile:
    """在指定数据集规格内解析 profile。"""
    profile_id = select_profile_id(spec, profile_key)
    return _build_flat_profile(spec) if spec.is_flat else _build_profile(profile_id, spec)


def list_profiles(
    dataset_id: str | None = None,
    project_root: Path | str | None = None,
) -> list[DatasetProfile]:
    """返回指定数据集目录中发现的 profile。"""
    return list_spec_profiles(_load_registry(project_root).resolve(dataset_id))


def resolve_selection(
    dataset_id: str | None = None,
    profile_id: str | None = None,
    project_root: Path | str | None = None,
) -> DatasetSelection:
    """唯一的 dataset/profile 解析入口。

    未给数据集时按 profile 归属反查；两者都为空时使用默认数据集。
    """
    registry = _load_registry(project_root)
    dataset_key = "" if dataset_id is None else str(dataset_id).strip()
    profile_key = "" if profile_id is None else str(profile_id).strip()
    if dataset_key:
        spec = registry.resolve(dataset_key)
    elif profile_key:
        spec = registry.resolve_for_profile(profile_key)
    else:
        spec = registry.resolve()
    profile = resolve_spec_profile(spec, profile_key)
    return DatasetSelection(
        spec=spec,
        profile=profile,
        dataset_dir=spec.profile_root(profile.profile_id),
    )


def _build_profile(profile_id: str, spec: DatasetSpec) -> DatasetProfile:
    return DatasetProfile(
        profile_id=profile_id,
        dataset_name=profile_id,
        dataset_id=spec.dataset_id,
        dataset_dir_name=spec.directory.name,
        subdir=profile_id,
    )


def _build_flat_profile(spec: DatasetSpec) -> DatasetProfile:
    return DatasetProfile(
        profile_id=FLAT_PROFILE_ID,
        dataset_name=spec.dataset_id,
        dataset_id=spec.dataset_id,
        dataset_dir_name=spec.directory.name,
        subdir="",
    )


def _load_registry(project_root: Path | str | None) -> DatasetRegistry:
    """项目根目录下的 ``dataset`` 子目录即注册表根目录。"""
    dataset_root = None if project_root is None else Path(project_root) / "dataset"
    return DatasetRegistry.load(dataset_root)
