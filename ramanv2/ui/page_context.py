"""Streamlit 页面共享的数据上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.common.paths import PROJECT_ROOT
from ramanv2.data.dataset_mapping import (
    DatasetMapping,
    MAPPING_FILE_NAME,
    ensure_dataset_mapping,
)
from ramanv2.data.datasets import DatasetRegistry, DatasetSpec, load_dataset_registry
from ramanv2.data.hierarchy_spec import HierarchySpec
from ramanv2.data.profiles import (
    DatasetProfile,
    DatasetSelection,
    list_spec_profiles,
    select_profile_id,
)
from ramanv2.pipeline.state import pipeline_state_path_for_spec


@dataclass(frozen=True)
class PageContext:
    """页面之间共享的数据目录与解析结果，不保存页面交互状态。"""

    project_root: Path
    registry: DatasetRegistry
    dataset_id: str
    dataset_spec: DatasetSpec
    hierarchy: HierarchySpec
    profiles: tuple[DatasetProfile, ...]
    has_dataset_config: bool
    bad_bands: tuple[tuple[float, float], ...]
    data_root: Path
    shift_root: Path
    init_root: Path
    init_test_root: Path
    fig_root: Path
    supplement_root: Path
    metadata_dir: Path
    state_file: Path
    output_root: Path
    bundle_root: Path
    classification_workbook: Path | None
    dataset_mapping_file: Path

    @property
    def default_profile_id(self) -> str:
        """优先使用 dataset.json 的默认 profile，否则取第一个可用 profile。"""
        return select_profile_id(self.dataset_spec)

    def profile(self, profile_id: str | None = None) -> DatasetProfile:
        """返回指定 profile；未指定时使用默认 profile。"""
        selected = select_profile_id(self.dataset_spec, profile_id)
        for profile in self.profiles:
            if profile.profile_id == selected:
                return profile
        raise KeyError(f"数据集 {self.dataset_id} 没有 profile：{selected or '<空>'}")

    def profile_dir(self, profile: DatasetProfile) -> Path:
        """返回主数据源的阶段目录根。"""
        del profile
        return self.dataset_spec.dataset_dir

    def training_dir(self, profile: DatasetProfile) -> Path:
        """返回主数据源的 train 目录，缺失时回退 init。"""
        return DatasetSelection(
            self.dataset_spec,
            profile,
            self.profile_dir(profile),
            self.dataset_spec.hierarchy,
        ).train_dir

    def experiment_root(self, profile: DatasetProfile) -> Path:
        """返回当前数据集的实验输出目录。"""
        del profile
        return self.output_root / self.dataset_spec.directory.name

    def experiment_roots(self) -> tuple[Path, ...]:
        """返回当前数据集的实验输出目录。"""
        return (self.experiment_root(self.profile()),)

    def dataset_mapping(self) -> DatasetMapping:
        """按 label_source 返回当前数据集的通用前缀映射。"""
        return ensure_dataset_mapping(
            self.dataset_spec.label_source,
            self.dataset_spec.dataset_dir,
            self.dataset_mapping_file,
        )


def build_page_context(
    dataset_id: str | None = None,
    registry: DatasetRegistry | None = None,
    project_root: Path | str | None = None,
) -> PageContext:
    """按数据集目录构建页面上下文；registry 由上层传入时不再重复扫描。"""
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    registry = registry or load_dataset_registry(root / "dataset")
    spec = registry.resolve(dataset_id)
    return PageContext(
        project_root=root,
        registry=registry,
        dataset_id=spec.dataset_id,
        dataset_spec=spec,
        hierarchy=spec.hierarchy,
        profiles=tuple(list_spec_profiles(spec)),
        has_dataset_config=spec.config_path.is_file(),
        bad_bands=spec.bad_bands,
        data_root=spec.cosmic_dir,
        shift_root=spec.shift_dir,
        init_root=spec.init_dir,
        init_test_root=spec.init_test_dir,
        fig_root=spec.fig_dir,
        supplement_root=spec.supplement_path or spec.dataset_dir / "extra_data",
        metadata_dir=spec.metadata_dir,
        state_file=pipeline_state_path_for_spec(spec),
        output_root=root / "output",
        bundle_root=root / "output" / "bundle",
        classification_workbook=spec.workbook_path,
        dataset_mapping_file=spec.metadata_dir / MAPPING_FILE_NAME,
    )
