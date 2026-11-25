"""把人工补充数据按文件夹发布到全量 init。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ramanv2.data.arc_data import (
    count_arc_data,
    is_arc_data_name,
    list_arc_data_paths,
    read_arc_data,
)
from ramanv2.data.naming import CANONICAL_FOLDER_PATTERN, parse_folder_prefix
from ramanv2.common.paths import build_natural_key
from ramanv2.pipeline.init_builder import build_full_init
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.data.hierarchy_spec import HierarchySpec
from ramanv2.data.datasets import DatasetSpec, annotate_dataset_build
from ramanv2.pipeline.state import PipelineState, write_pipeline_state


@dataclass(frozen=True)
class SupplementGroup:
    """一个待写入 init 的补充文件夹。"""

    folder_name: str
    prefix: str
    genus: str
    source_dir: Path
    target_relative: Path
    file_count: int


@dataclass(frozen=True)
class SupplementPlan:
    """补充数据预检结果；errors 非空时禁止执行全量重建。"""

    groups: tuple[SupplementGroup, ...]
    errors: tuple[str, ...]
    unbacked_init_folders: tuple[str, ...]

    @property
    def file_count(self) -> int:
        return sum(group.file_count for group in self.groups)


def scan_supplement_plan(
    supplement_root: Path | str,
    init_root: Path | str,
    mapping: DatasetMapping,
    hierarchy: HierarchySpec | None = None,
) -> SupplementPlan:
    """扫描补充数据目录并核对现有 init 中未被覆盖的直放组，不修改任何文件。"""
    root = Path(supplement_root)
    init = Path(init_root)
    prefix_to_genus = {
        str(prefix).upper(): str(genus)
        for genus, prefixes in mapping.genera.items()
        for prefix in prefixes
    }
    groups: list[SupplementGroup] = []
    errors: list[str] = []
    if root.is_dir():
        for folder in sorted(
            (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
            key=lambda path: build_natural_key(path.name),
        ):
            # 预检只需要计数：单次 scandir 同时统计直接光谱与嵌套光谱，避免排序和二次遍历。
            direct_count = 0
            nested_count = 0
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_dir():
                        nested_count += count_arc_data(Path(entry.path), recursive=True)
                    elif entry.is_file() and is_arc_data_name(entry.name):
                        direct_count += 1
            if nested_count:
                errors.append(f"{folder.name}：补充文件夹只支持单层光谱，发现 {nested_count} 条嵌套文件")
                continue
            if not direct_count:
                errors.append(f"{folder.name}：补充文件夹没有 .arc_data 光谱")
                continue
            prefix = parse_folder_prefix(folder.name, uppercase_enable=True)
            genus = prefix_to_genus.get(prefix)
            if genus is None:
                errors.append(f"{folder.name}：种前缀 {prefix} 未映射到属")
                continue
            groups.append(
                SupplementGroup(
                    folder_name=folder.name,
                    prefix=prefix,
                    genus=genus,
                    source_dir=folder,
                    target_relative=(
                        hierarchy.init_relative_dir(folder.name) / folder.name
                        if hierarchy is not None
                        else Path(genus) / folder.name
                    ),
                    file_count=direct_count,
                )
            )
    planned_targets = {group.target_relative.as_posix() for group in groups}
    unbacked: list[str] = []
    if init.is_dir():
        parent_depth = max(hierarchy.directory_depth, 1) if hierarchy is not None else 1
        for directory, children, _ in os.walk(init, followlinks=hierarchy is None):
            parent = Path(directory).relative_to(init)
            if len(parent.parts) != parent_depth:
                continue
            # 只核对来源组名称，走到组所在层就停止遍历。
            for name in children:
                if not CANONICAL_FOLDER_PATTERN.fullmatch(name):
                    relative = (parent / name).as_posix()
                    if relative not in planned_targets:
                        unbacked.append(relative)
            children.clear()
    if unbacked and not root.is_dir():
        errors.append(f"补充数据目录不存在，且 init 中有无来源直放组：{'、'.join(unbacked)}")
    return SupplementPlan(
        groups=tuple(groups),
        errors=tuple(errors),
        unbacked_init_folders=tuple(sorted(unbacked, key=build_natural_key)),
    )


def validate_supplement_plan(plan: SupplementPlan) -> tuple[str, ...]:
    """读取补充光谱并校验两列数据，供重建前预检。"""
    errors = list(plan.errors)
    for group in plan.groups:
        try:
            for path in list_arc_data_paths(
                group.source_dir,
                key=lambda item: build_natural_key(item.name),
            ):
                wavenumbers, intensities = read_arc_data(path)
                if not wavenumbers.size or wavenumbers.size != intensities.size:
                    errors.append(f"{group.folder_name}/{path.name}：无效光谱数据")
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{group.folder_name}：{error}")
    return tuple(errors)


def prepare_supplement_plan(
    supplement_root: Path | str | None,
    init_root: Path | str,
    mapping: DatasetMapping | None,
    hierarchy: HierarchySpec,
) -> SupplementPlan:
    """按层级准备补充数据计划；扁平数据集返回空计划。"""
    if hierarchy.is_flat:
        return SupplementPlan(groups=(), errors=(), unbacked_init_folders=())
    if supplement_root is None:
        raise ValueError("缺少补充数据目录")
    if mapping is None:
        raise ValueError("缺少数据集前缀映射")
    return scan_supplement_plan(supplement_root, init_root, mapping, hierarchy=hierarchy)


def run_full_init(
    spec: DatasetSpec,
    state: PipelineState,
    state_path: Path,
    mapping: DatasetMapping | None,
    *,
    shift_root: Path | str | None = None,
    init_root: Path | str | None = None,
    selected_test_folders: Sequence[str] | None = None,
    plan: SupplementPlan | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """按当前门禁构建全量 init，并在同一次发布中登记构建报告。"""
    target = Path(init_root) if init_root is not None else spec.init_dir
    supplement_groups = None
    if not spec.hierarchy.is_flat:
        plan = plan or prepare_supplement_plan(spec.supplement_path, target, mapping, spec.hierarchy)
        errors = validate_supplement_plan(plan)
        if errors:
            raise ValueError("补充数据校验失败：" + "；".join(errors))
        supplement_groups = tuple((group.source_dir, group.target_relative) for group in plan.groups)

    working = PipelineState.from_payload(state.to_payload())

    def record_report(report: dict[str, object]) -> None:
        working.dataset_builds["full_init"] = annotate_dataset_build(report, spec)
        write_pipeline_state(state_path, working)

    report = build_full_init(
        shift_root if shift_root is not None else spec.shift_dir,
        target,
        mapping,
        state.affine_completed_dates,
        test_root=spec.init_test_dir,
        selected_test_folders=tuple(selected_test_folders) if selected_test_folders is not None else None,
        hierarchy=spec.hierarchy,
        progress_callback=progress_callback,
        supplement_groups=supplement_groups,
        record_report=record_report,
        state_path=state_path,
    )
    state.dataset_builds = working.dataset_builds
    return report
