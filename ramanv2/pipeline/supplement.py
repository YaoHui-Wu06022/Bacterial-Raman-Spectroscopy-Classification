"""把人工补充数据按文件夹发布到全量 init。

补充数据目录约定为 ``补充数据/<文件夹>/*.arc_data``，文件夹名前缀通过
``profile_catalog`` 映射到属；全量 init 重建完成后，这些文件夹会被原样写入
``dataset/init/<属>/<文件夹>``。该类数据没有 cosmic/shift 来源，不能由
``build_full_init`` 从 shift_data 生成，因此必须单独保留。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.naming import build_natural_key, parse_folder_prefix
from ramanv2.data.builders.init import CANONICAL_FOLDER_PATTERN, build_spectrum_folder
from ramanv2.data.catalog import DatasetMapping


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
            files = sorted(
                (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".arc_data"),
                key=lambda path: build_natural_key(path.name),
            )
            nested = [path for path in folder.rglob("*.arc_data") if path.parent != folder]
            if nested:
                errors.append(f"{folder.name}：补充文件夹只支持单层光谱，发现 {len(nested)} 条嵌套文件")
                continue
            if not files:
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
                    target_relative=Path(genus) / folder.name,
                    file_count=len(files),
                )
            )
    planned_targets = {group.target_relative.as_posix() for group in groups}
    unbacked: list[str] = []
    if init.is_dir():
        for genus_dir in sorted((path for path in init.iterdir() if path.is_dir()), key=lambda path: build_natural_key(path.name)):
            for folder in sorted((path for path in genus_dir.iterdir() if path.is_dir()), key=lambda path: build_natural_key(path.name)):
                if CANONICAL_FOLDER_PATTERN.fullmatch(folder.name):
                    continue
                relative = f"{genus_dir.name}/{folder.name}"
                if relative not in planned_targets:
                    unbacked.append(relative)
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
            for path in sorted(
                (item for item in group.source_dir.iterdir() if item.is_file() and item.suffix.lower() == ".arc_data"),
                key=lambda item: build_natural_key(item.name),
            ):
                wavenumbers, intensities = read_arc_data(path)
                if not wavenumbers.size or wavenumbers.size != intensities.size:
                    errors.append(f"{group.folder_name}/{path.name}：无效光谱数据")
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{group.folder_name}：{error}")
    return tuple(errors)


def apply_supplement_plan(init_root: Path | str, plan: SupplementPlan) -> dict[str, object]:
    """把补充文件夹写入 init；逐组原子发布，失败时源文件保持不变。"""
    if plan.errors:
        raise ValueError("补充数据计划有错误：" + "；".join(plan.errors))
    targets: list[str] = []
    file_count = 0
    for group in plan.groups:
        target = Path(init_root) / group.target_relative
        file_count += build_spectrum_folder(group.source_dir, target)
        targets.append(group.target_relative.as_posix())
    return {
        "status": "ready",
        "folder_count": len(plan.groups),
        "file_count": file_count,
        "targets": targets,
    }


__all__ = [
    "SupplementGroup",
    "SupplementPlan",
    "apply_supplement_plan",
    "scan_supplement_plan",
    "validate_supplement_plan",
]
