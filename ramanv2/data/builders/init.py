"""从仿射校正并完成审核的数据构建 init 数据集。"""

from __future__ import annotations

import re
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np

from ramanv2.common.arc_data import read_arc_data, write_arc_data
from ramanv2.common.naming import build_natural_key, is_test_source_folder, parse_test_folder_prefix
from ramanv2.common.publishing import create_temporary_path, publish_directory, publish_file, rename_directory
from ramanv2.data.count import summarize_dataset
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.lineage import folder_compensation, initialize_folder_lineage, load_folder_lineage, record_output_path


SPECIAL_NAMES = {"小球", "shift", "delete", "audit_runs", ".raman_pipeline"}
CANONICAL_FOLDER_PATTERN = re.compile(r"^([A-Za-z]+)(\d{8})_(\d+)$")
DATE_PATTERN = re.compile(r"^\d{8}$")
BuildMode = Literal["full", "incremental"]


@dataclass(frozen=True)
class InitBuildReport:
    """保留给调用方的构建报告类型。"""

    source_dates: tuple[str, ...]
    skipped_dates: tuple[str, ...]
    source_files: int
    output_files: int
    generated_profiles: tuple[str, ...]


@dataclass(frozen=True)
class _SourceGroup:
    date_name: str
    source_dir: Path
    genus: str
    target_name: str
    files: tuple[Path, ...]
    is_test: bool = False
    source_name: str = ""


class InitBuildConflictError(FileExistsError):
    """增量构建发现已有日期或目标目录时抛出的冲突错误。"""

    def __init__(self, conflicts: list[str]) -> None:
        self.conflicts = tuple(conflicts)
        super().__init__("增量构建目标已存在：" + "; ".join(conflicts))


def build_full_init(
    shift_root: Path | str,
    init_root: Path | str,
    mapping: DatasetMapping,
    completed_dates: list[str] | set[str] | tuple[str, ...],
    test_root: Path | str | None = None,
    selected_cs_folders: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """从通过门禁的 shift 日期构建普通 init 与选定的独立 CSdata。"""
    cosmic_root = Path(shift_root).resolve().parent / "cosmic_data"
    if cosmic_root.is_dir():
        initialize_folder_lineage(cosmic_root, mapping)
    resolved_test_root = (
        Path(test_root).resolve()
        if test_root is not None
        else Path(init_root).resolve().parent / "CSdata"
    )
    report = build_init_datasets(
        shift_root,
        init_root,
        {},
        mapping,
        completed_dates,
        test_root=resolved_test_root,
        selected_test_folders=selected_cs_folders,
    )
    report["generated_profiles"] = []
    report["profile_stats"] = {}
    report["count"] = summarize_dataset(Path(init_root).resolve())
    return report


def build_profile_dataset(
    init_root: Path | str,
    profile_root: Path | str,
    profile_name: str,
    catalog: DatasetMapping,
) -> dict[str, object]:
    """从全量 init 原子构建一个固定 Profile，并返回 count 统计。"""
    if profile_name not in catalog.datasets:
        raise ValueError(f"未知 Profile：{profile_name}")
    mapping = catalog.subset(profile_name)
    report = build_profile_datasets(
        init_root,
        {profile_name: profile_root},
        mapping,
    )
    report["profile"] = profile_name
    report["count"] = summarize_dataset(Path(profile_root).resolve())
    return report


def update_compensated_init(
    shift_root: Path | str,
    date_name: str,
    folder_names: list[str] | tuple[str, ...] | set[str],
    init_root: Path | str,
    profile_roots: dict[str, Path | str],
    mapping: DatasetMapping,
    test_root: Path | str | None = None,
) -> dict[str, object]:
    """只替换文件夹补偿涉及的 init、Profile init 和 CSdata 目录。

    shift_data 已经完成日期级校正；这里按相同的前缀和日期序号定位目标目录，
    只发布选中文件夹对应的目标目录，不重新构建其他日期或文件夹。
    """
    source_root = Path(shift_root).resolve()
    cosmic_root = source_root.parent / "cosmic_data"
    if cosmic_root.is_dir() and not load_folder_lineage(cosmic_root).get("folders"):
        initialize_folder_lineage(cosmic_root)
    source_date = source_root / str(date_name)
    if not source_date.is_dir():
        raise FileNotFoundError(f"shift 日期目录不存在：{source_date}")
    if not _is_manual_approved(source_date / "shift" / "affine.json"):
        raise ValueError(f"日期尚未完成仿射校准：{date_name}")

    selected = {
        str(name)
        for name in (folder_names.keys() if isinstance(folder_names, dict) else folder_names)
    }
    if not selected:
        return {"status": "ready", "date": str(date_name), "updated_targets": [], "skipped_targets": []}
    prefix_to_genus = {
        prefix: genus
        for genus, prefixes in mapping.genera.items()
        for prefix in prefixes
    }
    groups = [
        group
        for group in _collect_source_groups(
            source_date,
            prefix_to_genus,
            selected,
            source_folder_names=selected,
        )
        if group.source_name in selected
    ]
    missing = sorted(selected - {group.source_name for group in groups}, key=build_natural_key)
    if missing:
        raise ValueError(f"shift_data 中不存在补偿文件夹：{missing}")

    destinations: dict[str, Path] = {"init": Path(init_root).resolve()}
    destinations.update({str(name): Path(path).resolve() for name, path in profile_roots.items()})
    if test_root is not None:
        destinations["CSdata"] = Path(test_root).resolve()

    staged: list[tuple[Path, Path, str, _SourceGroup]] = []
    skipped: list[dict[str, str]] = []
    try:
        for group in groups:
            relative = _target_relative(group)
            for label, root in destinations.items():
                target = root / relative
                if not target.is_dir():
                    skipped.append({"root": label, "path": relative.as_posix(), "reason": "target_missing"})
                    continue
                temporary = create_temporary_path(target)
                temporary.mkdir(parents=True, exist_ok=False)
                try:
                    for source in group.files:
                        _validate_spectrum(source)
                        _copy_source_file(
                            source,
                            temporary / source.name,
                            _source_folder_compensation(source_root, date_name, group.source_name),
                        )
                    staged.append((temporary, target, label, group))
                except Exception:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise

        backups: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for temporary, target, _, _ in staged:
                backup = create_temporary_path(target)
                rename_directory(target, backup)
                backups.append((target, backup))
                rename_directory(temporary, target)
                published.append(target)
        except Exception:
            for target in reversed(published):
                shutil.rmtree(target, ignore_errors=True)
            for target, backup in reversed(backups):
                if backup.exists() and not target.exists():
                    rename_directory(backup, target)
            raise
        finally:
            for _, backup in backups:
                shutil.rmtree(backup, ignore_errors=True)
            for temporary, _, _, _ in staged:
                shutil.rmtree(temporary, ignore_errors=True)

        updated_targets = [
            {
                "root": label,
                "path": target.relative_to(destinations[label]).as_posix(),
                "file_count": len(group.files),
            }
            for _, target, label, group in staged
        ]
        return {
            "status": "ready",
            "date": str(date_name),
            "folder_names": sorted(selected, key=build_natural_key),
            "updated_targets": updated_targets,
            "skipped_targets": skipped,
            "updated_file_count": sum(int(item["file_count"]) for item in updated_targets),
        }
    except Exception:
        for temporary, _, _, _ in staged:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_profile_datasets(
    init_root: Path | str,
    profile_roots: dict[str, Path | str],
    mapping: DatasetMapping,
) -> dict[str, object]:
    """从全量 init 按属名单原样构建各 profile init 目录。"""
    source_root = Path(init_root).resolve()
    roots = {name: Path(path).resolve() for name, path in profile_roots.items()}
    if not source_root.is_dir():
        raise FileNotFoundError(f"全量 init 不存在：{source_root}")
    if set(roots) != set(mapping.datasets):
        raise ValueError("profile 输出目录必须与名单中的 datasets 完全一致")
    if any(root == source_root or source_root in root.parents for root in roots.values()):
        raise ValueError("profile 输出目录不能位于全量 init 内部")
    if len(set(roots.values())) != len(roots):
        raise ValueError("profile 输出目录不能重复")
    for root in roots.values():
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"profile 输出目录不是目录：{root}")

    source_genera = {path.name: path for path in source_root.iterdir() if path.is_dir()}
    temporary_roots: dict[str, Path] = {}
    profiles: dict[str, dict[str, object]] = {}
    try:
        for profile_name, genera in mapping.datasets.items():
            temporary = create_temporary_path(roots[profile_name])
            temporary.mkdir(parents=True, exist_ok=False)
            temporary_roots[profile_name] = temporary
            source_files = 0
            source_folders: set[str] = set()
            for genus in genera:
                genus_dir = source_genera.get(genus)
                if genus_dir is None:
                    raise ValueError(f"profile {profile_name} 引用了不存在的属目录：{genus}")
                for source_file in sorted(genus_dir.rglob("*.arc_data"), key=lambda path: build_natural_key(path.as_posix())):
                    if not source_file.is_file():
                        continue
                    _validate_spectrum(source_file)
                    relative = source_file.relative_to(source_root)
                    target = temporary / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target)
                    source_axis, source_intensity = read_arc_data(source_file)
                    target_axis, target_intensity = read_arc_data(target)
                    if not np.array_equal(source_axis, target_axis) or not np.array_equal(source_intensity, target_intensity):
                        raise ValueError(f"复制后光谱校验失败：{source_file}")
                    source_files += 1
                    source_folders.add(relative.parent.as_posix())
            if source_files == 0:
                raise RuntimeError(f"profile {profile_name} 没有可用光谱")
            profiles[profile_name] = {
                "genera": list(genera),
                "genus_count": len(genera),
                "folder_count": len(source_folders),
                "source_files": source_files,
                "output_files": source_files,
                "output_root": str(roots[profile_name]),
            }
        _publish_all(temporary_roots, roots)
    except Exception:
        for temporary in temporary_roots.values():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "ready",
        "source_root": str(source_root),
        "profiles": profiles,
        "profile_count": len(profiles),
        "source_files": sum(int(item["source_files"]) for item in profiles.values()),
    }


def build_init_datasets(
    shift_root: Path | str,
    init_root: Path | str,
    profile_roots: dict[str, Path | str],
    mapping: DatasetMapping,
    audit_completed_dates: list[str] | set[str] | tuple[str, ...],
    mode: BuildMode = "full",
    date_names: list[str] | set[str] | tuple[str, ...] | None = None,
    test_root: Path | str | None = None,
    selected_test_folders: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """按批准日期原子构建全量 init 或增量追加 init。"""
    if mode not in {"full", "incremental"}:
        raise ValueError("mode 必须是 full 或 incremental")
    source_root = Path(shift_root).resolve()
    full_root = Path(init_root).resolve()
    roots = {name: Path(path).resolve() for name, path in profile_roots.items()}
    resolved_test_root = Path(test_root).resolve() if test_root is not None else None
    _validate_output_roots(full_root, roots, resolved_test_root)
    unknown_profiles = set(roots) - set(mapping.datasets)
    if unknown_profiles:
        raise ValueError(f"profile 没有对应名单：{sorted(unknown_profiles)}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"shift_data 不存在：{source_root}")

    prefix_to_genus = {
        prefix: genus
        for genus, prefixes in mapping.genera.items()
        for prefix in prefixes
    }
    selected_names = _select_date_names(source_root, date_names)
    audit_dates = {str(name) for name in audit_completed_dates}
    eligible_dates: list[str] = []
    skipped_dates: list[str] = []
    date_status: dict[str, dict[str, str]] = {}
    for date_name in selected_names:
        date_dir = source_root / date_name
        if not date_dir.is_dir():
            skipped_dates.append(date_name)
            date_status[date_name] = {"status": "skipped", "reason": "missing_source_date"}
        elif date_name not in audit_dates:
            skipped_dates.append(date_name)
            date_status[date_name] = {"status": "skipped", "reason": "audit_not_completed"}
        elif not _is_manual_approved(date_dir / "shift" / "affine.json"):
            skipped_dates.append(date_name)
            date_status[date_name] = {"status": "skipped", "reason": "affine_not_approved"}
        else:
            eligible_dates.append(date_name)
            date_status[date_name] = {"status": "eligible"}

    groups: list[_SourceGroup] = []
    selected_keys = set() if selected_test_folders is None else {
        str(item).replace("\\", "/") for item in selected_test_folders
    }
    for date_name in eligible_dates:
        groups.extend(
            _collect_source_groups(
                source_root / date_name,
                prefix_to_genus,
                selected_keys,
                include_unknown_tests=resolved_test_root is not None,
            )
        )
    cosmic_root = source_root.parent / "cosmic_data"
    if cosmic_root.is_dir():
        lineage = load_folder_lineage(cosmic_root)
        records = lineage.get("folders", {})
        if isinstance(records, dict):
            filtered_groups: list[_SourceGroup] = []
            for group in groups:
                record = records.get(f"{group.date_name}/{group.source_name or group.source_dir.name}")
                if not isinstance(record, dict) or record.get("folder_status") != "folder_deleted":
                    file_records = record.get("files", {}) if isinstance(record, dict) else {}
                    if isinstance(file_records, dict):
                        files = tuple(
                            source
                            for source in group.files
                            if not isinstance(file_records.get(source.name), dict)
                            or file_records.get(source.name, {}).get("status") == "active"
                        )
                        if files:
                            group = replace(group, files=files)
                        else:
                            continue
                    filtered_groups.append(group)
            groups = filtered_groups
    destination_roots = {"__all__": full_root, **roots}
    if resolved_test_root is not None:
        destination_roots["__test__"] = resolved_test_root
    _validate_test_group_targets(groups)
    if mode == "incremental":
        conflicts = _find_incremental_conflicts(eligible_dates, groups, destination_roots)
        if conflicts:
            raise InitBuildConflictError(conflicts)

    temporary_roots = _create_temporary_roots(destination_roots, mode)
    source_files = 0
    output_files = 0
    date_stats: dict[str, dict[str, int]] = {
        date_name: {"source_files": 0, "output_files": 0, "folder_count": 0, "test_files": 0}
        for date_name in eligible_dates
    }
    profile_stats = {name: 0 for name in roots}
    test_files = 0
    test_folders: set[str] = set()
    cs_exports: list[dict[str, object]] = []
    try:
        for group in groups:
            date_stats[group.date_name]["folder_count"] += 1
            destinations = _resolve_destinations(group, mapping, temporary_roots)
            for source_path in group.files:
                source_files += 1
                date_stats[group.date_name]["source_files"] += 1
                for profile_name, destination in destinations:
                    target = destination / source_path.name
                    _copy_source_file(
                        source_path,
                        target,
                        _source_folder_compensation(source_root, group.date_name, group.source_name),
                    )
                    output_files += 1
                    date_stats[group.date_name]["output_files"] += 1
                    if profile_name != "__all__":
                        if profile_name == "__test__":
                            test_files += 1
                            test_folders.add(group.target_name)
                            date_stats[group.date_name]["test_files"] += 1
                        else:
                            profile_stats[profile_name] += 1
            if group.is_test:
                cs_exports.append(
                    {
                        "source": f"{group.date_name}/{group.source_name or group.source_dir.name}",
                        "source_name": group.source_name or group.source_dir.name,
                        "target": group.target_name,
                        "genus": group.genus,
                        "date": group.date_name,
                        "file_count": len(group.files),
                    }
                )
        report = {
            "status": "ready",
            "mode": mode,
            "source_dates": eligible_dates,
            "eligible_dates": eligible_dates,
            "skipped_dates": skipped_dates,
            "date_status": date_status,
            "conflict_dates": [],
            "source_files": source_files,
            "output_files": output_files,
            "init_source_files": source_files - test_files,
            "init_output_files": output_files - test_files,
            "date_stats": date_stats,
            "profile_stats": profile_stats,
            "generated_profiles": sorted(roots),
            "init_root": str(full_root),
            "test_root": str(resolved_test_root) if resolved_test_root is not None else None,
            "test_files": test_files,
            "test_folders": sorted(test_folders, key=build_natural_key),
            "cs_folder_count": len(test_folders),
            "test_output_files": test_files,
            "cs_source_files": test_files,
            "cs_output_files": test_files,
            "cs_exports": cs_exports,
            "selected_cs_folders": sorted(selected_keys or set(), key=build_natural_key),
        }
        _publish_all(temporary_roots, destination_roots)
        cosmic_root = source_root.parent / "cosmic_data"
        if cosmic_root.is_dir():
            for group in groups:
                relative = _target_relative(group)
                if group.is_test:
                    record_output_path(
                        cosmic_root,
                        group.date_name,
                        group.source_name or group.source_dir.name,
                        csdata_path=str(resolved_test_root / relative) if resolved_test_root is not None else None,
                    )
                    continue
                profile_paths = {
                    name: str(root / relative)
                    for name, root in roots.items()
                    if group.genus in mapping.datasets.get(name, ())
                }
                record_output_path(
                    cosmic_root,
                    group.date_name,
                    group.source_name or group.source_dir.name,
                    init_path=str(full_root / relative),
                    profile_paths=profile_paths,
                )
        return report
    except Exception:
        for temporary in temporary_roots.values():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _select_date_names(source_root: Path, date_names: object) -> list[str]:
    if date_names is None:
        names = [path.name for path in source_root.iterdir() if path.is_dir() and DATE_PATTERN.fullmatch(path.name)]
    else:
        names = [str(name) for name in date_names]
    return sorted(set(names), key=build_natural_key)


def _collect_source_groups(
    date_dir: Path,
    prefix_to_genus: dict[str, str],
    selected_test_folders: set[str],
    *,
    include_unknown_tests: bool = False,
    source_folder_names: set[str] | None = None,
) -> list[_SourceGroup]:
    by_prefix: dict[str, list[tuple[Path, tuple[Path, ...]]]] = {}
    groups: list[_SourceGroup] = []
    for source_dir in sorted(date_dir.iterdir(), key=lambda path: build_natural_key(path.name)):
        if not source_dir.is_dir() or source_dir.name in SPECIAL_NAMES:
            continue
        if source_folder_names is not None and source_dir.name not in source_folder_names:
            continue
        files = tuple(
            sorted(
                (path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".arc_data"),
                key=lambda path: build_natural_key(path.name),
            )
        )
        if not files:
            continue
        source_key = f"{date_dir.name}/{source_dir.name}"
        if is_test_source_folder(source_dir.name):
            prefix = parse_test_folder_prefix(source_dir.name)
            genus = prefix_to_genus.get(prefix)
            is_unknown_test = genus is None
            is_selected_test = (include_unknown_tests and is_unknown_test) or source_key in selected_test_folders
            if is_selected_test:
                for source_path in files:
                    _validate_spectrum(source_path)
                groups.append(
                    _SourceGroup(
                        date_name=date_dir.name,
                        source_dir=source_dir,
                        genus=genus or "",
                        target_name=source_dir.name,
                        files=files,
                        is_test=True,
                        source_name=source_dir.name,
                    )
                )
                continue
            if is_unknown_test:
                # 未映射的 CS 没有普通属目录，只能留在默认的 CSdata 流程中。
                continue
            # 已映射但未选择进入 CSdata 的 CS，继续按普通来源目录处理。
        prefix = _source_prefix(source_dir.name)
        if prefix not in prefix_to_genus:
            raise ValueError(f"名单未映射来源前缀：{date_dir.name}/{source_dir.name} ({prefix})")
        for source_path in files:
            _validate_spectrum(source_path)
        by_prefix.setdefault(prefix, []).append((source_dir, files))

    for prefix in sorted(by_prefix, key=build_natural_key):
        genus = prefix_to_genus[prefix]
        for index, (source_dir, files) in enumerate(by_prefix[prefix], start=1):
            groups.append(
                _SourceGroup(
                    date_name=date_dir.name,
                    source_dir=source_dir,
                    genus=genus,
                    target_name=f"{prefix}{date_dir.name}_{index:02d}",
                    files=files,
                    source_name=source_dir.name,
                )
            )
    return groups


def _resolve_destinations(
    group: _SourceGroup,
    mapping: DatasetMapping,
    temporary_roots: dict[str, Path],
) -> list[tuple[str, Path]]:
    if group.is_test:
        if "__test__" not in temporary_roots:
            raise ValueError("CS 测试目录需要配置 test_root")
        return [("__test__", temporary_roots["__test__"] / group.target_name)]
    target_relative = Path(group.genus) / group.target_name
    destinations = [("__all__", temporary_roots["__all__"] / target_relative)]
    for profile_name, genera in mapping.datasets.items():
        if profile_name in temporary_roots and group.genus in genera:
            destinations.append((profile_name, temporary_roots[profile_name] / target_relative))
    return destinations


def _validate_output_roots(
    full_root: Path,
    profile_roots: dict[str, Path],
    test_root: Path | None = None,
) -> None:
    roots = [full_root, *profile_roots.values()]
    if test_root is not None:
        roots.append(test_root)
    if len(set(roots)) != len(roots):
        raise ValueError("init、profile 和 CSdata 输出目录不能重复")
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root in other.parents or other in root.parents:
                raise ValueError("输出目录不能互相嵌套")
    for root in roots:
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"输出目录不是目录：{root}")


def _find_incremental_conflicts(
    eligible_dates: list[str],
    groups: list[_SourceGroup],
    destination_roots: dict[str, Path],
) -> list[str]:
    conflicts: set[str] = set()
    eligible = set(eligible_dates)
    for name, root in destination_roots.items():
        if name == "__test__":
            continue
        if not root.is_dir():
            continue
        for genus_dir in root.iterdir():
            if not genus_dir.is_dir():
                continue
            for target_dir in genus_dir.iterdir():
                if not target_dir.is_dir():
                    continue
                match = CANONICAL_FOLDER_PATTERN.fullmatch(target_dir.name)
                if match is not None and match.group(2) in eligible:
                    conflicts.add(f"{name}:{target_dir.relative_to(root).as_posix()}")
    for group in groups:
        relative = _target_relative(group)
        for name, root in destination_roots.items():
            if group.is_test != (name == "__test__"):
                continue
            if (root / relative).exists():
                conflicts.add(f"{name}:{relative.as_posix()}")
    return sorted(conflicts)


def _target_relative(group: _SourceGroup) -> Path:
    """返回组在对应输出根下的相对目录。"""
    if group.is_test:
        return Path(group.target_name)
    return Path(group.genus) / group.target_name


def _validate_test_group_targets(groups: list[_SourceGroup]) -> None:
    """防止不同日期的同名 CS 目录在一次全量发布中互相覆盖。"""
    targets: dict[Path, _SourceGroup] = {}
    for group in groups:
        if not group.is_test:
            continue
        target = _target_relative(group)
        previous = targets.get(target)
        if previous is not None and previous.date_name != group.date_name:
            raise InitBuildConflictError(
                [
                    f"__test__:{target.as_posix()} ({previous.date_name}, {group.date_name})"
                ]
            )
        targets[target] = group


def _create_temporary_roots(destination_roots: dict[str, Path], mode: BuildMode) -> dict[str, Path]:
    temporary_roots: dict[str, Path] = {}
    try:
        for name, root in destination_roots.items():
            temporary = create_temporary_path(root)
            temporary.mkdir(parents=True, exist_ok=False)
            temporary_roots[name] = temporary
            if mode == "incremental" and root.exists():
                if not root.is_dir():
                    raise NotADirectoryError(f"输出目录不是目录：{root}")
                shutil.copytree(root, temporary, dirs_exist_ok=True)
    except Exception:
        for temporary in temporary_roots.values():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return temporary_roots


def _copy_source_file(source: Path, target: Path, wavenumber_offset: float = 0.0) -> None:
    """验证源谱后复制；仅在 init/CSdata 发布阶段叠加文件夹补偿。"""
    wavenumbers, intensities = read_arc_data(source)
    if not wavenumbers.size or wavenumbers.size != intensities.size:
        raise ValueError(f"无效光谱：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if float(wavenumber_offset) == 0.0:
        shutil.copy2(source, target)
    else:
        write_arc_data(target, wavenumbers + float(wavenumber_offset), intensities)


def _validate_spectrum(path: Path) -> None:
    wavenumbers, intensities = read_arc_data(path)
    if not wavenumbers.size or wavenumbers.size != intensities.size:
        raise ValueError(f"无效光谱：{path}")


def _source_folder_compensation(shift_root: Path, date_name: str, folder_name: str) -> float:
    """读取 cosmic 源头记录中的绝对补偿；测试目录没有源头记录时返回零。"""
    cosmic_root = Path(shift_root).resolve().parent / "cosmic_data"
    if not cosmic_root.is_dir():
        return 0.0
    return folder_compensation(cosmic_root, str(date_name), str(folder_name))


def _is_manual_approved(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "manual_approved"


def _publish_all(
    temporary_roots: dict[str, Path],
    destination_roots: dict[str, Path],
) -> None:
    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for name, target in destination_roots.items():
            temporary = temporary_roots[name]
            if target.exists():
                backup = target.parent / f".{target.name}_previous_{uuid4().hex}"
                rename_directory(target, backup)
                backups.append((target, backup))
            rename_directory(temporary, target)
            published.append((target, temporary))
        for _, backup in backups:
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        for target, temporary in reversed(published):
            if target.exists() and not temporary.exists():
                rename_directory(target, temporary)
        for target, backup in reversed(backups):
            if backup.exists() and not target.exists():
                rename_directory(backup, target)
        raise


def _source_prefix(name: str) -> str:
    if is_test_source_folder(name):
        return parse_test_folder_prefix(name)
    match = CANONICAL_FOLDER_PATTERN.fullmatch(name)
    if match is not None:
        return match.group(1).upper()
    match = re.match(r"([A-Za-z]+)", name)
    if match is None:
        raise ValueError(f"无法解析来源文件夹前缀：{name}")
    return match.group(1).upper()




__all__ = [
    "InitBuildConflictError",
    "InitBuildReport",
    "build_full_init",
    "build_profile_dataset",
    "build_init_datasets",
    "build_profile_datasets",
    "update_compensated_init",
]
