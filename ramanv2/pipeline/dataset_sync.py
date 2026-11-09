"""沿用源文件夹走向，从 shift 最小更新对应的 init 或 init_test 组。"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from ramanv2.common.arc_data import (
    count_arc_data,
    has_arc_data,
    is_arc_data_name,
    iter_arc_data_paths,
)
from ramanv2.common.naming import (
    CANONICAL_FOLDER_PATTERN,
    SPECIAL_FOLDER_NAMES,
    build_natural_key,
    is_date_name,
    parse_folder_prefix,
)
from ramanv2.common.file_ops import preserve_paths, publish_file, rename_directory, _remove_path
from ramanv2.data.builders.init import build_init_datasets
from ramanv2.data.hierarchy_config import HierarchySpec
from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.data.count import summarize_dataset
from ramanv2.data.plots.prefix import build_prefix_plots, build_prefix_plots_for_groups
from ramanv2.data.profiles import TEST_ROOT_NAME
from ramanv2.records.lineage import lineage_path, load_folder_lineage, write_folder_lineage
from ramanv2.pipeline.calibration import load_date_affine_parameters
from ramanv2.pipeline.state import PipelineState, write_pipeline_state


def collect_test_return_sources(init_root: Path, state: PipelineState) -> list[str]:
    """只列出仍有实际 init_test 光谱且有来源记录的目录，不扫描全量 init。"""
    report = state.dataset_builds.get("full_init")
    if not isinstance(report, dict):
        return []
    root = Path(report.get("test_root") or init_root.parent / TEST_ROOT_NAME).resolve()
    sources = set()
    for entry in report.get("test_exports", []):
        target = root / str(entry["target"])
        if target.parent == root and has_arc_data(target):
            sources.add(entry["source"])
    return sorted(sources, key=build_natural_key)


def run_test_to_init(
    data_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
) -> dict[str, object]:
    """直接迁移已有 init_test 单组到 init，复用双向迁移事务。"""
    return _run_dataset_transfer(data_root, init_root, mapping, state, state_file, source_id, False)


def collect_init_export_sources(data_root: Path, init_root: Path, state: PipelineState) -> dict[str, str]:
    """列出有来源记录的现有 init 组，保留日期/源目录作为操作主键。"""
    report = state.dataset_builds.get("full_init")
    if not isinstance(report, dict):
        return {}
    exports = {entry["source"] for entry in report.get("test_exports", [])}
    records = load_folder_lineage(data_root)["folders"]
    init_key = os.path.normcase(str(init_root.resolve()))
    sources = {}
    for source_id, record in records.items():
        value = record.get("outputs", {}).get("init")
        if source_id in exports or not value:
            continue
        # lineage 存绝对路径；用字符串比较父目录，避免为每个组做一次 resolve 磁盘查询。
        path = Path(value)
        if os.path.normcase(str(path.parent.parent)) != init_key or not has_arc_data(path):
            continue
        sources[source_id] = f"{path.parent.name}/{path.name}"
    return dict(sorted(sources.items(), key=lambda item: build_natural_key(item[1])))


def run_init_to_test(
    data_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
) -> dict[str, object]:
    """将所选 init 整夹移入 init_test，不重建其余数据。"""
    return _run_dataset_transfer(data_root, init_root, mapping, state, state_file, source_id, True)


def _run_dataset_transfer(
    data_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
    to_test_enable: bool,
) -> dict[str, object]:
    """双向共用预检查、迁移、绘图及状态发布；任一失败恢复原文件和路由。"""
    parts = source_id.split("/")
    if len(parts) != 2 or len(parts[0]) != 8 or not parts[0].isascii() or not parts[0].isdigit() or not parts[1] or parts[1] in {".", ".."} or "\\" in parts[1]:
        raise ValueError("文件夹必须是 日期/文件夹")
    date_name, folder_name = parts
    working = PipelineState.from_payload(deepcopy(state.to_payload()))
    report = working.dataset_builds.get("full_init")
    if not isinstance(report, dict) or not init_root.is_dir():
        raise ValueError("请先生成 init，缺少来源分流记录")
    exports = report.get("test_exports", [])
    selected = [entry for entry in exports if entry["source"] == source_id]
    test_root = Path(report.get("test_root") or init_root.parent / TEST_ROOT_NAME).resolve()
    if any(test_root.is_relative_to(root.resolve()) or root.resolve().is_relative_to(test_root)
           for root in (init_root, data_root, init_root.parent / "shift_data")):
        raise ValueError("init_test 路径不能覆盖 init 或源数据")
    payload = load_folder_lineage(data_root)
    records = payload["folders"]
    record = records.get(source_id)
    if record is None:
        raise ValueError("缺少源文件夹走向记录")
    outputs = record.setdefault("outputs", {})
    if to_test_enable:
        if selected:
            raise ValueError("该来源已经进入 init_test，不能重复移入")
        value = outputs.get("init")
        if not value:
            raise ValueError("缺少 init 输出路径")
        source = Path(value).resolve()
        match = CANONICAL_FOLDER_PATTERN.fullmatch(source.name)
        if source.parent.parent != init_root.resolve() or match is None or match.group(2) != date_name:
            raise ValueError("init 来源路径无效")
        if any(key != source_id and value.get("outputs", {}).get("init") == str(source) for key, value in records.items()):
            raise ValueError("init 文件夹被多个来源共用")
        prefix = match.group(1).upper()
    else:
        if len(selected) != 1:
            raise ValueError("缺少唯一的 init_test 来源记录，不能重复移回")
        entry = selected[0]
        if entry.get("date") != date_name:
            raise ValueError("init_test 日期与来源记录不一致")
        source = (test_root / str(entry["target"])).resolve()
        if source.parent != test_root:
            raise ValueError("init_test 来源路径无效")
        if any(other is not entry and other["target"] == entry["target"] for other in exports):
            raise ValueError("init_test 文件夹被多个来源共用")
        if outputs.get("test") and Path(outputs["test"]).resolve() != source:
            raise ValueError("init_test 路径与来源走向不一致")
        prefix = parse_folder_prefix(folder_name, uppercase_enable=True)
    filenames = {path.name for path in iter_arc_data_paths(source)}
    if not filenames:
        raise ValueError("所选文件夹没有光谱")
    genus = next((name for name, prefixes in mapping.genera.items() if prefix in prefixes), None)
    if genus is None:
        raise ValueError(f"种前缀 {prefix} 未映射到属，请先补充分类名单")
    if record.get("init_pending") or filenames != {p.name for p in iter_arc_data_paths(data_root / source_id)}:
        raise ValueError("所选数据尚未同步源谱变更，请先按日期更新后再迁移")
    genus_dir = (init_root / genus).resolve()
    if genus_dir.parent != init_root.resolve():
        raise ValueError("属目录超出 init")
    if to_test_enable:
        if source.parent != genus_dir:
            raise ValueError("init 的属目录与种前缀映射不一致")
        # 统一编号后文件夹名已唯一，移入 init_test 保持原名、扁平存放。
        target = test_root / source.name
    else:
        # 移回 init 保持原名，放入对应属目录。
        target = genus_dir / source.name
    if target.exists():
        raise FileExistsError(f"目标文件夹已存在：{target}")
    summary = summarize_dataset(init_root)
    if to_test_enable:
        summary["file_count"] -= len(filenames)
        summary["folder_count"] -= 1
        summary["genus_count"] -= int(
            not any(
                not path.is_relative_to(source)
                for path in iter_arc_data_paths(genus_dir, recursive=True)
            )
        )
    else:
        summary["file_count"] += len(filenames)
        summary["folder_count"] += 1
        summary["genus_count"] += int(not has_arc_data(genus_dir, recursive=True))
    figure_root = init_root.parent / "fig_init"
    plot_path = figure_root / genus / f"{prefix}.png"
    snapshots = [source, target if target.parent.exists() else target.parent, plot_path, lineage_path(data_root), state_file]
    with preserve_paths(snapshots):
        target.parent.mkdir(parents=True, exist_ok=True)
        rename_directory(source, target)
        with TemporaryDirectory(prefix=".test_return_plot_", dir=figure_root.parent) as temporary:
            plots = build_prefix_plots_for_groups(init_root, Path(temporary), {(genus, prefix)})
            if plots:
                publish_file(plots[0], plot_path)
            elif to_test_enable:
                _remove_path(plot_path)
            else:
                raise RuntimeError("对应前缀图未生成")
        outputs["init"] = None if to_test_enable else str(target)
        outputs["test"] = str(target) if to_test_enable else None
        record["genus"] = genus
        record["prefix"] = prefix
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["updated_at"] = record["updated_at"]
        if to_test_enable:
            report["test_exports"] = [*exports, {
                "source": source_id, "source_name": folder_name, "target": target.name,
                "genus": genus, "date": date_name, "file_count": len(filenames),
            }]
            report["selected_test_folders"] = sorted(set(report.get("selected_test_folders", [])) | {source_id}, key=build_natural_key)
        else:
            report["test_exports"] = [other for other in exports if other is not entry]
            report["selected_test_folders"] = [value for value in report.get("selected_test_folders", []) if value != source_id]
        _update_dataset_counts(report, init_root, date_name, summary)
        write_folder_lineage(data_root, payload)
        working.dataset_builds.pop("latest", None)
        write_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    return {"source_id": source_id, "target_dir": str(target), "spectrum_count": len(filenames)}


def _update_dataset_counts(report: dict, init_root: Path, date_name: str, summary: dict) -> None:
    """供单组更新和分流迁移共用，不扫描或重建其他谱的数值。"""
    _refresh_dataset_counts(report, init_root, [date_name], summary)


def _refresh_dataset_counts(
    report: dict,
    init_root: Path,
    date_names: list[str] | set[str] | tuple[str, ...],
    summary: dict | None = None,
) -> None:
    """按测试目录与 init 磁盘重算报告统计；``summary`` 可复用调用方已扫描的结果。"""
    exports = report.get("test_exports", [])
    test_root = Path(report.get("test_root") or init_root.parent / TEST_ROOT_NAME).resolve()
    for entry in exports:
        target = (test_root / str(entry["target"])).resolve()
        if target.parent != test_root:
            raise ValueError("无效的 init_test 输出文件夹")
        entry["file_count"] = count_arc_data(target)
    report["count"] = summarize_dataset(init_root) if summary is None else summary
    init_count = report["count"]["file_count"]
    test_count = sum(int(entry["file_count"]) for entry in exports)
    for key in ("source_files", "output_files"):
        report[key] = init_count + test_count
    for key in ("init_source_files", "init_output_files"):
        report[key] = init_count
    for key in ("test_files", "test_output_files", "test_source_files"):
        report[key] = test_count
    report["test_folders"] = sorted({entry["target"] for entry in exports if int(entry["file_count"]) > 0}, key=build_natural_key)
    report["test_folder_count"] = len(report["test_folders"])
    date_dirs: dict[str, list[Path]] = {}
    for path in init_root.glob("*/*"):
        if not path.is_dir():
            continue
        match = CANONICAL_FOLDER_PATTERN.fullmatch(path.name)
        if match is not None:
            date_dirs.setdefault(match.group(2), []).append(path)
    date_stats = report.setdefault("date_stats", {})
    for date_name in date_names:
        dirs = date_dirs.get(date_name, [])
        date_exports = [entry for entry in exports if entry.get("date") == date_name and int(entry["file_count"]) > 0]
        date_init_count = sum(count_arc_data(path) for path in dirs)
        date_test_count = sum(int(entry["file_count"]) for entry in date_exports)
        date_stats[date_name] = {
            "source_files": date_init_count + date_test_count,
            "output_files": date_init_count + date_test_count,
            "folder_count": len(dirs) + len(date_exports),
            "test_files": date_test_count,
        }


def _arc_entries(folder: Path, allowed_names: set[str] | None = None) -> tuple[tuple[str, int, int], ...]:
    """按名称、大小和修改时间汇总光谱；用于比较 shift 与已有产物的内容摘要。"""
    if not folder.is_dir():
        return ()
    entries = []
    for entry in os.scandir(folder):
        if not entry.is_file() or not is_arc_data_name(entry.name):
            continue
        if allowed_names is not None and entry.name not in allowed_names:
            continue
        stat = entry.stat()
        entries.append((entry.name, stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(entries))


def _active_cosmic_folders(data_root: Path, date_name: str) -> dict[str, set[str]]:
    """列出日期下活动源文件夹与活动光谱名；cosmic 活动区是重建准据。"""
    date_dir = Path(data_root) / date_name
    active: dict[str, set[str]] = {}
    if not date_dir.is_dir():
        return active
    for folder in date_dir.iterdir():
        if not folder.is_dir() or folder.name in SPECIAL_FOLDER_NAMES or folder.name.startswith("."):
            continue
        names = {path.name for path in iter_arc_data_paths(folder)}
        if names:
            active[folder.name] = names
    return active


def _record_output_path(
    source_id: str,
    record: dict,
    test_targets: dict[str, str],
    test_root: Path,
) -> Path | None:
    """返回来源当前的输出目录；优先走向记录，缺失时回退到测试分流记录。"""
    outputs = record.get("outputs")
    if isinstance(outputs, dict):
        value = outputs.get("test") or outputs.get("init")
        if value:
            return Path(str(value)).resolve()
    target = test_targets.get(source_id)
    return (test_root / target).resolve() if target else None


def _record_output_is_test(source_id: str, record: dict, test_targets: dict[str, str]) -> bool:
    """判断来源当前进入 init_test 还是 init，优先级与 ``_record_output_path`` 一致。"""
    outputs = record.get("outputs")
    if isinstance(outputs, dict):
        value = outputs.get("test") or outputs.get("init")
        if value:
            return bool(outputs.get("test"))
    return source_id in test_targets


def collect_date_update_candidates(
    data_root: Path,
    shift_root: Path,
    init_root: Path,
    report: dict,
) -> list[dict[str, object]]:
    """检测可更新日期：新增、shift 待更新、待同步或文件与 shift 不一致。"""
    data_root = Path(data_root).resolve()
    shift_root = Path(shift_root).resolve()
    init_root = Path(init_root).resolve()
    report_dates = {str(value) for value in report.get("source_dates", [])}
    report_pending = {str(value) for value in report.get("pending_dates", [])}
    test_targets = {
        str(entry["source"]): str(entry["target"])
        for entry in report.get("test_exports", [])
        if isinstance(entry, dict) and entry.get("source") and entry.get("target")
    }
    test_root = Path(report.get("test_root") or init_root.parent / TEST_ROOT_NAME).resolve()
    records = load_folder_lineage(data_root).get("folders", {})
    if not isinstance(records, dict):
        records = {}
    init_dates = {
        match.group(2)
        for path in init_root.glob("*/*")
        if path.is_dir() and (match := CANONICAL_FOLDER_PATTERN.fullmatch(path.name))
    }
    date_names = init_dates | report_dates
    if shift_root.is_dir():
        date_names |= {
            path.name for path in shift_root.iterdir()
            if path.is_dir() and is_date_name(path.name)
        }
    candidates = []
    for date_name in sorted(date_names, key=build_natural_key):
        if not is_date_name(date_name):
            continue
        if (load_calibration_report(shift_root, date_name) or {}).get("status") != "manual_approved":
            continue
        active = _active_cosmic_folders(data_root, date_name)
        date_records = {
            source_id: record for source_id, record in records.items()
            if isinstance(record, dict) and record.get("date") == date_name
        }
        outputs = {
            source_id: _record_output_path(source_id, record, test_targets, test_root)
            for source_id, record in date_records.items()
        }
        test_sources = {
            source_id for source_id, record in date_records.items()
            if _record_output_is_test(source_id, record, test_targets)
        }
        if not active and not any(path is not None and path.is_dir() for path in outputs.values()):
            continue
        is_new = date_name not in report_dates and date_name not in init_dates and bool(active)
        pending: list[str] = []
        drifted: list[str] = []
        shift_missing: list[str] = []
        for folder_name in sorted(active, key=build_natural_key):
            names = active[folder_name]
            source_id = f"{date_name}/{folder_name}"
            shift_entries = _arc_entries(shift_root / date_name / folder_name)
            shift_names = {entry[0] for entry in shift_entries}
            missing = names - shift_names
            if missing:
                shift_missing.append(source_id)
                continue
            record = date_records.get(source_id, {})
            if record.get("init_pending"):
                pending.append(source_id)
            if is_new:
                continue
            output = outputs.get(source_id)
            expected = tuple(entry for entry in shift_entries if entry[0] in names)
            if output is None or not output.is_dir() or _arc_entries(output) != expected:
                drifted.append(source_id)
        for source_id, output in outputs.items():
            if source_id.split("/", 1)[-1] in active or output is None or not output.is_dir():
                continue
            drifted.append(source_id)
        if shift_missing:
            status = "shift_missing"
        elif is_new:
            status = "new"
        elif pending or drifted or date_name in report_pending:
            status = "changed"
        else:
            status = "ready"
        test_folder_count = sum(
            f"{date_name}/{folder_name}" in test_sources for folder_name in active
        )
        candidates.append({
            "date": date_name,
            "status": status,
            "is_new": is_new,
            "folder_count": len(active),
            "file_count": sum(len(names) for names in active.values()),
            "init_folder_count": len(active) - test_folder_count,
            "test_folder_count": test_folder_count,
            "pending_folders": tuple(pending),
            "drifted_folders": tuple(sorted(set(drifted), key=build_natural_key)),
            "shift_missing_folders": tuple(shift_missing),
        })
    return candidates


def _validate_date_shift_coverage(data_root: Path, shift_root: Path, date_name: str) -> None:
    """cosmic 活动谱必须已在 shift 中存在；缺失说明该日期 shift 尚未重建。"""
    missing: list[str] = []
    active = _active_cosmic_folders(data_root, date_name)
    for folder_name in sorted(active, key=build_natural_key):
        shift_names = {entry[0] for entry in _arc_entries(shift_root / date_name / folder_name)}
        missing.extend(
            f"{date_name}/{folder_name}/{name}"
            for name in sorted(active[folder_name] - shift_names, key=build_natural_key)
        )
    if missing:
        preview = "、".join(missing[:3])
        suffix = f" 等 {len(missing)} 条" if len(missing) > 3 else ""
        raise ValueError(f"shift_data 缺少活动光谱，请先在仿射页按日期更新 shift：{preview}{suffix}")


def _merge_date_init_report(
    previous: dict[str, object],
    updates: list[dict[str, object]],
    updated_dates: set[str],
    init_root: Path,
) -> dict[str, object]:
    """把按日期更新的报告并回全量摘要，替换受影响日期的明细。"""
    merged = dict(previous)
    date_status = {
        key: value for key, value in previous.get("date_status", {}).items()
        if key not in updated_dates
    }
    for update in updates:
        date_status.update(update.get("date_status", {}))
    merged["date_status"] = date_status
    for key in ("source_dates", "eligible_dates", "skipped_dates"):
        values = {value for value in previous.get(key, []) if value not in updated_dates}
        for update in updates:
            values.update(update.get(key, []))
        merged[key] = sorted(values, key=build_natural_key)
    merged["test_exports"] = [
        entry for entry in previous.get("test_exports", [])
        if str(entry.get("date")) not in updated_dates
    ]
    for update in updates:
        merged["test_exports"].extend(update.get("test_exports", []))
    selected = {
        value for value in previous.get("selected_test_folders", [])
        if str(value).split("/", 1)[0] not in updated_dates
    }
    for update in updates:
        selected.update(update.get("selected_test_folders", []))
    merged["selected_test_folders"] = sorted(selected, key=build_natural_key)
    merged["date_stats"] = dict(previous.get("date_stats", {}))
    merged["conflict_dates"] = list(previous.get("conflict_dates", []))
    merged["generated_profiles"] = list(previous.get("generated_profiles", []))
    merged["profile_stats"] = dict(previous.get("profile_stats", {}))
    merged["init_root"] = str(init_root.resolve())
    if previous.get("test_root"):
        merged["test_root"] = previous["test_root"]
    elif updates and updates[0].get("test_root"):
        merged["test_root"] = updates[0]["test_root"]
    if "supplement" in previous:
        merged["supplement"] = previous["supplement"]
    merged["mode"] = "full"
    return merged


def run_date_dataset_update(
    data_root: Path,
    shift_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    date_names: list[str] | tuple[str, ...] | set[str],
    selected_test_folders: list[str] | tuple[str, ...] | set[str] = (),
    *,
    hierarchy: HierarchySpec,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """按日期重建 init/init_test：已有日期沿用原分流，新增日期使用本次选择。"""
    dates = sorted({str(name) for name in date_names}, key=build_natural_key)
    if not dates:
        raise ValueError("请选择要更新的日期")
    data_root = Path(data_root).resolve()
    shift_root = Path(shift_root).resolve()
    init_root = Path(init_root).resolve()
    working = PipelineState.from_payload(deepcopy(state.to_payload()))
    report = working.dataset_builds.get("full_init")
    if not isinstance(report, dict) or not init_root.is_dir():
        raise ValueError("请先生成全量 init")
    for date_name in dates:
        if not is_date_name(date_name):
            raise ValueError("日期必须是八位数字")
        if not (shift_root / date_name).is_dir():
            raise FileNotFoundError(f"shift 日期不存在：{date_name}")
        if load_date_affine_parameters(shift_root, date_name) is None:
            raise ValueError(f"日期 {date_name} 尚未完成仿射校正")
        _validate_date_shift_coverage(data_root, shift_root, date_name)
    report_dates = {str(value) for value in report.get("source_dates", [])}
    init_dates = {
        match.group(2)
        for path in init_root.glob("*/*")
        if path.is_dir() and (match := CANONICAL_FOLDER_PATTERN.fullmatch(path.name))
    }
    new_dates = [
        date_name for date_name in dates
        if date_name not in report_dates and date_name not in init_dates
        and _active_cosmic_folders(data_root, date_name)
    ]
    existing_dates = [date_name for date_name in dates if date_name not in new_dates]
    new_selected = {str(value).replace("\\", "/") for value in selected_test_folders}
    if any(value.split("/", 1)[0] not in new_dates for value in new_selected):
        raise ValueError("测试分流只能来自本次新增的日期")
    records = load_folder_lineage(data_root).get("folders", {})
    existing_test_sources = sorted(
        {
            source_id for source_id, record in records.items()
            if isinstance(record, dict)
            and record.get("date") in existing_dates
            and isinstance(record.get("outputs"), dict)
            and record["outputs"].get("test")
        },
        key=build_natural_key,
    )
    test_root = Path(report.get("test_root") or init_root.parent / TEST_ROOT_NAME).resolve()
    updates = []
    if existing_dates:
        updates.append(
            build_init_datasets(
                shift_root,
                init_root,
                {},
                mapping,
                set(dates),
                mode="replace_dates",
                date_names=existing_dates,
                test_root=test_root,
                selected_test_folders=existing_test_sources,
                hierarchy=hierarchy,
                progress_callback=progress_callback,
            )
        )
    if new_dates:
        updates.append(
            build_init_datasets(
                shift_root,
                init_root,
                {},
                mapping,
                set(dates),
                mode="incremental",
                date_names=new_dates,
                test_root=test_root,
                selected_test_folders=sorted(new_selected, key=build_natural_key),
                hierarchy=hierarchy,
                progress_callback=progress_callback,
            )
        )
    merged = _merge_date_init_report(report, updates, set(dates), init_root)
    _refresh_dataset_counts(merged, init_root, dates)
    pending = {str(value) for value in merged.get("pending_dates", [])} - set(dates)
    records = load_folder_lineage(data_root).get("folders", {})
    source_dates = {str(value) for value in merged.get("source_dates", [])}
    for date_name in dates:
        if any(record.get("date") == date_name and record.get("init_pending") for record in records.values() if isinstance(record, dict)):
            pending.add(date_name)
        else:
            working.init_generated_dates.add(date_name)
    for record in records.values():
        if isinstance(record, dict) and record.get("init_pending") and str(record.get("date")) in source_dates:
            pending.add(str(record["date"]))
    merged["pending_dates"] = sorted(pending, key=build_natural_key)
    merged["status"] = "pending" if pending else "ready"
    working.dataset_builds["full_init"] = merged
    working.dataset_builds.pop("latest", None)
    write_pipeline_state(state_file, working)
    figure_root = init_root.parent / "fig_init"
    plot_error = None
    plot_count = 0
    try:
        plot_count = len(build_prefix_plots(init_root, figure_root))
    except (OSError, ValueError, RuntimeError) as error:
        plot_error = str(error)
    if plot_error is None:
        plot_state = working.dataset_builds.get("prefix_plots")
        if not isinstance(plot_state, dict):
            plot_state = {}
        plot_state["full_init"] = {"output_root": str(figure_root), "plot_count": plot_count}
        working.dataset_builds["prefix_plots"] = plot_state
        write_pipeline_state(state_file, working)
    date_stats = merged.get("date_stats", {})
    state.__dict__.update(working.__dict__)
    return {
        "updated_dates": dates,
        "new_dates": new_dates,
        "changed_dates": existing_dates,
        "spectrum_count": sum(int(date_stats.get(name, {}).get("output_files", 0)) for name in dates),
        "folder_count": sum(int(date_stats.get(name, {}).get("folder_count", 0)) for name in dates),
        "plot_count": plot_count,
        "plot_error": plot_error,
    }
