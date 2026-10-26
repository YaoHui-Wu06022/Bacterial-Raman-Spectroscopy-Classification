"""沿用源文件夹走向，从 shift 最小更新对应的 init 或 CSdata 组。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from ramanv2.common.naming import build_natural_key, is_test_source_folder, parse_folder_prefix, parse_test_folder_prefix
from ramanv2.common.publishing import preserve_paths, publish_file, rename_directory, _remove_path
from ramanv2.data.builders.init import CANONICAL_FOLDER_PATTERN, build_spectrum_folder
from ramanv2.data.catalog import DatasetMapping
from ramanv2.data.count import summarize_dataset
from ramanv2.data.plots.prefix import build_prefix_plots_for_groups
from ramanv2.pipeline.lineage import lineage_path, load_folder_lineage, save_folder_lineage
from ramanv2.pipeline.shift import load_date_affine_parameters
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


def collect_dataset_source_folders(data_root: Path, shift_root: Path, state: PipelineState) -> list[str]:
    """列出可更新的来源，以及已整夹删除但尚未同步移除旧产物的来源。"""
    previous = state.dataset_builds.get("full_init")
    if not isinstance(previous, dict):
        return []
    exports = {entry["source"] for entry in previous.get("cs_exports", [])}
    folders = load_folder_lineage(data_root)["folders"]
    return sorted(
        (
            source_id for source_id, record in folders.items()
            if record.get("date") in state.affine_completed_dates
            and (record.get("outputs", {}).get("init") or source_id in exports)
            and (
                ((data_root / source_id).is_dir() and (shift_root / source_id).is_dir())
                or (record.get("folder_status") == "folder_deleted" and record.get("init_pending")
                    and not (data_root / source_id).exists())
            )
        ),
        key=build_natural_key,
    )


def collect_cs_return_sources(init_root: Path, state: PipelineState) -> list[str]:
    """只列出仍有实际 CSdata 光谱且有来源记录的目录，不扫描全量 init。"""
    report = state.dataset_builds.get("full_init")
    if not isinstance(report, dict):
        return []
    root = Path(report.get("test_root") or init_root.parent / "CSdata").resolve()
    return sorted({
        entry["source"] for entry in report.get("cs_exports", [])
        if (root / str(entry["target"])).resolve().parent == root
        and any((root / str(entry["target"])).glob("*.arc_data"))
    }, key=build_natural_key)


def run_cs_to_init(
    data_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
) -> dict[str, object]:
    """直接迁移已有 CSdata 单组到 init，复用双向迁移事务。"""
    return _run_dataset_transfer(data_root, init_root, mapping, state, state_file, source_id, False)


def collect_init_export_sources(data_root: Path, init_root: Path, state: PipelineState) -> dict[str, str]:
    """列出有来源记录的现有 init 组，保留日期/源目录作为操作主键。"""
    report = state.dataset_builds.get("full_init")
    if not isinstance(report, dict):
        return {}
    exports = {entry["source"] for entry in report.get("cs_exports", [])}
    records = load_folder_lineage(data_root)["folders"]
    sources = {}
    for source_id, record in records.items():
        value = record.get("outputs", {}).get("init")
        if source_id in exports or not value:
            continue
        path = Path(value).resolve()
        if path.parent.parent == init_root.resolve() and path.is_dir() and any(path.glob("*.arc_data")):
            sources[source_id] = path.relative_to(init_root.resolve()).as_posix()
    return dict(sorted(sources.items(), key=lambda item: build_natural_key(item[1])))


def run_init_to_cs(
    data_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
) -> dict[str, object]:
    """将所选 init 整夹移入 CSdata，不重建其余数据。"""
    return _run_dataset_transfer(data_root, init_root, mapping, state, state_file, source_id, True)


def _run_dataset_transfer(
    data_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
    to_csdata_enable: bool,
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
    exports = report.get("cs_exports", [])
    selected = [entry for entry in exports if entry["source"] == source_id]
    cs_root = Path(report.get("test_root") or init_root.parent / "CSdata").resolve()
    if any(cs_root.is_relative_to(root.resolve()) or root.resolve().is_relative_to(cs_root)
           for root in (init_root, data_root, init_root.parent / "shift_data")):
        raise ValueError("CSdata 路径不能覆盖 init 或源数据")
    payload = load_folder_lineage(data_root)
    records = payload["folders"]
    record = records.get(source_id)
    if record is None:
        raise ValueError("缺少源文件夹走向记录")
    outputs = record.setdefault("outputs", {})
    if to_csdata_enable:
        if selected:
            raise ValueError("该来源已经进入 CSdata，不能重复移入")
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
            raise ValueError("缺少唯一的 CSdata 来源记录，不能重复移回")
        entry = selected[0]
        if entry.get("date") != date_name:
            raise ValueError("CSdata 日期与来源记录不一致")
        source = (cs_root / str(entry["target"])).resolve()
        if source.parent != cs_root:
            raise ValueError("CSdata 来源路径无效")
        if any(other is not entry and other["target"] == entry["target"] for other in exports):
            raise ValueError("CSdata 文件夹被多个来源共用")
        if outputs.get("csdata") and Path(outputs["csdata"]).resolve() != source:
            raise ValueError("CSdata 路径与来源走向不一致")
        if outputs.get("init") and Path(outputs["init"]).exists():
            raise ValueError("该来源已有 init 输出，不能重复移回")
        prefix = parse_test_folder_prefix(folder_name)
    filenames = {path.name for path in source.glob("*.arc_data") if path.is_file()}
    if not filenames:
        raise ValueError("所选文件夹没有光谱")
    genus = next((name for name, prefixes in mapping.genera.items() if prefix in prefixes), None)
    if genus is None:
        raise ValueError(f"种前缀 {prefix} 未映射到属，请先补充分类名单")
    if record.get("init_pending") or filenames != {p.name for p in (data_root / source_id).glob("*.arc_data") if p.is_file()}:
        raise ValueError("所选数据尚未同步源谱变更，请先按文件夹更新后再迁移")
    genus_dir = (init_root / genus).resolve()
    if genus_dir.parent != init_root.resolve():
        raise ValueError("属目录超出 init")
    if to_csdata_enable:
        if source.parent != genus_dir:
            raise ValueError("init 的属目录与种前缀映射不一致")
        # CSdata 命名沿用 init 模式：{prefix}{date}_{number}，扁平结构。
        existing_numbers = [
            int(match.group(3))
            for path in cs_root.glob("*")
            if path.is_dir() and (match := CANONICAL_FOLDER_PATTERN.fullmatch(path.name))
            and match.group(1).upper() == prefix and match.group(2) == date_name
        ]
        new_folder_name = f"{prefix}{date_name}_{max(existing_numbers, default=0) + 1:02d}"
        target = cs_root / new_folder_name
        if any(other["target"] == target.name for other in exports) or any(
            key != source_id and value.get("outputs", {}).get("csdata") == str(target)
            for key, value in records.items()
        ):
            raise ValueError("CSdata 目标名称已被其他来源保留")
    else:
        # 跳过仍被其他来源（含删除归档）保留的编号，不改动现有组号。
        reserved = [*genus_dir.glob("*"), *(Path(value["outputs"]["init"]) for value in records.values() if value.get("outputs", {}).get("init"))]
        numbers = [int(match.group(3)) for path in reserved
                   if path.parent.resolve() == genus_dir and (match := CANONICAL_FOLDER_PATTERN.fullmatch(path.name))
                   and match.group(1).upper() == prefix and match.group(2) == date_name]
        target = genus_dir / f"{prefix}{date_name}_{max(numbers, default=0) + 1:02d}"
    if target.exists():
        raise FileExistsError(f"目标文件夹已存在：{target}")
    summary = summarize_dataset(init_root)
    if to_csdata_enable:
        summary["file_count"] -= len(filenames)
        summary["folder_count"] -= 1
        summary["genus_count"] -= int(not any(not path.is_relative_to(source) for path in genus_dir.rglob("*.arc_data")))
    else:
        summary["file_count"] += len(filenames)
        summary["folder_count"] += 1
        summary["genus_count"] += int(not any(genus_dir.rglob("*.arc_data")))
    figure_root = init_root.parent / "fig_init"
    plot_path = figure_root / genus / f"{prefix}.png"
    snapshots = [source, target if target.parent.exists() else target.parent, plot_path, lineage_path(data_root), state_file]
    with preserve_paths(snapshots):
        target.parent.mkdir(parents=True, exist_ok=True)
        rename_directory(source, target)
        with TemporaryDirectory(prefix=".cs_return_plot_", dir=figure_root.parent) as temporary:
            plots = build_prefix_plots_for_groups(init_root, Path(temporary), {(genus, prefix)})
            if plots:
                publish_file(plots[0], plot_path)
            elif to_csdata_enable:
                _remove_path(plot_path)
            else:
                raise RuntimeError("对应前缀图未生成")
        outputs["init"] = None if to_csdata_enable else str(target)
        outputs["csdata"] = str(target) if to_csdata_enable else None
        record["genus"] = genus
        record["prefix"] = prefix
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["updated_at"] = record["updated_at"]
        if to_csdata_enable:
            report["cs_exports"] = [*exports, {
                "source": source_id, "source_name": folder_name, "target": target.name,
                "genus": genus, "date": date_name, "file_count": len(filenames),
            }]
            report["selected_cs_folders"] = sorted(set(report.get("selected_cs_folders", [])) | {source_id}, key=build_natural_key)
        else:
            report["cs_exports"] = [other for other in exports if other is not entry]
            report["selected_cs_folders"] = [value for value in report.get("selected_cs_folders", []) if value != source_id]
        _update_dataset_counts(report, init_root, date_name, summary)
        save_folder_lineage(data_root, payload)
        working.dataset_builds.pop("latest", None)
        save_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    return {"source_id": source_id, "target_dir": str(target), "spectrum_count": len(filenames)}


def _update_dataset_counts(report: dict, init_root: Path, date_name: str, summary: dict) -> None:
    """供单组更新和分流迁移共用，不扫描或重建其他谱的数值。"""
    exports = report.get("cs_exports", [])
    cs_root = Path(report.get("test_root") or init_root.parent / "CSdata").resolve()
    for entry in exports:
        target = (cs_root / str(entry["target"])).resolve()
        if target.parent != cs_root:
            raise ValueError("无效的 CSdata 输出文件夹")
        entry["file_count"] = sum(path.is_file() for path in target.glob("*.arc_data"))
    report["count"] = summary
    init_count = summary["file_count"]
    cs_count = sum(int(entry["file_count"]) for entry in exports)
    for key in ("source_files", "output_files"):
        report[key] = init_count + cs_count
    for key in ("init_source_files", "init_output_files"):
        report[key] = init_count
    for key in ("test_files", "test_output_files", "cs_source_files", "cs_output_files"):
        report[key] = cs_count
    report["test_folders"] = sorted({entry["target"] for entry in exports if int(entry["file_count"]) > 0}, key=build_natural_key)
    report["cs_folder_count"] = len(report["test_folders"])
    date_dirs = [path for path in init_root.glob("*/*")
                 if path.is_dir() and (match := CANONICAL_FOLDER_PATTERN.fullmatch(path.name)) and match.group(2) == date_name]
    date_exports = [entry for entry in exports if entry["date"] == date_name and int(entry["file_count"]) > 0]
    date_init_count = sum(len(list(path.glob("*.arc_data"))) for path in date_dirs)
    date_cs_count = sum(int(entry["file_count"]) for entry in date_exports)
    report.setdefault("date_stats", {})[date_name] = {
        "source_files": date_init_count + date_cs_count,
        "output_files": date_init_count + date_cs_count,
        "folder_count": len(date_dirs) + len(date_exports),
        "test_files": date_cs_count,
    }


def run_folder_dataset_update(
    data_root: Path,
    shift_root: Path,
    init_root: Path,
    state: PipelineState,
    state_file: Path,
    source_id: str,
) -> dict[str, object]:
    """只替换所选源的原目标组并更新统计；绘图或状态保存失败时整体回滚。"""
    parts = source_id.split("/")
    if len(parts) != 2:
        raise ValueError("文件夹必须是 日期/文件夹")
    date_name, folder_name = parts
    if len(date_name) != 8 or not date_name.isascii() or not date_name.isdigit() or not folder_name or folder_name in {".", ".."} or "\\" in folder_name:
        raise ValueError("无效的源文件夹")
    source = (shift_root / source_id).resolve()
    active = (data_root / source_id).resolve()
    if source.parent != (shift_root / date_name).resolve() or active.parent != (data_root / date_name).resolve():
        raise ValueError("源文件夹超出数据目录")
    if date_name not in state.affine_completed_dates or load_date_affine_parameters(shift_root, date_name) is None:
        raise ValueError("请先完成所选日期的仿射校正并更新 shift_data")
    working = PipelineState.from_payload(deepcopy(state.to_payload()))
    report = working.dataset_builds.get("full_init")
    if not isinstance(report, dict):
        raise ValueError("缺少 init 来源清单，请先生成全量 init")
    payload = load_folder_lineage(data_root)
    records = payload["folders"]
    record = records.get(source_id)
    if record is None:
        raise ValueError("缺少文件夹输出走向，请先生成全量 init")
    # 只有已记录的整夹删除才同步移除旧产物；普通缺谱或路径缺失仍报错。
    is_deleted = record.get("folder_status") == "folder_deleted" and not active.exists()
    exports = report.get("cs_exports", [])
    cs_export = next((entry for entry in exports if entry["source"] == source_id), None)
    is_cs = cs_export is not None
    if is_cs:
        output_root = Path(report.get("test_root") or init_root.parent / "CSdata").resolve()
        target = (output_root / str(cs_export["target"])).resolve()
        if target.parent != output_root:
            raise ValueError("无效的 CSdata 输出文件夹")
        if any(entry["source"] != source_id and entry["target"] == cs_export["target"] for entry in exports):
            raise ValueError("CSdata 目标被其他来源占用")
    else:
        output_root = init_root.resolve()
        target_value = record.get("outputs", {}).get("init")
        if not target_value:
            raise ValueError("缺少原 init 输出路径，请先生成全量 init")
        target = Path(target_value).resolve()
        match = CANONICAL_FOLDER_PATTERN.fullmatch(target.name)
        if target.parent.parent != output_root or match is None or match.group(2) != date_name:
            raise ValueError("无效的 init 输出文件夹")
        if any(key != source_id and value.get("outputs", {}).get("init") == str(target) for key, value in records.items()):
            raise ValueError("init 目标被其他来源占用，请先按日期更新")
    if target.is_relative_to(shift_root.resolve()) or target.is_relative_to(data_root.resolve()):
        raise ValueError("输出目录不能覆盖源数据")
    if is_cs and (target.is_relative_to(init_root.resolve()) or init_root.resolve().is_relative_to(target)):
        raise ValueError("CSdata 输出不能覆盖 init")
    filenames = {path.name for path in active.glob("*.arc_data") if path.is_file()}
    if not filenames and not is_deleted:
        raise ValueError("所选文件夹没有活动光谱")
    if any(not (source / name).is_file() for name in filenames):
        raise ValueError("shift 缺少活动光谱，请先更新对应 shift 文件夹")

    # 在快照创建前统计，避免把同级临时备份误计入 init 数量。
    summary = summarize_dataset(init_root)
    old_count = len(list(target.glob("*.arc_data")))
    if not is_cs:
        summary["exists"] = True
        if is_deleted:
            summary["folder_count"] -= int(old_count > 0)
            summary["genus_count"] -= int(old_count > 0 and not any(
                not path.is_relative_to(target) for path in target.parent.rglob("*.arc_data")
            ))
        else:
            summary["genus_count"] += int(not any(target.parent.rglob("*.arc_data")))
            summary["folder_count"] += int(old_count == 0)
        summary["file_count"] += len(filenames) - old_count

    figure_root = init_root.parent / "fig_init"
    prefix = parse_folder_prefix(target.name, uppercase_enable=True)
    plot_path = figure_root / target.parent.name / f"{prefix}.png"
    snapshots = [target, lineage_path(data_root), state_file]
    if not is_cs:
        snapshots.append(plot_path)
    with preserve_paths(snapshots):
        if is_deleted:
            _remove_path(target)
            count = 0
        else:
            count = build_spectrum_folder(source, target, filenames)
        if is_cs:
            cs_export["file_count"] = count
        else:
            # 只计算这一属/前缀的图；临时图生成成功后再发布，保留其他图片。
            with TemporaryDirectory(prefix=".folder_plot_", dir=figure_root.parent) as temporary:
                plots = build_prefix_plots_for_groups(init_root, Path(temporary), {(target.parent.name, prefix)})
                if plots:
                    publish_file(plots[0], plot_path)
                elif is_deleted:
                    _remove_path(plot_path)
                else:
                    raise RuntimeError("对应前缀图未生成")
        _update_dataset_counts(report, init_root, date_name, summary)
        # 逐组清除待同步状态；同一天尚有其他变更时不能提前标为完成。
        was_pending = record.get("init_pending", False)
        record["init_pending"] = False
        save_folder_lineage(data_root, payload)
        pending = set(report.get("pending_dates", []))
        date_pending = any(value.get("date") == date_name and value.get("init_pending") for value in records.values())
        if date_pending:
            pending.add(date_name)
            working.init_generated_dates.discard(date_name)
        elif was_pending or date_name not in pending:
            pending.discard(date_name)
            working.init_generated_dates.add(date_name)
        report["pending_dates"] = sorted(pending)
        report["status"] = "pending" if pending else "ready"
        working.dataset_builds.pop("latest", None)
        save_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    result = {"source_id": source_id, "target_dir": str(target), "spectrum_count": count, "csdata_updated": is_cs}
    if is_deleted:
        result.update(deleted=True, removed_file_count=old_count)
    return result
