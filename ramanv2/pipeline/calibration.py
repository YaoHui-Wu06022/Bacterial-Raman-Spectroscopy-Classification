"""缺参数时计算小球仿射，已有参数时复用；所有派生重建共用同一内核。"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from ramanv2.common.publishing import preserve_paths
from ramanv2.data.calibration.affine import analyze_date, approve_and_apply_date, apply_folder_affine_compensation, load_calibration_report
from ramanv2.pipeline.lineage import lineage_path, load_folder_lineage, save_folder_lineage, record_affine_approval, record_output_path
from ramanv2.data.builders.init import CANONICAL_FOLDER_PATTERN, build_test_folder, build_spectrum_folder
from ramanv2.data.catalog import DatasetMapping
from ramanv2.data.plots.prefix import build_prefix_plots_for_groups
from ramanv2.data.profiles import TEST_ROOT_NAME
from ramanv2.pipeline.invalidation import invalidate_downstream
from ramanv2.pipeline.state import PipelineState, save_pipeline_state
from ramanv2.pipeline.shift import load_date_affine_parameters, rebuild_shift_data, rebuild_shift_folder


def run_affine_calibration(
    source_root: Path | str,
    date_names: list[str],
    output_root: Path | str,
) -> list[dict[str, object]]:
    """已有合格参数直接重建；缺失才分析小球，失败保留旧日期产物。"""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    reports = []
    for date_name in sorted(set(date_names)):
        try:
            existing = load_calibration_report(output, date_name)
            ready = existing is not None and existing.get("status") in {"ready", "limited_replicates"}
            parameters = None if ready else load_date_affine_parameters(output, date_name)
            if parameters is not None:
                rebuilt = rebuild_shift_data(source, output, [date_name])[0]
                if rebuilt["status"] != "rebuilt":
                    reports.append(rebuilt)
                    continue
                report = load_calibration_report(output, date_name)
                reports.append({**report, **rebuilt, "status": "manual_approved", "reused": True})
                continue
            date_dir = source / date_name
            if not ready and not any((date_dir / "小球").glob("*.arc_data")):
                reports.append({"date": date_name, "status": "skipped", "error": "缺少小球光谱"})
                continue
            with preserve_paths([output / date_name, lineage_path(source)]):
                analysis = existing if ready else analyze_date(date_dir, output)
                status = str(analysis.get("status", "needs_review"))
                if status not in {"ready", "limited_replicates"}:
                    reports.append({**analysis, "date": date_name, "status": status})
                    continue
                applied = approve_and_apply_date(date_dir, output)
                record_affine_approval(source, date_name, applied.get("approved_at"), applied["scale"], applied["offset"])
                reports.append({**analysis, **applied, "date": date_name, "status": "manual_approved"})
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            reports.append({"date": date_name, "status": "error", "error": str(error)})
    return reports


def run_date_rebuild(
    data_root: Path,
    shift_root: Path,
    state: PipelineState,
    state_file: Path,
    date_name: str,
) -> dict[str, object]:
    """仅复用参数重建所选日期的 shift；下游沿用已有待同步标记。"""
    if load_date_affine_parameters(shift_root, date_name) is None:
        raise ValueError("缺少已确认的仿射参数，请先完成该日期校准")
    working = PipelineState.from_payload(deepcopy(state.to_payload()))
    snapshots = [shift_root / date_name, lineage_path(data_root), state_file]
    with preserve_paths(snapshots):
        rebuilt = rebuild_shift_data(data_root, shift_root, [date_name])[0]
        if rebuilt["status"] != "rebuilt":
            raise RuntimeError(str(rebuilt.get("error", "shift 重建失败")))
        records = load_folder_lineage(data_root)["folders"]
        if any(record.get("date") == date_name and record.get("init_pending") for record in records.values()):
            invalidate_downstream(working, [date_name])
        working.affine_completed_dates.add(date_name)
        working.dataset_builds.pop("latest", None)
        save_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    return rebuilt


def run_folder_offset(
    data_root: Path,
    shift_root: Path,
    init_root: Path,
    mapping: DatasetMapping,
    state: PipelineState,
    state_file: Path,
    source_id: str,
    delta: float,
    figure_root: Path,
) -> dict[str, object]:
    """平移源谱及归档，按原分流同步 init 或对应 init_test 组；失败整体回滚。"""
    if not math.isfinite(delta):
        raise ValueError("频移量必须是有限数值")
    if delta == 0:
        return {"changed_folders": [], "spectrum_count": 0}
    parts = source_id.split("/")
    if len(parts) != 2:
        raise ValueError("文件夹必须是 日期/文件夹")
    date_name, folder_name = parts
    if len(date_name) != 8 or not date_name.isascii() or not date_name.isdigit() or not folder_name or folder_name in {".", ".."} or "\\" in folder_name:
        raise ValueError("无效的源文件夹")
    root = data_root.resolve()
    folders = [root / date_name / folder_name, root / "delete" / "manual" / date_name / folder_name, root / "delete" / "folder_delete" / date_name / folder_name]
    if any(not path.resolve().is_relative_to(root) for path in folders):
        raise ValueError("平移目标超出 cosmic 目录")
    completed = state.affine_completed_dates
    if date_name not in completed or not init_root.is_dir():
        raise ValueError("请先完成该日期仿射校正并生成 init，再补充频移")
    parameters = load_date_affine_parameters(shift_root, date_name)
    if parameters is None:
        raise ValueError("缺少已确认的仿射参数")
    working = PipelineState.from_payload(deepcopy(state.to_payload()))
    previous = working.dataset_builds.get("full_init")
    if not isinstance(previous, dict):
        raise ValueError("缺少 init 来源清单，请先重建全量 init")
    test_export = next((entry for entry in previous.get("test_exports", []) if entry.get("source") == source_id), None)
    test_target = None
    if test_export is not None:
        test_root = Path(previous.get("test_root") or init_root.parent / TEST_ROOT_NAME).resolve()
        target_name = str(test_export["target"])
        test_target = (test_root / target_name).resolve()
        if not target_name or target_name in {".", ".."} or "/" in target_name or "\\" in target_name or test_target.parent != test_root:
            raise ValueError("无效的 init_test 输出文件夹")
        if test_target.is_relative_to(init_root.resolve()) or init_root.resolve().is_relative_to(test_target):
            raise ValueError("init_test 输出不能覆盖 init")
    init_target = None
    plot_groups = set()
    if test_target is None:
        record = load_folder_lineage(root).get("folders", {}).get(source_id, {})
        output = record.get("outputs", {}).get("init")
        if not output:
            raise ValueError("缺少该文件夹的 init 来源记录，请先按日期更新")
        init_target = Path(output).resolve()
        match = CANONICAL_FOLDER_PATTERN.fullmatch(init_target.name)
        if init_target.parent.parent != init_root.resolve() or not match or match.group(2) != date_name:
            raise ValueError("该文件夹的 init 输出路径无效")
        if not init_target.is_dir():
            raise ValueError("该文件夹尚未生成 init，请先按日期更新")
        plot_groups = {(init_target.parent.name, match.group(1).upper())}
    old_count = len(list(init_target.glob("*.arc_data"))) if init_target is not None else 0
    snapshots = [*folders, shift_root / date_name / folder_name, lineage_path(root), state_file]
    snapshots.append(test_target if test_target is not None else init_target)
    snapshots.extend(figure_root / genus / f"{prefix}.png" for genus, prefix in plot_groups)
    with preserve_paths(snapshots):
        result = apply_folder_affine_compensation(root, shift_root, date_name, {folder_name: delta})
        rebuild_shift_folder(root, shift_root, date_name, folder_name)
        if test_target is not None:
            count = build_test_folder(shift_root / date_name / folder_name, test_target)
            change = count - int(test_export["file_count"])
            test_export["file_count"] = count
            for key in ("source_files", "output_files", "test_files", "test_output_files", "test_source_files", "test_output_files"):
                if key in previous:
                    previous[key] += change
            for key in ("source_files", "output_files", "test_files"):
                stats = previous.get("date_stats", {}).get(date_name, {})
                if key in stats:
                    stats[key] += change
            record_output_path(root, date_name, folder_name, test_path=str(test_target))
            result["test_updated"] = True
        else:
            count = build_spectrum_folder(shift_root / date_name / folder_name, init_target)
            change = count - old_count
            for key in ("source_files", "output_files", "init_source_files", "init_output_files"):
                if key in previous:
                    previous[key] += change
            stats = previous.get("date_stats", {}).get(date_name, {})
            for key in ("source_files", "output_files"):
                if key in stats:
                    stats[key] += change
            if isinstance(previous.get("count"), dict):
                previous["count"]["file_count"] += change
            payload = load_folder_lineage(root)
            payload["folders"][source_id]["init_pending"] = False
            save_folder_lineage(root, payload)
            pending = set(previous.get("pending_dates", []))
            if not any(record.get("date") == date_name and record.get("init_pending") for record in payload["folders"].values()):
                pending.discard(date_name)
                working.init_generated_dates.add(date_name)
            previous["pending_dates"] = sorted(pending)
            previous["status"] = "pending" if pending else "ready"
            build_prefix_plots_for_groups(init_root, figure_root, plot_groups)
        working.dataset_builds.pop("affine_compensations", None)
        save_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    return result


__all__ = ["run_affine_calibration", "run_date_rebuild", "run_folder_offset"]
