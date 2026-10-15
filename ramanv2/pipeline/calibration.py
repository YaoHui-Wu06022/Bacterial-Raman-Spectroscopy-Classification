"""缺参数时计算小球仿射，已有参数时复用；所有派生重建共用同一内核。"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from ramanv2.common.publishing import preserve_paths
from ramanv2.data.calibration.affine import analyze_date, approve_and_apply_date, apply_folder_affine_compensation, load_calibration_report
from ramanv2.pipeline.lineage import lineage_path, record_affine_approval, record_output_path
from ramanv2.data.builders.init import build_cs_folder, build_full_init
from ramanv2.data.catalog import DatasetMapping
from ramanv2.data.plots.prefix import build_prefix_plots
from ramanv2.pipeline.state import PipelineState, save_pipeline_state
from ramanv2.pipeline.shift import load_date_affine_parameters, rebuild_shift_data


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


def rebuild_shift_dates(
    source_root: Path | str,
    output_root: Path | str,
    date_names: list[str] | tuple[str, ...] | set[str],
) -> list[dict[str, object]]:
    """兼容原 CLI 调用顺序，复用参数重建，绝不重新拟合小球。"""
    return rebuild_shift_data(source_root, output_root, date_names)


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
    """平移源谱及归档，按原分流同步 init 或对应 CSdata 组；失败整体回滚。"""
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
    cs_export = next((entry for entry in previous.get("cs_exports", []) if entry.get("source") == source_id), None)
    cs_target = None
    if cs_export is not None:
        cs_root = Path(previous.get("test_root") or init_root.parent / "CSdata").resolve()
        target_name = str(cs_export["target"])
        cs_target = (cs_root / target_name).resolve()
        if not target_name or target_name in {".", ".."} or "/" in target_name or "\\" in target_name or cs_target.parent != cs_root:
            raise ValueError("无效的 CSdata 输出文件夹")
        if cs_target.is_relative_to(init_root.resolve()) or init_root.resolve().is_relative_to(cs_target):
            raise ValueError("CSdata 输出不能覆盖 init")
    previous_dates = set(previous.get("source_dates", [])) if cs_target is None else set()
    for name in previous_dates:
        if any(folder.is_dir() and folder.name not in {"小球", "shift", "delete"} and any(folder.rglob("*.arc_data")) for folder in (root / name).glob("*")):
            if name not in completed or load_date_affine_parameters(shift_root, name) is None:
                raise ValueError(f"已有 init 日期未满足重建门禁：{name}")
    selected_cs = previous.get("selected_cs_folders", [])
    snapshots = [*folders, shift_root / date_name, lineage_path(root), state_file]
    snapshots.extend([cs_target] if cs_target is not None else [init_root, figure_root])
    with preserve_paths(snapshots):
        result = apply_folder_affine_compensation(root, shift_root, date_name, {folder_name: delta})
        rebuilt = rebuild_shift_data(root, shift_root, [date_name])[0]
        if rebuilt["status"] != "rebuilt":
            raise RuntimeError(str(rebuilt.get("error", "shift 重建失败")))
        if cs_target is not None:
            count = build_cs_folder(shift_root / date_name / folder_name, cs_target)
            change = count - int(cs_export["file_count"])
            cs_export["file_count"] = count
            for key in ("source_files", "output_files", "test_files", "test_output_files", "cs_source_files", "cs_output_files"):
                if key in previous:
                    previous[key] += change
            for key in ("source_files", "output_files", "test_files"):
                stats = previous.get("date_stats", {}).get(date_name, {})
                if key in stats:
                    stats[key] += change
            record_output_path(root, date_name, folder_name, csdata_path=str(cs_target))
            result["csdata_updated"] = True
        else:
            report = build_full_init(shift_root, init_root, mapping, completed, selected_cs_folders=selected_cs, rebuild_csdata_enable=False)
            # init 分支不发布 CSdata，保留上次成功导出信息。
            for key in ("test_root", "test_files", "test_folders", "cs_folder_count", "test_output_files", "cs_source_files", "cs_output_files", "cs_exports"):
                if key in previous:
                    report[key] = previous[key]
            working.dataset_builds["full_init"] = report
            working.init_generated_dates = set(report["source_dates"])
            build_prefix_plots(init_root, figure_root)
        working.dataset_builds.pop("affine_compensations", None)
        save_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    return result


__all__ = ["rebuild_shift_dates", "run_affine_calibration", "run_folder_offset"]
