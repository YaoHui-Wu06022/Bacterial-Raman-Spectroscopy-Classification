"""缺参数时计算小球仿射，已有参数时复用；所有派生重建共用同一内核。"""

from __future__ import annotations

import math
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from ramanv2.common import file_ops
from ramanv2.common.arc_data import count_arc_data, has_arc_data
from ramanv2.common.file_ops import create_temporary_path, preserve_paths, publish_directory, write_json
from ramanv2.common.naming import (
    BEAD_FOLDER_NAME,
    CANONICAL_FOLDER_PATTERN,
    FOLDER_DELETE_RELATIVE,
    MANUAL_DELETE_RELATIVE,
    SPECIAL_FOLDER_NAMES,
    build_natural_key,
)
from ramanv2.data.calibration.affine import (
    analyze_date,
    apply_folder_affine_compensation,
    approve_and_apply_date,
    build_corrected_date,
    build_corrected_folder,
    load_calibration_report,
)
from ramanv2.records.lineage import lineage_path, load_folder_lineage, write_folder_lineage, record_affine_approval, record_output_path
from ramanv2.data.builders.init import build_test_folder, build_spectrum_folder
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.data.plots.prefix import build_prefix_plots_for_groups
from ramanv2.data.profiles import TEST_ROOT_NAME
from ramanv2.pipeline.state import PipelineState, invalidate_downstream, write_pipeline_state


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
            if not ready and not has_arc_data(date_dir / BEAD_FOLDER_NAME):
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
    with file_ops.preserve_paths(snapshots):
        rebuilt = rebuild_shift_data(data_root, shift_root, [date_name])[0]
        if rebuilt["status"] != "rebuilt":
            raise RuntimeError(str(rebuilt.get("error", "shift 重建失败")))
        records = load_folder_lineage(data_root)["folders"]
        if any(record.get("date") == date_name and record.get("init_pending") for record in records.values()):
            invalidate_downstream(working, [date_name])
        working.affine_completed_dates.add(date_name)
        working.dataset_builds.pop("latest", None)
        write_pipeline_state(state_file, working)
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
    folders = [root / date_name / folder_name, root / MANUAL_DELETE_RELATIVE / date_name / folder_name, root / FOLDER_DELETE_RELATIVE / date_name / folder_name]
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
    old_count = count_arc_data(init_target) if init_target is not None else 0
    snapshots = [*folders, shift_root / date_name / folder_name, lineage_path(root), state_file]
    snapshots.append(test_target if test_target is not None else init_target)
    snapshots.extend(figure_root / genus / f"{prefix}.png" for genus, prefix in plot_groups)
    with file_ops.preserve_paths(snapshots):
        result = apply_folder_affine_compensation(root, date_name, {folder_name: delta})
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
            write_folder_lineage(root, payload)
            pending = set(previous.get("pending_dates", []))
            if not any(record.get("date") == date_name and record.get("init_pending") for record in payload["folders"].values()):
                pending.discard(date_name)
                working.init_generated_dates.add(date_name)
            previous["pending_dates"] = sorted(pending)
            previous["status"] = "pending" if pending else "ready"
            build_prefix_plots_for_groups(init_root, figure_root, plot_groups)
        working.dataset_builds.pop("affine_compensations", None)
        write_pipeline_state(state_file, working)
    state.__dict__.update(working.__dict__)
    return result


def load_date_affine_parameters(shift_root: Path | str, date_name: str) -> tuple[float, float] | None:
    """缺参数返回 None；已有报告不合格则报错，不能静默重算或使用坏参数。"""
    if len(date_name) != 8 or not date_name.isascii() or not date_name.isdigit():
        raise ValueError("日期必须是八位数字")
    report = load_calibration_report(shift_root, date_name)
    if report is None:
        return None
    if report.get("status") == "pending" and report.get("invalidated_reason") == "restore_invalidated":
        return None
    if report.get("status") != "manual_approved":
        raise ValueError(f"日期仿射参数尚未通过校验：{date_name}")
    scale = float(report.get("scale", float("nan")))
    offset = float(report.get("offset", float("nan")))
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
        raise ValueError(f"日期仿射参数无效：{date_name}")
    return scale, offset


def run_identity_affine(
    source_root: Path | str,
    date_names: list[str] | tuple[str, ...] | set[str],
    output_root: Path | str,
) -> list[dict[str, object]]:
    """无小球数据集：发布 scale=1/offset=0 的已确认报告，并按字节复制 shift。"""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    reports = []
    for date_name in sorted(set(str(name) for name in date_names), key=build_natural_key):
        temporary = None
        try:
            if len(date_name) != 8 or not date_name.isascii() or not date_name.isdigit():
                raise ValueError("日期必须是八位数字")
            source_date = (source / date_name).resolve()
            target = (output / date_name).resolve()
            if source_date.parent != source or target.parent != output:
                raise ValueError("日期目录超出数据根目录")
            if not source_date.is_dir() or not has_arc_data(source_date, recursive=True):
                raise FileNotFoundError(f"cosmic 日期不存在或没有光谱：{date_name}")
            temporary = create_temporary_path(target)
            temporary.mkdir()
            count = _copy_identity_date(source_date, temporary)
            shift_dir = temporary / "shift"
            shift_dir.mkdir()
            approved_at = datetime.now(timezone.utc).isoformat()
            write_json(
                shift_dir / "affine.json",
                {
                    "status": "manual_approved",
                    "mode": "identity",
                    "manual_approval_enable": True,
                    "date": date_name,
                    "formula": "corrected_wavenumber = raw_wavenumber",
                    "scale": 1.0,
                    "offset": 0.0,
                    "spectrum_count": count,
                    "note": "该数据集没有小球，使用 identity 校准",
                    "approved_at": approved_at,
                    "errors": [],
                },
            )
            folders = sorted(
                path.name for path in temporary.iterdir() if path.is_dir() and path.name != "shift"
            )
            with file_ops.preserve_paths([target, lineage_path(source)]):
                publish_directory(temporary, target)
                record_affine_approval(
                    source,
                    date_name,
                    approved_at,
                    1.0,
                    0.0,
                    shift_dir=target,
                )
            reports.append(
                {
                    "date": date_name,
                    "status": "manual_approved",
                    "mode": "identity",
                    "scale": 1.0,
                    "offset": 0.0,
                    "spectrum_count": count,
                    "folders": folders,
                }
            )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            reports.append({"date": date_name, "status": "error", "error": str(error)})
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
    return reports


def _copy_identity_date(source_date: Path, temporary_date: Path) -> int:
    """按原字节复制日期内所有普通文件，跳过校准球与流水线保留目录。"""
    count = 0
    for source in sorted(
        source_date.rglob("*"),
        key=lambda path: build_natural_key(str(path.relative_to(source_date))),
    ):
        relative = source.relative_to(source_date)
        if any(part in SPECIAL_FOLDER_NAMES or part.startswith(".") for part in relative.parts):
            continue
        target = temporary_date / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def rebuild_shift_data(
    data_root: Path | str,
    shift_root: Path | str,
    date_names: list[str] | tuple[str, ...] | set[str],
) -> list[dict[str, object]]:
    """先完整生成日期临时目录再发布，成功后可直接构建 init。"""
    cosmic = Path(data_root).resolve()
    output = Path(shift_root).resolve()
    reports = []
    for date_name in sorted(set(date_names), key=build_natural_key):
        temporary = None
        try:
            parameters = load_date_affine_parameters(output, date_name)
            if parameters is None:
                reports.append({"date": date_name, "status": "needs_calibration", "spectrum_count": 0, "folders": []})
                continue
            source = (cosmic / date_name).resolve()
            target = (output / date_name).resolve()
            if source.parent != cosmic or target.parent != output:
                raise ValueError("日期目录超出数据根目录")
            if not source.is_dir():
                raise FileNotFoundError(f"cosmic 日期不存在：{date_name}")
            temporary = create_temporary_path(target)
            temporary.mkdir()
            count = build_corrected_date(source, temporary, *parameters, target / "shift")
            folders = sorted(path.name for path in temporary.iterdir() if path.is_dir() and path.name != "shift")
            snapshots = [target, lineage_path(cosmic)]
            with file_ops.preserve_paths(snapshots):
                publish_directory(temporary, target)
                report = load_calibration_report(output, date_name)
                record_affine_approval(cosmic, date_name, report.get("approved_at"), *parameters, shift_dir=target)
            reports.append({"date": date_name, "status": "rebuilt", "spectrum_count": count, "folders": folders})
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            reports.append({"date": date_name, "status": "error", "error": str(error), "spectrum_count": 0, "folders": []})
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
    return reports


def rebuild_shift_folder(data_root: Path, shift_root: Path, date_name: str, folder_name: str) -> int:
    """只重建一个来源文件夹；保留同日其他组及日期校准报告，失败恢复旧组。"""
    parameters = load_date_affine_parameters(shift_root, date_name)
    if parameters is None:
        raise ValueError("缺少已确认的仿射参数")
    if not folder_name or folder_name in SPECIAL_FOLDER_NAMES or folder_name in {".", ".."} or Path(folder_name).name != folder_name or "/" in folder_name or "\\" in folder_name:
        raise ValueError("无效的源文件夹")
    cosmic = data_root.resolve()
    output = shift_root.resolve()
    source = (cosmic / date_name / folder_name).resolve()
    target = (output / date_name / folder_name).resolve()
    if not source.is_relative_to(cosmic) or not target.is_relative_to(output):
        raise ValueError("文件夹超出数据根目录")
    if not source.is_dir():
        raise FileNotFoundError(f"cosmic 来源不存在：{source}")
    temporary = create_temporary_path(target)
    temporary.mkdir()
    try:
        count = build_corrected_folder(source, temporary, *parameters)
        if count == 0:
            raise ValueError("所选文件夹没有活动光谱")
        with file_ops.preserve_paths([target, lineage_path(cosmic)]):
            publish_directory(temporary, target)
            record_output_path(cosmic, date_name, folder_name, shift_path=str(target))
        return count
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "load_date_affine_parameters",
    "rebuild_shift_data",
    "rebuild_shift_folder",
    "run_affine_calibration",
    "run_date_rebuild",
    "run_folder_offset",
]
