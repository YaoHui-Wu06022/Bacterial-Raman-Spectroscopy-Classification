"""复用日期级仿射参数，从 cosmic 原子重建 shift；不计算参数、不执行审核。"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from ramanv2.common.naming import (
    NON_TRAINING_FOLDER_NAMES,
    PIPELINE_RESERVED_FOLDER_NAMES,
    build_natural_key,
)
from ramanv2.common.publishing import create_temporary_path, preserve_paths, publish_directory
from ramanv2.data.calibration.affine import build_corrected_date, build_corrected_folder, load_calibration_report
from ramanv2.pipeline.lineage import lineage_path, record_affine_approval, record_output_path


EXCLUDED_FOLDER_NAMES = PIPELINE_RESERVED_FOLDER_NAMES | NON_TRAINING_FOLDER_NAMES


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
    scale = float(report.get("base_scale", report.get("scale", float("nan"))))
    offset = float(report.get("base_offset", report.get("offset", float("nan"))))
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
        raise ValueError(f"日期仿射参数无效：{date_name}")
    return scale, offset


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
            with preserve_paths(snapshots):
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
    if not folder_name or folder_name in EXCLUDED_FOLDER_NAMES or folder_name in {".", ".."} or Path(folder_name).name != folder_name or "/" in folder_name or "\\" in folder_name:
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
        with preserve_paths([target, lineage_path(cosmic)]):
            publish_directory(temporary, target)
            record_output_path(cosmic, date_name, folder_name, shift_path=str(target))
        return count
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = ["EXCLUDED_FOLDER_NAMES", "load_date_affine_parameters", "rebuild_shift_data", "rebuild_shift_folder"]
