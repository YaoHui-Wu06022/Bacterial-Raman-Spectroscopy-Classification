"""清洗完成日期的小球仿射校准编排。"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from ramanv2.common.publishing import publish_directory
from ramanv2.data.calibration.affine import analyze_date, approve_and_apply_date, load_calibration_report


def run_affine_calibration(
    source_root: Path | str,
    date_names: list[str],
    output_root: Path | str,
) -> list[dict[str, object]]:
    """分析并发布通过校验的日期级校正数据。"""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    reports: list[dict[str, object]] = []
    for date_name in sorted(set(date_names)):
        date_dir = source / date_name
        bead_dir = date_dir / "小球"
        if not date_dir.is_dir():
            reports.append({"date": date_name, "status": "skipped", "error": "日期目录不存在"})
            continue
        if not any(bead_dir.glob("*.arc_data")):
            reports.append({"date": date_name, "status": "skipped", "error": "缺少小球光谱"})
            continue
        existing = load_calibration_report(output, date_name)
        if existing is not None and existing.get("status") == "manual_approved":
            reports.append(
                {
                    **existing,
                    "date": date_name,
                    "status": "manual_approved",
                    "skipped": True,
                    "message": "已有校准结果，跳过重复处理",
                }
            )
            continue
        try:
            analysis = analyze_date(date_dir, output)
            status = str(analysis.get("status", "needs_review"))
            if status in {"ready", "limited_replicates"}:
                applied = approve_and_apply_date(date_dir, output)
                reports.append({**analysis, **applied, "date": date_name, "status": "manual_approved"})
            else:
                reports.append({**analysis, "date": date_name, "status": status})
        except (OSError, RuntimeError, ValueError) as error:
            reports.append({"date": date_name, "status": "error", "error": str(error)})
    return reports


def rebuild_shift_dates(
    source_root: Path | str,
    output_root: Path | str,
    date_names: list[str] | tuple[str, ...] | set[str],
) -> list[dict[str, object]]:
    """从原始 cosmic_data 重建指定日期，失败时保留旧 shift 输出。"""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    # 暂存目录放在输出根的同级，清理暂存目录时不会误触目标日期目录。
    staging = output.parent / f".{output.name}_rebuild_{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, object]] = []
    try:
        for date_name in sorted({str(item) for item in date_names}):
            date_dir = source / date_name
            if not date_dir.is_dir():
                reports.append({"date": date_name, "status": "skipped", "error": "日期目录不存在"})
                continue
            try:
                analysis = analyze_date(date_dir, staging)
                status = str(analysis.get("status", "needs_review"))
                if status not in {"ready", "limited_replicates"}:
                    reports.append({**analysis, "date": date_name, "status": status, "rebuilt": False})
                    continue
                approved = approve_and_apply_date(date_dir, staging)
                staged_date = staging / date_name
                publish_directory(staged_date, output / date_name)
                report = {**analysis, **approved, "date": date_name, "status": "manual_approved", "rebuilt": True}
                report["shift_dir"] = str(output / date_name / "shift")
                reports.append(report)
            except (OSError, RuntimeError, ValueError) as error:
                reports.append({"date": date_name, "status": "error", "error": str(error), "rebuilt": False})
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return reports


__all__ = ["rebuild_shift_dates", "run_affine_calibration"]
