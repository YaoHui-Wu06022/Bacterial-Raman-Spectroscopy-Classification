"""校正后日期的文件夹级 audit 编排。"""

from __future__ import annotations

import json
from pathlib import Path

from ramanv2.audit.folder import run_folder_audit
from ramanv2.pipeline.lineage import initialize_folder_lineage, record_audit_result
from ramanv2.pipeline.state import PipelineState, save_pipeline_state


_AUDIT_IGNORED_NAMES = {"小球", "shift", "delete", "audit_runs", ".raman_pipeline"}


def reconcile_audit_completed_dates(
    shift_root: Path | str,
    state: PipelineState,
) -> bool:
    """根据已应用的 audit 报告补齐日期完成状态。

    按文件夹核对最新的 applied 报告，只有当前每个普通文件夹都有不早于其
    文件的报告时，才把日期加入完成集合。
    """
    root = Path(shift_root).resolve()
    cosmic_root = root.parent / "cosmic_data"
    if cosmic_root.is_dir():
        initialize_folder_lineage(cosmic_root)
    audit_root = root / "audit_runs"
    if not audit_root.is_dir():
        return False
    changed = False
    for date_name in sorted(state.affine_completed_dates):
        date_dir = root / date_name
        if not date_dir.is_dir():
            continue
        folders = [
            path for path in date_dir.iterdir()
            if path.is_dir()
            and path.name not in _AUDIT_IGNORED_NAMES
            and any(path.glob("*.arc_data"))
        ]
        if not folders:
            continue
        latest_by_folder: dict[str, int] = {}
        for run_dir in audit_root.iterdir():
            summary_path = run_dir / "summary.json"
            if not run_dir.is_dir() or not summary_path.is_file():
                continue
            try:
                report = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if report.get("status") != "applied":
                continue
            report_folders = report.get("folders", [])
            if not isinstance(report_folders, list):
                continue
            try:
                report_mtime = summary_path.stat().st_mtime_ns
            except OSError:
                continue
            for relative in report_folders:
                relative_text = str(relative).replace("\\", "/")
                prefix = f"{date_name}/"
                if relative_text.startswith(prefix):
                    folder_name = relative_text[len(prefix):].split("/", 1)[0]
                    key = f"{date_name}/{folder_name}"
                    latest_by_folder[key] = max(latest_by_folder.get(key, 0), report_mtime)
        date_complete = True
        for folder_dir in folders:
            key = f"{date_name}/{folder_dir.name}"
            report_mtime = latest_by_folder.get(key, 0)
            if report_mtime == 0:
                date_complete = False
                break
            file_mtimes = [
                path.stat().st_mtime_ns
                for path in folder_dir.glob("*.arc_data")
                if path.is_file()
            ]
            if file_mtimes and report_mtime < max(file_mtimes):
                date_complete = False
                break
        if date_complete and date_name not in state.audit_completed_dates:
            state.audit_completed_dates.add(date_name)
            changed = True
    return changed


def run_shift_folder_audit(
    shift_root: Path | str,
    date_names: list[str],
    state: PipelineState | None = None,
    state_path: Path | str | None = None,
) -> list[dict[str, object]]:
    """逐日期执行 audit 并直接应用候选移除结果。"""
    root = Path(shift_root).resolve()
    reports: list[dict[str, object]] = []
    for date_name in sorted(set(date_names)):
        if state is not None and date_name in state.audit_completed_dates:
            reports.append(
                {
                    "date": date_name,
                    "status": "skipped",
                    "reason": "already_completed",
                    "message": "该日期已经完成文件夹审核，跳过重复审核",
                }
            )
            continue
        date_dir = root / date_name
        if not date_dir.is_dir():
            reports.append({"date": date_name, "status": "skipped", "error": "shift 日期目录不存在"})
            continue
        try:
            report = run_folder_audit(root, date_name=date_name)
            report = dict(report)
            report["date"] = date_name
            reports.append(report)
            report_folders = report.get("folders", [])
            folder_names = {
                str(item).replace("\\", "/").split("/", 1)[1].split("/", 1)[0]
                for item in report_folders
                if str(item).replace("\\", "/").startswith(f"{date_name}/")
            }
            for folder_name in sorted(folder_names):
                if cosmic_root.is_dir():
                    record_audit_result(
                        cosmic_root,
                        date_name,
                        folder_name,
                        "applied" if report.get("status") == "applied" else str(report.get("status", "failed")),
                        None if report.get("status") == "applied" else str(report.get("error", report.get("reason", "audit_failed"))),
                    )
            if state is not None:
                if report.get("status") == "applied":
                    state.audit_completed_dates.add(date_name)
                else:
                    state.audit_completed_dates.discard(date_name)
        except (OSError, RuntimeError, ValueError) as error:
            reports.append({"date": date_name, "status": "error", "error": str(error)})
            if state is not None:
                state.audit_completed_dates.discard(date_name)
    if state is not None and state_path is not None:
        save_pipeline_state(state_path, state)
    return reports


__all__ = ["reconcile_audit_completed_dates", "run_shift_folder_audit"]
