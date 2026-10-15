"""复用或计算仿射参数，生成可直接构建 init 的 shift 数据。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.pipeline.calibration import run_affine_calibration, run_date_rebuild
from ramanv2.pipeline.cleaning import commit_date_items, reconcile_cleaning_records, scan_review_folders
from ramanv2.pipeline.invalidation import invalidate_downstream
from ramanv2.pipeline.lineage import has_active_source_folders, load_folder_lineage
from ramanv2.pipeline.shift import rebuild_shift_data
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.state import (
    invalidate_ui_cache,
    load_page_state as _load_state,
    save_page_state as _save,
)


def _number(value: object) -> str:
    return "-" if value is None else f"{float(value):.8f}"


def _pending_date_names(
    eligible_dates: list[str],
    affine_dates: set[str],
) -> list[str]:
    """返回尚未完成仿射的日期。"""
    return sorted(set(eligible_dates) - affine_dates)


def _render_images(report: dict[str, object]) -> None:
    shift_dir = report.get("shift_dir")
    if not shift_dir:
        return
    paths = sorted(Path(str(shift_dir)).glob("*_calibration.png"))
    if not paths:
        return
    with st.expander("小球校准图", expanded=False):
        for path in paths:
            st.image(str(path), caption=path.name, width="stretch")


def _run_analyze_and_rebuild(
    context: PageContext,
    pending_affine: list[str],
    reuse_dates: list[str],
) -> dict[str, list[dict[str, object]]]:
    """一键分析并校准：缺参数的日期算参数，已有参数的日期复用参数，两者都重建 shift。

    等价于「从 cosmic_data 重建 shift_data」：输出只由 cosmic_data 的当前内容与已确认的
    ``affine.json`` 参数决定，不会重复分析小球、不会重算已有参数。
    """
    calibration_reports: list[dict[str, object]] = []
    if pending_affine:
        calibration_reports = run_affine_calibration(
            context.data_root,
            pending_affine,
            context.shift_root,
        )
    rebuild_reports: list[dict[str, object]] = []
    if reuse_dates:
        rebuild_reports = rebuild_shift_data(context.data_root, context.shift_root, reuse_dates)
    return {"calibration": calibration_reports, "rebuild": rebuild_reports}


def _render_analyze_reports(reports: dict[str, list[dict[str, object]]] | None) -> None:
    """显示上一次「一键分析并校准」的结果。"""
    if not reports:
        return
    calibration = list(reports.get("calibration") or [])
    rebuild = list(reports.get("rebuild") or [])
    approved = [item for item in calibration if item.get("status") == "manual_approved"]
    rejected = [
        item for item in calibration
        if item.get("status") in {"needs_review", "error"}
    ]
    rebuilt = [item for item in rebuild if item.get("status") == "rebuilt"]
    spectra = sum(int(item.get("spectrum_count", 0) or 0) for item in rebuilt)
    if approved:
        st.success(f"已分析并发布 {len(approved)} 个日期的小球仿射参数。")
    if rebuilt:
        st.success(
            f"已复用已有参数、从 cosmic_data 重建 {len(rebuilt)} 个日期的 shift"
            f"（{spectra} 条光谱）。"
        )
    if approved or rebuilt:
        st.info("shift 数据已生成，可直接到数据集获取页建立 init。")
    for item in rejected:
        detail = item.get("error") or "；".join(str(error) for error in item.get("errors") or [])
        st.error(f"{item.get('date')}：{item.get('status')}{' · ' + str(detail) if detail else ''}")
    for item in rebuild:
        if item.get("status") == "error":
            st.error(f"{item.get('date')}：{item.get('error')}")


def _parse_change_time(value: str) -> datetime:
    """统一操作记录的时区，兼容旧记录中的无时区 UTC 时间。"""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _collect_folder_changes(data_root: Path) -> list[dict[str, object]]:
    """按日期/文件夹展示删除恢复批次，shift 时间只用于判断同步状态。"""
    operation_labels = {
        "manual_selection": "删除光谱",
        "default_cell1_cell2": "删除光谱",
        "folder_remove": "整夹删除",
        "date_remove": "日期删除",
        "prefix_folder_remove": "按前缀删除",
        "restore": "恢复光谱",
    }
    rows = []
    for record in load_folder_lineage(data_root)["folders"].values():
        # 同一批次逐谱写入的历史只展示一行，删除与恢复批次分别保留。
        events = {
            (_parse_change_time(event["recorded_at"]), operation_labels[event["operation"]])
            for event in record.get("history", [])
            if event.get("operation") in operation_labels and event.get("recorded_at")
        }
        if not events:
            continue
        shift_times = [
            _parse_change_time(value) for value in (
                record.get("outputs", {}).get("shift_updated_at"),
                record.get("affine", {}).get("approved_at"),
            ) if value
        ]
        shift_time = max(shift_times, default=None)
        for changed_at, operation in events:
            rows.append({
                "source_id": f"{record['date']}/{record['folder']}",
                "date": str(record["date"]),
                "folder": str(record["folder"]),
                "operation": operation,
                "changed_at": changed_at,
                "is_shift_pending": shift_time is None or changed_at > shift_time,
            })
    return sorted(rows, key=lambda row: row["changed_at"], reverse=True)


def _render_date_rebuild_section(context: PageContext, state) -> None:
    """按日期复用已有参数，只更新对应 shift 数据。"""
    with st.expander("按日期更新", expanded=False):
        message = st.session_state.pop("affine_date_rebuild_message", None)
        if message:
            st.success(message)
        changes = _collect_folder_changes(context.data_root)
        pending = list(dict.fromkeys(row["date"] for row in changes if row["is_shift_pending"]))
        if pending:
            st.warning("删除/恢复后待更新 shift 的日期：" + "、".join(pending))
        if changes:
            st.caption("删除/恢复影响的日期与文件夹（按操作时间倒序，同一批次多条光谱合并展示）")
            st.dataframe([
                {
                    "日期": row["date"],
                    "文件夹": row["folder"],
                    "操作": row["operation"],
                    "操作时间（北京时间）": row["changed_at"].astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
                    "shift 状态": "待更新" if row["is_shift_pending"] else "已更新",
                }
                for row in changes
            ], hide_index=True, width="stretch")
        else:
            st.caption("暂无删除/恢复操作记录。")
        dates = [
            path.name for path in sorted(context.shift_root.glob("*"))
            if path.is_dir() and len(path.name) == 8 and path.name.isascii() and path.name.isdigit()
            and (context.data_root / path.name).is_dir()
            and (load_calibration_report(context.shift_root, path.name) or {}).get("status") == "manual_approved"
        ]
        if not dates:
            st.info("当前没有可复用仿射参数的日期。")
            return
        st.caption("复用已有仿射参数，从 cosmic_data 更新所选日期的 shift_data。init/CSdata 请到「数据集获取」页按文件夹更新。")
        selected = st.selectbox("选择更新日期", dates, key="affine_rebuild_date")
        if not st.button("更新所选日期", key="affine_rebuild_date_confirm", width="stretch"):
            return
        try:
            with st.spinner(f"正在更新 {selected} 的 shift_data……"):
                result = run_date_rebuild(context.data_root, context.shift_root, state, context.state_file, selected)
            for name in ("cleaning", "affine", "dataset"):
                invalidate_ui_cache(name)
            st.session_state["affine_date_rebuild_message"] = f"{selected} 的 shift_data 已更新（{result['spectrum_count']} 条光谱）。init/CSdata 可到「数据集获取」页按文件夹更新。"
            st.rerun()
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            st.error(f"日期更新失败：{error}")


def render_affine_calibration_page(context: PageContext) -> None:
    state = _load_state(context)
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        _save(context, state)
    cleaned_dates = sorted(state.cleaning.completed_date_names)
    eligible_dates = [
        date_name for date_name in cleaned_dates
        if has_active_source_folders(context.data_root, date_name)
        and (load_calibration_report(context.shift_root, date_name) is not None or any((context.data_root / date_name / "小球").glob("*.arc_data")))
    ]
    reports = {
        date_name: load_calibration_report(context.shift_root, date_name)
        for date_name in eligible_dates
    }
    report_affine_dates = {
        date_name
        for date_name, report in reports.items()
        if report is not None and str(report.get("status", "")) == "manual_approved"
    }
    affine_dates = sorted(report_affine_dates)
    if state.affine_completed_dates != report_affine_dates:
        state.affine_completed_dates = set(report_affine_dates)
        _save(context, state)
    failed_dates = [
        date_name
        for date_name, report in reports.items()
        if date_name not in affine_dates
        and report is not None
        and str(report.get("status", "")) in {"needs_review", "error"}
    ]
    pending_affine = [
        date_name
        for date_name in eligible_dates
        if date_name not in affine_dates and date_name not in failed_dates
    ]
    pending_dates = _pending_date_names(
        eligible_dates,
        set(affine_dates),
    )
    pending_dates = sorted(set(pending_dates) | set(failed_dates))

    st.title("仿射校准")
    st.caption("校准处理清洗完成的日期；缺少参数时计算小球，已有合格参数直接复用；完成后直接建立 init。")
    st.caption(
        "一键分析并校准 = 从 cosmic_data 重建 shift_data：没有仿射参数的日期先分析小球算参数，"
        "已有参数的日期直接复用、不重算；两者都会重建该日期的 shift 派生谱，随后可直接建立 init。"
    )
    st.caption(
        f"本次将复用参数重建 {len(affine_dates)} 个已校准日期，分析 {len(pending_affine)} 个待校准日期。"
    )
    metrics = st.columns(3)
    metrics[0].metric("清洗完成日期", f"{len(cleaned_dates)} 个")
    metrics[1].metric("可校准日期", f"{len(eligible_dates)} 个")
    metrics[2].metric("待校准日期", f"{len(pending_affine)} 个")

    if st.button(
        "一键分析并校准",
        type="primary",
        disabled=not (pending_affine or affine_dates),
        help="缺参数的分析小球算参数，已有参数的直接复用；两者都从 cosmic_data 重建 shift。",
        width="stretch",
    ):
        with st.spinner("正在分析小球并从 cosmic_data 重建 shift 数据……"):
            reports = _run_analyze_and_rebuild(context, pending_affine, list(affine_dates))
        for report in reports["calibration"]:
            date_name = str(report["date"])
            if report.get("status") == "manual_approved":
                state.affine_completed_dates.add(date_name)
            elif not report.get("skipped"):
                state.affine_completed_dates.discard(date_name)
        rebuilt_dates = {
            str(report["date"])
            for report in reports["rebuild"]
            if report.get("status") == "rebuilt"
        }
        changed_dates = rebuilt_dates | {str(report["date"]) for report in reports["calibration"] if report.get("status") == "manual_approved" and not report.get("skipped")}
        if changed_dates:
            invalidate_downstream(state, changed_dates)
        state.dataset_builds.pop("latest", None)
        invalidate_ui_cache("dataset")
        invalidate_ui_cache("cleaning")
        invalidate_ui_cache("affine")
        _save(context, state)
        st.session_state["affine_analyze_reports"] = reports
        st.rerun()
    _render_analyze_reports(st.session_state.pop("affine_analyze_reports", None))
    _render_date_rebuild_section(context, state)

    if not eligible_dates:
        st.info("当前没有满足清洗完成和小球数据门禁的日期。")
        return

    if not pending_dates:
        st.success("所有可用日期都已完成仿射校准，可直接建立 init。")
        return

    for date_name in pending_dates:
        report = reports.get(date_name)
        status = "待分析" if report is None else str(report.get("status", "unknown"))
        bead_count = len(list((context.data_root / date_name / "小球").glob("*.arc_data")))
        with st.container(border=True):
            columns = st.columns((1.2, 1.0, 1.2, 2.0))
            columns[0].markdown(f"**{date_name}**")
            columns[1].write(f"小球谱：{bead_count}")
            columns[2].write(f"校准：{status}")
            if report is None:
                columns[3].write("参数：-")
                continue
            columns[3].write(f"scale={_number(report.get('scale'))}\noffset={_number(report.get('offset'))}")
            residuals = {
                **dict(report.get("primary_residuals_cm-1", {})),
                **dict(report.get("auxiliary_residuals_cm-1", {})),
            }
            if residuals:
                st.write("残差：" + "；".join(f"{key}={float(value):.3f}" for key, value in residuals.items()))
            st.write(f"有效重复数：{report.get('valid_repeat_count', 0)}")
            if report.get("errors"):
                st.warning("；".join(str(error) for error in report["errors"]))
            if status in {"needs_review", "error"}:
                st.error("该日期无法通过仿射校准，不能进入后续数据流程。")
                if st.button("整日期移除", key=f"remove-failed-calibration-{date_name}", width="stretch"):
                    folders = scan_review_folders(context.data_root, [date_name])
                    with st.spinner(f"正在移除失败日期 {date_name} 的全部原始光谱……"):
                        result = commit_date_items(
                            context.data_root,
                            state,
                            context.state_file,
                            tuple(folders),
                        )
                    state.affine_completed_dates.discard(date_name)
                    state.dataset_builds.pop("latest", None)
                    invalidate_ui_cache("cleaning")
                    invalidate_ui_cache("dataset")
                    _save(context, state)
                    if result.missing_paths:
                        st.warning(f"有 {len(result.missing_paths)} 条光谱已经不存在，其余数据已移除。")
                    else:
                        st.success(f"已将 {date_name} 的 {len(result.moved_paths)} 条原始光谱移入人工删除目录。")
                    st.rerun()
            _render_images(report)


__all__ = ["render_affine_calibration_page"]
