"""复用或计算仿射参数，生成可直接构建 init 的 shift 数据。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from ramanv2.common.arc_data import count_arc_data, has_arc_data
from ramanv2.common.naming import BEAD_FOLDER_NAME, MANUAL_DELETE_RELATIVE
from ramanv2.pipeline.calibration import (
    rebuild_shift_data,
    run_affine_calibration,
    run_date_rebuild,
    run_identity_affine,
)
from ramanv2.pipeline.cleaning import commit_date_items, reconcile_cleaning_records, scan_review_folders
from ramanv2.pipeline.state import invalidate_downstream
from ramanv2.records.lineage import has_active_source_folders, lineage_path, load_folder_lineage
from ramanv2.ui.formatting import progress_ratio
from ramanv2.ui.read_cache import cached_calibration_reports
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.page_state import (
    INVALIDATE_CLEANING_AFFINE_DATASET,
    INVALIDATE_CLEANING_DATASET,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    load_page_state,
    render_action_error,
    render_flash,
    set_flash,
    write_page_state,
)


def _number(value: object) -> str:
    return "-" if value is None else f"{float(value):.8f}"


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


def _collect_source_date_names(data_root: Path) -> set[str]:
    """一次性列出活动区与两个删除归档中仍有内容的日期。"""
    names: set[str] = set()
    for base in (
        data_root,
        data_root / MANUAL_DELETE_RELATIVE,
        data_root / "delete" / "folder_delete",
    ):
        if not base.is_dir():
            continue
        for path in base.iterdir():
            if path.is_dir() and any(path.iterdir()):
                names.add(path.name)
    return names


def _report_fingerprint(shift_root: Path, date_names: list[str]) -> tuple[tuple[str, int, int], ...]:
    """报告文件的 mtime/size 指纹；缺失日期记为 (-1, 0)，供报告缓存精确失效。"""
    fingerprint = []
    for date_name in date_names:
        path = shift_root / date_name / "shift" / "affine.json"
        try:
            stat = path.stat()
        except OSError:
            fingerprint.append((date_name, -1, 0))
            continue
        fingerprint.append((date_name, stat.st_mtime_ns, stat.st_size))
    return tuple(fingerprint)


@st.cache_data(show_spinner=False)
def _collect_folder_changes(data_root: str, lineage_modified_ns: int) -> list[dict[str, object]]:
    """按日期/文件夹展示删除恢复批次；走向文件变化时缓存自动失效。"""
    del lineage_modified_ns
    root = Path(data_root)
    source_dates = _collect_source_date_names(root)
    operation_labels = {
        "manual_selection": "删除光谱",
        "default_cell1_cell2": "删除光谱",
        "folder_remove": "整夹删除",
        "date_remove": "日期删除",
        "prefix_folder_remove": "按前缀删除",
        "restore": "恢复光谱",
    }
    rows = []
    for record in load_folder_lineage(root)["folders"].values():
        if str(record["date"]) not in source_dates:
            continue
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
        render_flash("affine_date_rebuild")
        lineage_file = lineage_path(context.data_root)
        changes = _collect_folder_changes(
            str(context.data_root.resolve()),
            lineage_file.stat().st_mtime_ns if lineage_file.is_file() else 0,
        )
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
        shift_date_names = [
            path.name for path in sorted(context.shift_root.glob("*"))
            if path.is_dir() and len(path.name) == 8 and path.name.isascii() and path.name.isdigit()
            and (context.data_root / path.name).is_dir()
        ]
        date_reports = cached_calibration_reports(
            str(context.shift_root.resolve()),
            _report_fingerprint(context.shift_root, shift_date_names),
        )
        dates = [
            date_name for date_name in shift_date_names
            if (date_reports.get(date_name) or {}).get("status") == "manual_approved"
        ]
        if not dates:
            st.info("当前没有可复用仿射参数的日期。")
            return
        st.caption("复用已有仿射参数，从 cosmic_data 更新所选日期的 shift_data。init/init_test 请到「数据集获取」页按日期更新。")
        update_targets = [date_name for date_name in pending if date_name in dates]
        skipped_targets = [date_name for date_name in pending if date_name not in dates]
        if skipped_targets:
            st.caption("缺少 cosmic 数据或已确认参数，暂无法更新：" + "、".join(skipped_targets))
        if update_targets and st.button(
            f"一键更新全部待更新日期（{len(update_targets)} 个）",
            key="affine_rebuild_pending_confirm",
            width="stretch",
        ):
            rebuilt_dates: list[str] = []
            spectrum_total = 0
            failures: list[str] = []
            total = len(update_targets)
            with st.status(f"正在更新 {total} 个日期的 shift_data…", expanded=True) as status:
                progress = st.progress(0.0, text=f"0/{total}")
                for index, date_name in enumerate(update_targets, start=1):
                    progress.progress(
                        progress_ratio(index - 1, total),
                        text=f"[{index}/{total}] 正在更新 {date_name}…",
                    )
                    try:
                        result = run_date_rebuild(
                            context.data_root, context.shift_root, state, context.state_file, date_name
                        )
                    except UI_ACTION_ERRORS as error:
                        failures.append(f"{date_name}：{error}")
                        status.write(f"{date_name} 更新失败：{error}")
                        progress.progress(
                            progress_ratio(index, total),
                            text=f"[{index}/{total}] {date_name} 失败",
                        )
                        continue
                    count = int(result.get("spectrum_count", 0) or 0)
                    rebuilt_dates.append(date_name)
                    spectrum_total += count
                    progress.progress(
                        progress_ratio(index, total),
                        text=f"[{index}/{total}] {date_name} 完成（{count} 条光谱）",
                    )
                status.update(label=f"已处理 {total} 个日期", state="complete", expanded=False)
            invalidate_ui_cache(*INVALIDATE_CLEANING_AFFINE_DATASET)
            set_flash(
                "affine_date_rebuild",
                f"已更新 {len(rebuilt_dates)} 个日期的 shift_data（{spectrum_total} 条光谱）。"
                "init/init_test 可到「数据集获取」页按日期更新。",
            )
            if failures:
                set_flash(
                    "affine_date_rebuild",
                    "以下日期更新失败：" + "；".join(failures),
                    kind="error",
                )
            st.rerun()
        selected = st.selectbox("选择更新日期", dates, key="affine_rebuild_date")
        if not st.button("更新所选日期", key="affine_rebuild_date_confirm", width="stretch"):
            return
        try:
            with st.spinner(f"正在更新 {selected} 的 shift_data……"):
                result = run_date_rebuild(context.data_root, context.shift_root, state, context.state_file, selected)
            invalidate_ui_cache(*INVALIDATE_CLEANING_AFFINE_DATASET)
            set_flash(
                "affine_date_rebuild",
                f"{selected} 的 shift_data 已更新（{result['spectrum_count']} 条光谱）。"
                "init/init_test 可到「数据集获取」页按日期更新。",
            )
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error("日期更新", error)


def _render_identity_affine_page(context: PageContext, state) -> None:
    """无小球数据集的 identity 校准页：直接按字节复制并发布 scale=1/offset=0。"""
    st.title("仿射校准")
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        write_page_state(context, state)
    cleaned_dates = sorted(state.cleaning.completed_date_names)
    if not cleaned_dates and context.data_root.is_dir():
        cleaned_dates = sorted(
            path.name for path in context.data_root.iterdir() if path.is_dir()
        )
    eligible_dates = [
        date_name for date_name in cleaned_dates
        if has_active_source_folders(context.data_root, date_name)
    ]
    reports = cached_calibration_reports(
        str(context.shift_root.resolve()),
        _report_fingerprint(context.shift_root, eligible_dates),
    )
    approved = {
        date_name
        for date_name, report in reports.items()
        if report is not None and report.get("status") == "manual_approved"
    }
    pending = [date_name for date_name in eligible_dates if date_name not in approved]

    metrics = st.columns(3)
    metrics[0].metric("可选日期", f"{len(eligible_dates)} 个")
    metrics[1].metric("已发布", f"{len(approved)} 个")
    metrics[2].metric("待发布", f"{len(pending)} 个")

    render_flash("affine_identity")
    if st.button(
        "发布 identity 校准并重建 shift",
        type="primary",
        disabled=not eligible_dates,
        width="stretch",
    ):
        with st.spinner("正在复制 cosmic 光谱并发布 identity 校准……"):
            results = run_identity_affine(context.data_root, eligible_dates, context.shift_root)
        changed = [
            str(result["date"])
            for result in results
            if result.get("status") == "manual_approved"
        ]
        failed = [
            f"{result['date']}: {result.get('error')}"
            for result in results
            if result.get("status") == "error"
        ]
        for date_name in changed:
            state.affine_completed_dates.add(date_name)
        if changed:
            invalidate_downstream(state, changed)
        invalidate_ui_cache(*INVALIDATE_CLEANING_AFFINE_DATASET)
        write_page_state(context, state)
        set_flash(
            "affine_identity",
            f"已发布 {len(changed)} 个日期的 identity 校准"
            + (f"；失败 {len(failed)} 个：{'; '.join(failed)}" if failed else ""),
        )
        st.rerun()

    if not eligible_dates:
        st.info("当前没有可校准的日期。")
        return
    rows = [
        {
            "日期": date_name,
            "状态": (reports[date_name] or {}).get("status", "待发布"),
            "scale": (reports[date_name] or {}).get("scale", 1.0),
            "offset": (reports[date_name] or {}).get("offset", 0.0),
            "光谱数": (reports[date_name] or {}).get("spectrum_count", "-"),
        }
        for date_name in eligible_dates
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_affine_calibration_page(context: PageContext) -> None:
    state = load_page_state(context)
    has_bead = any(
        has_arc_data(date_dir / BEAD_FOLDER_NAME, recursive=True)
        for date_dir in context.data_root.iterdir()
        if date_dir.is_dir()
    ) if context.data_root.is_dir() else False
    if not has_bead:
        _render_identity_affine_page(context, state)
        return
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        write_page_state(context, state)
    cleaned_dates = sorted(state.cleaning.completed_date_names)
    all_reports = cached_calibration_reports(
        str(context.shift_root.resolve()),
        _report_fingerprint(context.shift_root, cleaned_dates),
    )
    eligible_dates = [
        date_name for date_name in cleaned_dates
        if has_active_source_folders(context.data_root, date_name)
        and (
            all_reports.get(date_name) is not None
            or has_arc_data(context.data_root / date_name / BEAD_FOLDER_NAME)
        )
    ]
    reports = {date_name: all_reports.get(date_name) for date_name in eligible_dates}
    report_affine_dates = {
        date_name
        for date_name, report in reports.items()
        if report is not None and str(report.get("status", "")) == "manual_approved"
    }
    affine_dates = sorted(report_affine_dates)
    if state.affine_completed_dates != report_affine_dates:
        state.affine_completed_dates = set(report_affine_dates)
        write_page_state(context, state)
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
    pending_dates = sorted((set(eligible_dates) - set(affine_dates)) | set(failed_dates))

    st.title("仿射校准")
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
        invalidate_ui_cache(*INVALIDATE_CLEANING_AFFINE_DATASET)
        write_page_state(context, state)
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
        bead_count = count_arc_data(context.data_root / date_name / BEAD_FOLDER_NAME)
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
                    invalidate_ui_cache(*INVALIDATE_CLEANING_DATASET)
                    write_page_state(context, state)
                    if result.missing_paths:
                        set_flash(
                            "affine_date_rebuild",
                            f"有 {len(result.missing_paths)} 条光谱已经不存在，其余数据已移除。",
                            kind="warning",
                        )
                    else:
                        set_flash(
                            "affine_date_rebuild",
                            f"已移除 {date_name} 的 {len(result.moved_paths)} 条原始光谱"
                            "（小球谱直接删除，其余进入人工删除目录）。",
                        )
                    st.rerun()
            _render_images(report)
