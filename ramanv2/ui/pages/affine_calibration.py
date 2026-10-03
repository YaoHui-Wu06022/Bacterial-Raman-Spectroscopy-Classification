"""仿射校准和校正后文件夹审核页面。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.pipeline.audit import reconcile_audit_completed_dates, run_shift_folder_audit
from ramanv2.pipeline.calibration import run_affine_calibration
from ramanv2.pipeline.cleaning import commit_date_items, reconcile_cleaning_records, scan_review_folders
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.state import invalidate_ui_cache, load_page_state as _load_state, save_page_state as _save


def _number(value: object) -> str:
    return "-" if value is None else f"{float(value):.8f}"


def _pending_date_names(
    eligible_dates: list[str],
    affine_dates: set[str],
    audit_dates: set[str],
) -> list[str]:
    """只返回尚未完成仿射或文件夹审核的日期，供页面增量显示。"""
    affine_pending = set(eligible_dates) - affine_dates
    audit_pending = (set(eligible_dates) & affine_dates) - audit_dates
    return sorted(affine_pending | audit_pending)


def _render_images(report: dict[str, object]) -> None:
    shift_dir = report.get("shift_dir")
    if not shift_dir:
        return
    paths = sorted(Path(str(shift_dir)).glob("*_calibration.png"))
    if not paths:
        return
    with st.expander("小球校准图", expanded=False):
        for path in paths:
            st.image(str(path), caption=path.name, use_container_width=True)


def render_affine_calibration_page(context: PageContext) -> None:
    state = _load_state(context)
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        _save(context, state)
    if reconcile_audit_completed_dates(context.shift_root, state):
        _save(context, state)
    cleaned_dates = sorted(state.cleaning.completed_date_names)
    eligible_dates = [
        date_name for date_name in cleaned_dates
        if (context.data_root / date_name / "小球").is_dir()
        and any((context.data_root / date_name / "小球").glob("*.arc_data"))
    ]
    affine_dates = sorted(state.affine_completed_dates & set(eligible_dates))
    audit_dates = sorted(state.audit_completed_dates & set(affine_dates))
    reports = {
        date_name: load_calibration_report(context.shift_root, date_name)
        for date_name in eligible_dates
    }
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
    pending_audit = [date_name for date_name in affine_dates if date_name not in audit_dates]
    pending_dates = _pending_date_names(
        eligible_dates,
        set(affine_dates),
        set(audit_dates),
    )
    pending_dates = sorted(set(pending_dates) | set(failed_dates))

    st.title("仿射校准")
    st.caption("校准只处理清洗完成且存在小球光谱的日期。文件夹审核需要单独点击。")
    metrics = st.columns(5)
    metrics[0].metric("清洗完成日期", f"{len(cleaned_dates)} 个")
    metrics[1].metric("可校准日期", f"{len(eligible_dates)} 个")
    metrics[2].metric("待校准日期", f"{len(pending_affine)} 个")
    metrics[3].metric("待文件夹审核", f"{len(pending_audit)} 个")
    metrics[4].metric("审核完成", f"{len(audit_dates)} 个")

    controls = st.columns(2)
    if controls[0].button("一键分析并校准", type="primary", disabled=not pending_affine, use_container_width=True):
        with st.spinner("正在分析小球并发布通过校验的校正数据……"):
            reports = run_affine_calibration(context.data_root, pending_affine, context.shift_root)
        for report in reports:
            date_name = str(report["date"])
            if report.get("status") == "manual_approved":
                state.affine_completed_dates.add(date_name)
                if not report.get("skipped"):
                    state.audit_completed_dates.discard(date_name)
            elif not report.get("skipped"):
                state.affine_completed_dates.discard(date_name)
        state.dataset_builds.pop("latest", None)
        invalidate_ui_cache("dataset")
        _save(context, state)
        st.session_state["affine_reports"] = reports
        st.rerun()
    if controls[1].button("运行文件夹审核", disabled=not pending_audit, use_container_width=True):
        with st.spinner("正在运行文件夹级 audit 并应用候选移除……"):
            reports = run_shift_folder_audit(context.shift_root, pending_audit, state, context.state_file)
        state.dataset_builds.pop("latest", None)
        invalidate_ui_cache("dataset")
        _save(context, state)
        st.session_state["audit_reports"] = reports
        st.rerun()

    if not eligible_dates:
        st.info("当前没有满足清洗完成和小球数据门禁的日期。")
        return

    if not pending_dates:
        st.success("所有可用日期都已完成仿射校准和文件夹审核。")
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
            st.write(f"文件夹审核：{'已完成' if date_name in audit_dates else '待审核'}")
            if report.get("errors"):
                st.warning("；".join(str(error) for error in report["errors"]))
            if status in {"needs_review", "error"}:
                st.error("该日期无法通过仿射校准，不能进入后续数据流程。")
                if st.button("整日期移除", key=f"remove-failed-calibration-{date_name}", use_container_width=True):
                    folders = scan_review_folders(context.data_root, [date_name])
                    with st.spinner(f"正在移除失败日期 {date_name} 的全部原始光谱……"):
                        result = commit_date_items(
                            context.data_root,
                            state,
                            context.state_file,
                            tuple(folders),
                        )
                    state.affine_completed_dates.discard(date_name)
                    state.audit_completed_dates.discard(date_name)
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
