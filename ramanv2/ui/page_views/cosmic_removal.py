"""宇宙射线去除页：从 raw_data 预览并生成 cosmic_data。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st

from ramanv2.common.arc_data import count_arc_data
from ramanv2.common.naming import BEAD_FOLDER_NAME, is_date_name
from ramanv2.common.plotting import configure_matplotlib_fonts
from ramanv2.data.folder_cosmic import (
    build_folder_cosmic_options,
    build_folder_cosmic_payload,
    build_folder_cosmic_reference,
    read_spectrum_folder,
)
from ramanv2.pipeline.folder_ingest import (
    FolderIngestRequest,
    is_valid_prefix,
    run_folder_ingest,
)
from ramanv2.ui.formatting import progress_ratio
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.page_state import (
    INVALIDATE_ALL,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    render_action_error,
    render_flash,
    set_flash,
)


configure_matplotlib_fonts()


def _scan_raw_batches(raw_root: Path, cosmic_root: Path, enabled: bool) -> tuple[list[dict], list[str]]:
    """扫描 raw_data/<日期>/<前缀>，返回批次状态与需展示的警告。"""
    batches: list[dict] = []
    warnings: list[str] = []
    if not raw_root.is_dir():
        return batches, warnings
    for date_dir in sorted(raw_root.iterdir(), key=lambda path: path.name):
        if not date_dir.is_dir() or date_dir.name.startswith("."):
            continue
        if not is_date_name(date_dir.name):
            warnings.append(f"忽略非日期目录：{date_dir.name}")
            continue
        for folder in sorted(date_dir.iterdir(), key=lambda path: path.name):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            is_bead = folder.name == BEAD_FOLDER_NAME
            if not is_bead and not is_valid_prefix(folder.name):
                warnings.append(f"忽略无效批次：{date_dir.name}/{folder.name}")
                continue
            direct_count = count_arc_data(folder)
            nested_count = count_arc_data(folder, recursive=True)
            target = cosmic_root / date_dir.name / folder.name
            target_exists = target.is_dir()
            if target_exists:
                status = "已生成（只去文件头）" if is_bead else "已生成（不覆盖）"
            elif nested_count != direct_count:
                status = "阻止：含子目录光谱"
            elif direct_count == 0:
                status = "阻止：无光谱"
            elif is_bead:
                status = "待生成（只去文件头）"
            elif not enabled:
                status = "未启用"
            else:
                status = "待生成"
            batches.append({
                "key": f"{date_dir.name}/{folder.name}",
                "date": date_dir.name,
                "prefix": folder.name,
                "source_dir": str(folder),
                "file_count": int(direct_count),
                "nested_count": int(nested_count),
                "target_exists": bool(target_exists),
                "is_bead": bool(is_bead),
                "status": status,
                "blocked": status.startswith("阻止"),
                "pending": status.startswith("待生成"),
            })
    return batches, warnings


def _selected_batch(batches: list[dict], key: str) -> dict | None:
    for batch in batches:
        if batch["key"] == key:
            return batch
    return None


def _report_matches(report: dict, context: PageContext, batch: dict) -> bool:
    """判断预览是否仍对应当前数据集、批次和配置。"""
    target_text = report.get("target_dir")
    try:
        target_path = Path(target_text).resolve() if target_text else None
    except (TypeError, OSError):
        return False
    if target_path is None or not target_path.is_relative_to(context.dataset_spec.cosmic_dir):
        return False
    if report.get("dataset_id") != context.dataset_spec.dataset_id:
        return False
    if report.get("folder_key") != batch["key"]:
        return False
    if bool(report.get("bead")) != bool(batch["is_bead"]):
        return False
    if batch["is_bead"]:
        return True
    current_options = build_folder_cosmic_payload(context.dataset_spec.cosmic_removal)
    return report.get("options") == current_options


def _render_preview(report: dict) -> None:
    """展示清洗指标、逐文件统计与文件夹中值/guard 图。"""
    metrics = st.columns(4)
    metrics[0].metric("光谱数", report.get("spectrum_count", 0))
    metrics[1].metric("替换点", report.get("removed_points", 0))
    metrics[2].metric("涉及谱", report.get("removed_spectra", 0))
    metrics[3].metric("保护峰位", report.get("guard_peak_count", 0))
    if report.get("bead"):
        st.info("小球批次：只去文件头，不做宇宙射线去除。")
        return
    if report.get("blocked_reason"):
        st.warning(report["blocked_reason"])
    rows = report.get("files") or []
    if rows:
        st.dataframe(pd.DataFrame(rows))
    try:
        folder = read_spectrum_folder(report["source_dir"])
        options = build_folder_cosmic_options(report.get("options"))
        reference = build_folder_cosmic_reference(folder.intensities, options)
        if reference is not None:
            figure, axis = plt.subplots(figsize=(10, 3))
            x = folder.wavenumbers[0]
            axis.plot(x, reference.median, color="#4C72B0", lw=1.0, label="文件夹中值")
            guard = reference.guard_mask
            axis.scatter(
                x[guard], reference.median[guard],
                color="#D62728", s=12, label="真实峰保护",
            )
            axis.set_xlabel("拉曼位移 (cm$^{-1}$)")
            axis.set_ylabel("强度")
            axis.legend(fontsize=7)
            axis.tick_params(labelsize=7)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
    except UI_ACTION_ERRORS:
        pass


def _generate_batches(context: PageContext, batches: list[dict]) -> list[str]:
    """按顺序生成待生成批次；单个失败不中断其余批次，返回失败描述。"""
    failures: list[str] = []
    total = len(batches)
    with st.status("正在生成 cosmic_data…", expanded=True) as status:
        progress = st.progress(0.0, text=f"0/{total}")
        for index, batch in enumerate(batches, start=1):
            status.write(f"[{index}/{total}] {batch['key']}")
            try:
                run_folder_ingest(FolderIngestRequest(
                    spec=context.dataset_spec,
                    source_dir=batch["source_dir"],
                    date_name=batch["date"],
                    prefix=batch["prefix"],
                    apply=True,
                    bead=batch["is_bead"],
                ))
            except UI_ACTION_ERRORS as error:
                failures.append(f"{batch['key']}：{error}")
                status.write(f"失败：{error}")
            progress.progress(progress_ratio(index, total), text=f"{index}/{total}")
        status.update(
            label="生成完成" if not failures else "部分批次失败",
            state="complete" if not failures else "error",
        )
    return failures


def render_cosmic_removal_page(context: PageContext) -> None:
    """从 raw_data 预览并生成 cosmic_data；目标存在时不覆盖。"""
    st.title("宇宙射线去除")
    render_flash("cosmic_removal")
    st.caption(
        "把未去宇宙射线的文件夹放在 "
        f"dataset/{context.dataset_spec.directory.name}/raw_data/<日期>/<前缀>/；"
        "小球谱放在 raw_data/<日期>/小球/。"
    )
    enabled = context.dataset_spec.cosmic_removal.enable
    if not enabled:
        st.info(
            "当前数据集未启用宇宙射线去除：普通批次需启用后才能预览/生成；"
            "小球批次只去文件头，可直接预览/生成。普通批次请到"
            "「数据集配置」勾选「启用文件夹级去宇宙射线」。"
        )

    batches, warnings = _scan_raw_batches(context.raw_root, context.data_root, enabled)
    if warnings:
        with st.expander("raw_data 中的忽略项", expanded=False):
            for warning in warnings:
                st.warning(warning)
    if not batches:
        st.info(
            "raw_data 下还没有可识别的批次。请按 "
            "raw_data/<八位日期>/<ASCII 前缀>/ 放入 .arc_data 文件"
        )
        return

    pending_batches = [batch for batch in batches if batch["pending"]]
    generated_count = sum(1 for batch in batches if batch["target_exists"])
    blocked_count = sum(1 for batch in batches if batch["blocked"])
    idle_count = len(batches) - len(pending_batches) - generated_count - blocked_count
    summary = [f"待生成 {len(pending_batches)}", f"已生成 {generated_count}"]
    if idle_count:
        summary.append(f"未启用 {idle_count}")
    if blocked_count:
        summary.append(f"阻止 {blocked_count}")
    st.caption(f"共 {len(batches)} 个批次：" + "，".join(summary) + "。")

    selected_batch = None
    selected_date = None
    preview_clicked = False
    update_clicked = False
    batch_tab, all_tab = st.tabs(["单个日期更新", "全部生成"])
    with batch_tab:
        if not pending_batches:
            st.caption("没有待生成的批次。")
        else:
            date_counts = Counter(batch["date"] for batch in pending_batches)
            dates = sorted(date_counts)
            selected_date = st.selectbox(
                "更新日期",
                dates,
                format_func=lambda value: f"{value}（{date_counts[value]} 个待生成）",
            )
            date_batches = [batch for batch in pending_batches if batch["date"] == selected_date]
            selected_key = st.selectbox("预览批次", [batch["key"] for batch in date_batches])
            selected_batch = _selected_batch(date_batches, selected_key)
            actions = st.columns(2)
            preview_clicked = actions[0].button(
                "预览",
                key="cosmic_removal_preview",
                width="stretch",
            )
            update_clicked = actions[1].button(
                f"更新 {selected_date}",
                key="cosmic_removal_update_date",
                type="primary",
                width="stretch",
            )
    with all_tab:
        st.caption("一次生成 raw_data 下所有待生成批次；已生成的批次不覆盖。")
        generate_all_clicked = st.button(
            f"生成全部待生成批次（{len(pending_batches)}）",
            key="cosmic_removal_generate_all",
            type="primary",
            disabled=not pending_batches,
            width="stretch",
        )

    if preview_clicked and selected_batch is not None:
        try:
            with st.spinner("正在读取并清洗 raw_data 批次…"):
                report = run_folder_ingest(FolderIngestRequest(
                    spec=context.dataset_spec,
                    source_dir=selected_batch["source_dir"],
                    date_name=selected_batch["date"],
                    prefix=selected_batch["prefix"],
                    apply=False,
                    bead=selected_batch["is_bead"],
                ))
        except UI_ACTION_ERRORS as error:
            render_action_error("预览", error)
            st.session_state.pop("cosmic_removal_report", None)
        else:
            st.session_state["cosmic_removal_report"] = report

    targets: list[dict] = []
    if update_clicked:
        targets = [batch for batch in pending_batches if batch["date"] == selected_date]
    elif generate_all_clicked:
        targets = pending_batches
    if targets:
        failures = _generate_batches(context, targets)
        st.session_state.pop("cosmic_removal_report", None)
        done_count = len(targets) - len(failures)
        if failures:
            set_flash(
                "cosmic_removal",
                f"已生成 {done_count} 个批次，{len(failures)} 个失败：" + "；".join(failures),
                kind="warning",
            )
        else:
            set_flash("cosmic_removal", f"已生成 {done_count} 个批次。")
        invalidate_ui_cache(*INVALIDATE_ALL)
        st.rerun()

    report = st.session_state.get("cosmic_removal_report")
    if (
        selected_batch is None
        or not isinstance(report, dict)
        or not _report_matches(report, context, selected_batch)
    ):
        if isinstance(report, dict):
            st.session_state.pop("cosmic_removal_report", None)
        return
    st.caption(f"上次预览：{selected_batch['key']}")
    _render_preview(report)