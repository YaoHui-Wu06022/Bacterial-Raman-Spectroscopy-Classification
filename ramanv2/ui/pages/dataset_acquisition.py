"""全量 init 和固定 Profile 数据集获取页面。"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from ramanv2.data.count import summarize_dataset
from ramanv2.data.builders.init import (
    build_full_init,
    build_init_datasets,
    build_profile_dataset,
    update_compensated_init,
)
from ramanv2.data.calibration.affine import apply_folder_affine_compensation, load_calibration_report
from ramanv2.data.catalog import DatasetMapping, ensure_dataset_mapping
from ramanv2.common.naming import build_natural_key, is_test_source_folder, parse_folder_prefix, parse_test_folder_prefix
from ramanv2.data.plots.prefix import build_prefix_plots_for_groups
from ramanv2.data.profiles import PIPELINE_PROFILE_IDS, get_dataset_dir, get_profile
from ramanv2.pipeline.cleaning import reconcile_cleaning_records
from ramanv2.pipeline.lineage import initialize_folder_lineage, list_lineage_folders, remove_source_folder
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.cache import cached_approved_affine_dates, cached_dataset_summary
from ramanv2.ui.state import (
    invalidate_ui_cache,
    load_page_state as _load_state,
    save_page_state as _save_state,
    ui_cache_version,
)


def _load_catalog(context: PageContext) -> DatasetMapping:
    cached = st.session_state.get("dataset_mapping")
    if isinstance(cached, DatasetMapping):
        return cached
    catalog = ensure_dataset_mapping(
        context.classification_workbook,
        context.dataset_mapping_file,
    )
    st.session_state["dataset_mapping"] = catalog
    return catalog


def _render_count_metrics(summary: dict[str, int | bool]) -> None:
    metrics = st.columns(3)
    metrics[0].metric("属数量", f"{summary['genus_count']} 个")
    metrics[1].metric("来源文件夹", f"{summary['folder_count']} 个")
    metrics[2].metric("光谱数量", f"{summary['file_count']} 条")


def _render_source_folder_removal(
    context: PageContext,
    state,
    source_records: list[dict[str, object]],
    key: str,
    label: str,
) -> None:
    """显示源头整夹删除入口；删除从 cosmic_data 开始并同步清理下游。"""
    if not source_records:
        st.info("当前没有可移除的源头文件夹。")
        return
    source_ids = [str(item.get("source_id")) for item in source_records if item.get("source_id")]
    if not source_ids:
        st.info("当前没有可移除的源头文件夹。")
        return
    source_id = st.selectbox(label, source_ids, key=f"{key}_source_folder")
    if not st.button("从 cosmic 源头整夹删除", key=f"{key}_remove_source_folder", use_container_width=True):
        return
    try:
        result = remove_source_folder(
            context.data_root,
            source_id,
            shift_root=context.shift_root,
        )
        state.dataset_builds.pop("profiles", None)
        state.dataset_builds.pop("prefix_plots", None)
        invalidate_ui_cache("cleaning")
        invalidate_ui_cache("dataset")
        invalidate_ui_cache("training")
        invalidate_ui_cache("inference")
        _save_state(context, state)
        st.success(f"已从源头删除 {source_id}，并清理对应下游产物。")
        st.rerun()
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"源头整夹删除失败：{error}")


def _render_prefix_plot_section(
    context: PageContext,
    init_root: Path,
    figure_root: Path,
    key: str,
    state,
) -> None:
    """将已有图片查看和增量更新分开，默认不展开图像区域。"""
    if not init_root.is_dir():
        return
    with st.expander("prefix-plot", expanded=False):
        controls = st.columns(2)
        view_clicked = controls[0].button("查看已有图信息", key=f"{key}_view", use_container_width=True)
        update_clicked = controls[1].button("更新图信息", key=f"{key}_update", use_container_width=True)
        if view_clicked or update_clicked:
            st.session_state[f"{key}_show"] = True
        if update_clicked:
            try:
                outputs = build_prefix_plots(init_root, figure_root, force_enable=False)
                plots = state.dataset_builds.get("prefix_plots", {})
                if not isinstance(plots, dict):
                    plots = {}
                plots[key] = {"output_root": str(figure_root), "plot_count": len(outputs)}
                state.dataset_builds["prefix_plots"] = plots
                _save_state(context, state)
                st.success(f"已更新 {len(outputs)} 张 prefix-plot。")
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"prefix-plot 更新失败：{error}")
        if not st.session_state.get(f"{key}_show", False):
            return
        if not figure_root.is_dir():
            st.info("当前还没有 prefix-plot，请点击“更新图信息”。")
            return
        genus_paths = {
            path.name: sorted(path.glob("*.png"), key=lambda item: build_natural_key(item.name))
            for path in figure_root.iterdir()
            if path.is_dir() and any(path.glob("*.png"))
        }
        if not genus_paths:
            st.info("当前没有可查看的 prefix-plot。")
            return
        genus_name = st.selectbox("选择查看的属", sorted(genus_paths, key=build_natural_key), key=f"{key}_genus")
        plot_paths = genus_paths[genus_name]
        prefix_name = st.selectbox("选择查看的种前缀", [path.stem for path in plot_paths], key=f"{key}_prefix")
        selected = figure_root / genus_name / f"{prefix_name}.png"
        if selected.is_file():
            st.image(str(selected), caption=f"{genus_name} / {prefix_name}", use_container_width=True)
        source_records = list_lineage_folders(context.data_root, genus_name, prefix_name)
        _render_source_folder_removal(context, state, source_records, key, "选择源头文件夹")


def _render_affine_compensation(context: PageContext, state, catalog: DatasetMapping) -> None:
    """按文件夹输入补偿值，并从原始光谱重新发布校正日期。"""
    approved_dates = list(
        cached_approved_affine_dates(
            str(context.shift_root.resolve()),
            str(context.data_root.resolve()),
            ui_cache_version("dataset"),
        )
    )
    st.subheader("仿射校正补偿")
    if not approved_dates:
        st.info("当前没有可以补充仿射校正的日期。")
        return
    date_name = st.selectbox("选择日期", approved_dates, key="affine_compensation_date")
    report = load_calibration_report(context.shift_root, date_name) or {}
    source_date = context.data_root / date_name
    excluded = {"小球", "shift", "delete", "audit_runs", ".raman_pipeline"}
    folders = [
        path
        for path in sorted(source_date.iterdir(), key=lambda item: build_natural_key(item.name))
        if path.is_dir() and path.name not in excluded and any(path.rglob("*.arc_data"))
    ]
    if not folders:
        st.warning("该日期没有可补偿的普通采集文件夹。")
        return
    previous = report.get("folder_compensations", {})
    folder_names = [folder.name for folder in folders]
    selected_folder = st.selectbox(
        "选择补偿文件夹",
        folder_names,
        key=f"affine_compensation_folder_{date_name}",
    )
    value = st.number_input(
        f"{selected_folder} 补偿（cm⁻¹）",
        value=float(previous.get(selected_folder, 0.0)),
        step=0.1,
        format="%.1f",
        key=f"affine_compensation_{date_name}_{selected_folder}",
    )
    offsets = {selected_folder: float(value)}
    if st.button("应用文件夹补偿", key="apply_affine_compensation", use_container_width=True):
        try:
            result = apply_folder_affine_compensation(
                context.data_root,
                context.shift_root,
                date_name,
                offsets,
            )
            if not result.get("changed_folders"):
                st.info(f"{date_name}/{selected_folder} 的补偿值没有变化，未更新 init。")
                return

            if context.init_root.is_dir():
                profile_roots = {
                    profile_id: get_dataset_dir(
                        get_profile(profile_id),
                        context.init_root.parent.parent,
                    ) / "init"
                    for profile_id in PIPELINE_PROFILE_IDS
                }
                init_update = update_compensated_init(
                    context.shift_root,
                    date_name,
                    offsets,
                    context.init_root,
                    profile_roots,
                    catalog,
                    test_root=context.init_root.parent / "CSdata",
                )
                plot_roots = {"init": context.init_root, **profile_roots}
                prefix_updates: dict[str, int] = {}
                updated_roots = {
                    str(item.get("root"))
                    for item in init_update.get("updated_targets", [])
                    if isinstance(item, dict) and item.get("root")
                }
                for root_name in sorted(updated_roots):
                    init_root = plot_roots.get(root_name)
                    if init_root is None or not init_root.is_dir():
                        continue
                    group_keys = {
                        (
                            Path(str(item.get("path"))).parts[0],
                            parse_folder_prefix(Path(str(item.get("path"))).parts[1]),
                        )
                        for item in init_update.get("updated_targets", [])
                        if item.get("root") == root_name
                        and len(Path(str(item.get("path"))).parts) >= 2
                    }
                    outputs = build_prefix_plots_for_groups(
                        init_root,
                        init_root.parent / "fig_init",
                        group_keys,
                    )
                    prefix_updates[root_name] = len(outputs)
            else:
                init_update = {
                    "status": "skipped",
                    "reason": "init_missing",
                    "updated_targets": [],
                }
                prefix_updates = {}
            result["init_update"] = init_update
            result["prefix_updates"] = prefix_updates
            invalidate_ui_cache("dataset")
            invalidate_ui_cache("training")
            invalidate_ui_cache("inference")
            invalidate_ui_cache("analysis")
            compensation_reports = state.dataset_builds.get("affine_compensations", {})
            if not isinstance(compensation_reports, dict):
                compensation_reports = {}
            compensation_reports[date_name] = result
            state.dataset_builds["affine_compensations"] = compensation_reports
            _save_state(context, state)
            prefix_plot_count = sum(prefix_updates.values())
            if init_update["status"] == "ready":
                message = f"{date_name}/{selected_folder} 已记录补偿参数，并同步更新已有 init 和 {prefix_plot_count} 张 prefix-plot；shift_data 未改写。"
            else:
                message = f"{date_name}/{selected_folder} 已记录补偿参数；当前没有 init，后续生成数据集时会自动应用。"
            st.success(message)
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            st.error(f"仿射补偿失败：{error}")


@st.cache_data(show_spinner=False)
def _cached_cs_sources(shift_root: str, date_names: tuple[str, ...], version: int) -> tuple[str, ...]:
    """缓存日期下的 CS 候选目录扫描。"""
    del version
    root = Path(shift_root)
    candidates: list[str] = []
    for date_name in date_names:
        date_dir = root / date_name
        if not date_dir.is_dir():
            continue
        for folder in sorted(date_dir.iterdir(), key=lambda item: build_natural_key(item.name)):
            if folder.is_dir() and is_test_source_folder(folder.name) and any(folder.glob("*.arc_data")):
                candidates.append(f"{date_name}/{folder.name}")
    return tuple(candidates)


def _list_cs_sources(context: PageContext, date_names: list[str]) -> list[str]:
    """列出指定日期中可放入 CSdata 的 CS 文件夹。"""
    return list(
        _cached_cs_sources(
            str(context.shift_root.resolve()),
            tuple(sorted(date_names, key=build_natural_key)),
            ui_cache_version("dataset"),
        )
    )


def _default_unmapped_cs_sources(candidates: list[str], catalog: DatasetMapping | None) -> list[str]:
    """默认选择没有规范前缀映射的 CS 目录。"""
    mapped_prefixes = {
        prefix
        for prefixes in (catalog.genera.values() if catalog is not None else ())
        for prefix in prefixes
    }
    return sorted(
        [
            candidate
            for candidate in candidates
            if parse_test_folder_prefix(candidate.rsplit("/", 1)[-1]) not in mapped_prefixes
        ],
        key=build_natural_key,
    )


def _sync_cs_selection(
    candidates: list[str],
    catalog: DatasetMapping | None,
    selection_key: str,
    initialized_key: str,
) -> None:
    """保留用户选择，并只为新出现的未映射 CS 目录添加默认选择。"""
    previous_candidates_key = f"{selection_key}_candidates"
    previous_candidates = set(st.session_state.get(previous_candidates_key, []))
    if not st.session_state.get(initialized_key, False):
        selected = _default_unmapped_cs_sources(candidates, catalog)
    else:
        selected = [
            value
            for value in st.session_state.get(selection_key, [])
            if value in candidates
        ]
        new_candidates = sorted(set(candidates) - previous_candidates, key=build_natural_key)
        selected.extend(
            value
            for value in _default_unmapped_cs_sources(new_candidates, catalog)
            if value not in selected
        )
    st.session_state[selection_key] = selected
    st.session_state[previous_candidates_key] = list(candidates)
    st.session_state[initialized_key] = True


def _merge_incremental_init_report(
    previous: dict[str, object],
    update: dict[str, object],
    init_root: Path,
) -> dict[str, object]:
    """把新增日期报告合并到全量 init 的累计报告。"""
    merged = dict(previous)
    for key in ("source_dates", "eligible_dates"):
        merged[key] = sorted(
            set(previous.get(key, [])) | set(update.get(key, [])),
            key=build_natural_key,
        )
    for key in ("skipped_dates",):
        merged[key] = sorted(
            set(previous.get(key, [])) | set(update.get(key, [])),
            key=build_natural_key,
        )
    for key in ("source_files", "output_files", "init_source_files", "init_output_files", "test_files", "test_output_files", "cs_source_files", "cs_output_files"):
        merged[key] = int(previous.get(key, 0)) + int(update.get(key, 0))
    date_status = dict(previous.get("date_status", {}))
    date_status.update(update.get("date_status", {}))
    merged["date_status"] = date_status
    date_stats = dict(previous.get("date_stats", {}))
    date_stats.update(update.get("date_stats", {}))
    merged["date_stats"] = date_stats
    merged["test_folders"] = sorted(
        set(previous.get("test_folders", [])) | set(update.get("test_folders", [])),
        key=build_natural_key,
    )
    merged["cs_folder_count"] = len(merged["test_folders"])
    merged["selected_cs_folders"] = sorted(
        set(previous.get("selected_cs_folders", [])) | set(update.get("selected_cs_folders", [])),
        key=build_natural_key,
    )
    merged["cs_exports"] = [*previous.get("cs_exports", []), *update.get("cs_exports", [])]
    merged["count"] = summarize_dataset(init_root)
    merged["mode"] = "full"
    merged["status"] = "ready"
    return merged


def render_dataset_acquisition_page(context: PageContext) -> None:
    state = _load_state(context)
    initialize_folder_lineage(context.data_root)
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        _save_state(context, state)

    st.title("数据集获取")
    st.caption("先生成全量 dataset/init，再选择固定 Profile 生成对应数据集。")

    try:
        catalog = _load_catalog(context)
    except (OSError, ValueError, RuntimeError) as error:
        catalog = None
        st.error(f"Profile 配置加载失败：{error}")
    if catalog is not None:
        initialize_folder_lineage(context.data_root, catalog)

    st.subheader("全量 init")
    full_summary = cached_dataset_summary(
        str(context.init_root.resolve()), ui_cache_version("dataset")
    )
    full_report = state.dataset_builds.get("full_init")
    full_init_ready = (
        bool(full_summary["exists"])
        and isinstance(full_report, dict)
        and full_report.get("status") == "ready"
    )
    st.metric("状态", "已生成" if full_summary["exists"] else "未生成")
    _render_count_metrics(full_summary)
    eligible_dates = sorted(state.audit_completed_dates & state.affine_completed_dates, key=build_natural_key)
    cs_candidates = _list_cs_sources(context, eligible_dates)
    cs_selection_key = "full_init_cs_selection"
    initialized_key = "full_init_cs_selection_initialized"
    _sync_cs_selection(cs_candidates, catalog, cs_selection_key, initialized_key)
    selected_cs = st.multiselect(
        "选择进入 CSdata 的 CS 文件夹",
        cs_candidates,
        format_func=lambda value: value.replace("/", " / "),
        key=cs_selection_key,
    )
    if st.button("从 shift_data 重建全量 init", disabled=catalog is None, use_container_width=True):
        try:
            with st.status("正在重建全量 init…", expanded=True) as status:
                status.write("正在校验日期门禁并复制普通光谱。")
                status.write("正在按选择更新 CSdata。")
                report = build_full_init(
                    context.shift_root,
                    context.init_root,
                    catalog,
                    eligible_dates,
                    test_root=context.init_root.parent / "CSdata",
                    selected_cs_folders=selected_cs,
                )
                status.update(label="全量 init 重建完成", state="complete", expanded=False)
            state.dataset_builds["full_init"] = report
            state.dataset_builds.pop("profiles", None)
            state.dataset_builds.pop("prefix_plots", None)
            invalidate_ui_cache("dataset")
            invalidate_ui_cache("training")
            invalidate_ui_cache("inference")
            invalidate_ui_cache("analysis")
            _save_state(context, state)
            st.success("全量 init 生成完成，旧 Profile 构建记录已清除。")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"全量 init 生成失败：{error}")

    if full_init_ready:
        st.success("全量 init 已生成；后续新增日期可以增量更新。")
        processed_dates = set(full_report.get("source_dates", []))
        pending_dates = [date_name for date_name in eligible_dates if date_name not in processed_dates]
        if pending_dates:
            st.subheader("增量更新 init")
            st.caption("只处理尚未进入全量 init 的新日期，不重复构建已有数据。")
            st.multiselect(
                "选择新增日期",
                pending_dates,
                default=pending_dates,
                key="incremental_init_dates",
            )
            incremental_dates = st.session_state.get("incremental_init_dates", pending_dates)
            cs_candidates = _list_cs_sources(context, incremental_dates)
            cs_selection_key = "incremental_init_cs_selection"
            _sync_cs_selection(
                cs_candidates,
                catalog,
                cs_selection_key,
                "incremental_init_cs_selection_initialized",
            )
            selected_cs = st.multiselect(
                "新增 CS 中选择进入 CSdata 的文件夹",
                cs_candidates,
                format_func=lambda value: value.replace("/", " / "),
                key=cs_selection_key,
            )
            if st.button("增量更新 init", disabled=catalog is None, use_container_width=True):
                try:
                    with st.status("正在增量更新 init…", expanded=True) as status:
                        status.write("正在检查新增日期和目标目录。")
                        update = build_init_datasets(
                            context.shift_root,
                            context.init_root,
                            {},
                            catalog,
                            incremental_dates,
                            mode="incremental",
                            test_root=context.init_root.parent / "CSdata",
                            selected_test_folders=selected_cs,
                        )
                        status.update(label="增量更新完成", state="complete", expanded=False)
                    state.dataset_builds["full_init"] = _merge_incremental_init_report(
                        full_report,
                        update,
                        context.init_root,
                    )
                    state.dataset_builds.pop("profiles", None)
                    state.dataset_builds.pop("prefix_plots", None)
                    invalidate_ui_cache("dataset")
                    invalidate_ui_cache("training")
                    invalidate_ui_cache("inference")
                    invalidate_ui_cache("analysis")
                    _save_state(context, state)
                    st.success("init 已完成增量更新。")
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as error:
                    st.error(f"init 增量更新失败：{error}")
        else:
            st.info("当前没有尚未进入 init 的新日期。")

    if catalog is None:
        st.info("请先修复本地病原菌分类 Excel 文件。")
        return

    _render_affine_compensation(context, state, catalog)

    with st.expander("源头文件夹移除", expanded=False):
        st.caption("整夹移除从 cosmic_data 开始，并同步清理 shift、审核、init、Profile 和 prefix 产物。")
        _render_source_folder_removal(
            context,
            state,
            list_lineage_folders(context.data_root),
            "all_source_folders",
            "选择要移除的源头文件夹",
        )

    st.subheader("Profile 数据集")
    profile_name = st.selectbox("选择 Profile", PIPELINE_PROFILE_IDS, key="dataset_profile_name")
    profile = get_profile(profile_name)
    profile_root = get_dataset_dir(profile, context.init_root.parent.parent) / profile.root_init
    profile_summary = cached_dataset_summary(
        str(profile_root.resolve()), ui_cache_version("dataset")
    )
    _render_count_metrics(profile_summary)
    profile_reports = state.dataset_builds.get("profiles", {})
    if not isinstance(profile_reports, dict):
        profile_reports = {}

    if st.button(
        "从 init 生成数据集",
        type="primary",
        disabled=not full_init_ready,
        use_container_width=True,
    ):
        try:
            with st.status(f"正在生成 {profile_name} 数据集…", expanded=True) as status:
                status.write("正在读取全量 init 并生成 Profile 子集。")
                report = build_profile_dataset(context.init_root, profile_root, profile_name, catalog)
                status.update(label=f"{profile_name} 数据集生成完成", state="complete", expanded=False)
            profile_reports[profile_name] = report
            state.dataset_builds["profiles"] = profile_reports
            invalidate_ui_cache("dataset")
            invalidate_ui_cache("training")
            invalidate_ui_cache("inference")
            invalidate_ui_cache("analysis")
            _save_state(context, state)
            st.success(f"{profile_name} 数据集生成完成。")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"{profile_name} 数据集生成失败：{error}")

    _render_prefix_plot_section(
        context,
        profile_root,
        profile_root.parent / "fig_init",
        f"profile_{profile_name}",
        state,
    )


__all__ = ["render_dataset_acquisition_page"]
