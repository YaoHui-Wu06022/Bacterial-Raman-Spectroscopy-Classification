"""全量 init 和固定 Profile 数据集获取页面。"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from ramanv2.common.arc_data import count_arc_data, has_arc_data
from ramanv2.core.config import InputConfig
from ramanv2.common.naming import (
    CANONICAL_FOLDER_PATTERN,
    SPECIAL_FOLDER_NAMES,
    build_natural_key,
    parse_folder_prefix,
)
from ramanv2.data.builders.init import build_full_init, build_profile_dataset
from ramanv2.data.plots.prefix import PrefixPlotConfig, build_prefix_plots
from ramanv2.pipeline.calibration import run_folder_offset
from ramanv2.data.datasets import annotate_dataset_build
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.pipeline.dataset_sync import (
    collect_date_update_candidates,
    collect_init_export_sources,
    collect_test_return_sources,
    run_date_dataset_update,
    run_init_to_test,
    run_test_to_init,
)
from ramanv2.records.lineage import clear_test_routes, lineage_path, load_folder_lineage
from ramanv2.pipeline.supplement import (
    SupplementPlan,
    build_full_init_with_supplements,
    scan_supplement_plan,
    validate_supplement_plan,
)
from ramanv2.pipeline.state import PipelineState
from ramanv2.pipeline.db import database_path, load_test_defaults
from ramanv2.ui.components.pickers import render_profile_picker
from ramanv2.ui.formatting import progress_ratio
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.read_cache import cached_dataset_summary, cached_dataset_taxon_counts
from ramanv2.ui.page_state import (
    INVALIDATE_DATASET,
    INVALIDATE_SPECTRA,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    load_page_state,
    render_action_error,
    render_flash,
    set_flash,
    write_page_state,
    ui_cache_version,
)


# 候选源文件夹：流水线保留目录与校准球都不作为普通来源。
SOURCE_RESERVED_NAMES = SPECIAL_FOLDER_NAMES


def _render_count_metrics(summary: dict[str, int | bool], test_summary: dict[str, int | bool]) -> None:
    """标出 init 与 init_test 的分配，避免把 init 单独的光谱数误认为数据丢失。"""
    metrics = st.columns(3)
    metrics[0].metric("属数量", f"{summary['genus_count']} 个")
    metrics[1].metric("种数量", f"{summary['species_count']} 个")
    metrics[2].metric("init 光谱数量", f"{summary['file_count']} 条")
    if test_summary["exists"]:
        st.caption(
            f"init_test 另有 {test_summary['folder_count']} 个文件夹、{test_summary['file_count']} 条；"
            f"两处合计 {summary['file_count'] + test_summary['file_count']} 条。"
        )


def _render_taxon_counts(init_root: Path) -> None:
    """展示当前 init 的属汇总及种前缀明细，普通交互复用缓存。"""
    with st.expander("属／种数量统计", expanded=False):
        counts = cached_dataset_taxon_counts(str(init_root.resolve()), ui_cache_version("dataset"))
        if not counts["species"]:
            st.info("当前 init 中没有可统计的光谱。")
            return
        st.caption("仅统计当前 init；种按规范前缀区分，同一种跨日期合并。init_test 不计入。")
        genus_tab, species_tab = st.tabs(["各属汇总", "各种明细"])
        with genus_tab:
            st.dataframe([
                {"属": row["genus"], "种数": row["species_count"], "来源文件夹数": row["folder_count"], "光谱数": row["file_count"]}
                for row in counts["genera"]
            ], hide_index=True, width="stretch")
        with species_tab:
            st.dataframe([
                {"属": row["genus"], "种前缀": row["prefix"], "来源文件夹数": row["folder_count"], "光谱数": row["file_count"]}
                for row in counts["species"]
            ], hide_index=True, width="stretch")


def _render_prefix_plot_section(
    init_root: Path,
    figure_root: Path,
    key: str,
    input_config=None,
    flat: bool = False,
) -> None:
    """将已有图片查看和增量更新分开，默认不展开图像区域。"""
    if not init_root.is_dir():
        return
    with st.expander("prefix-plot", expanded=False):
        controls = st.columns(2)
        view_clicked = controls[0].button("查看已有图信息", key=f"{key}_view", width="stretch")
        update_clicked = controls[1].button("更新图信息", key=f"{key}_update", width="stretch")
        if view_clicked or update_clicked:
            st.session_state[f"{key}_show"] = True
        if update_clicked:
            try:
                with st.status("正在更新 prefix-plot…", expanded=True) as status:
                    progress = st.progress(0.0, text="正在扫描 init 文件夹…")

                    def report_plot_progress(genus: str, prefix: str, done: int, total: int) -> None:
                        """按已处理的分组数刷新绘图进度。"""
                        progress.progress(
                            progress_ratio(done, total),
                            text=f"[{done}/{total}] 已绘制 {genus} / {prefix}",
                        )

                    outputs = build_prefix_plots(
                        init_root,
                        figure_root,
                        force_enable=False,
                        config=PrefixPlotConfig(input=input_config, flat=flat) if input_config is not None else PrefixPlotConfig(flat=flat),
                        progress_callback=report_plot_progress,
                    )
                    progress.progress(1.0, text=f"已更新 {len(outputs)} 张 prefix-plot")
                    status.update(label="prefix-plot 更新完成", state="complete", expanded=False)
                st.success(f"已更新 {len(outputs)} 张 prefix-plot。")
            except UI_ACTION_ERRORS as error:
                render_action_error("prefix-plot 更新", error)
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
        genus_name = st.selectbox(
            "选择查看的类别" if flat else "选择查看的属",
            sorted(genus_paths, key=build_natural_key),
            key=f"{key}_genus",
        )
        plot_paths = genus_paths[genus_name]
        if flat:
            stems = [path.stem for path in plot_paths]
            prefix_name = genus_name if genus_name in stems else stems[0]
            caption = genus_name
        else:
            prefix_name = st.selectbox("选择查看的种前缀", [path.stem for path in plot_paths], key=f"{key}_prefix")
            caption = f"{genus_name} / {prefix_name}"
        selected = figure_root / genus_name / f"{prefix_name}.png"
        if selected.is_file():
            st.image(str(selected), caption=caption, width="stretch")


def _render_supplement_plan(plan: SupplementPlan | None, supplement_root: Path) -> None:
    """展示补充数据预检表；计划缺失或出错时由上游禁用重建。"""
    if plan is None:
        return
    st.markdown("**补充数据**")
    st.caption(f"来源目录：{supplement_root}；重建后按文件夹写入 dataset/init/<属>/<文件夹>。")
    if plan.groups:
        st.dataframe(
            [
                {"文件夹": group.folder_name, "属": group.genus, "光谱数": group.file_count}
                for group in plan.groups
            ],
            hide_index=True,
            width="stretch",
        )
    elif not plan.errors:
        st.info("未发现可写入的补充文件夹。")
    for error in plan.errors:
        st.error(error)
    if plan.unbacked_init_folders:
        st.warning(
            "以下 init 直放组在补充数据中没有对应文件夹，重建后会被移除："
            + "、".join(plan.unbacked_init_folders)
        )


def _report_cache_key(state) -> str:
    report = state.dataset_builds.get("full_init")
    return json.dumps(report, ensure_ascii=False, sort_keys=True, default=str) if isinstance(report, dict) else ""


@st.cache_data(show_spinner=False)
def _cached_init_export_sources(
    data_root: str,
    init_root: str,
    report_json: str,
    lineage_modified_ns: int,
    version: int,
) -> dict[str, str]:
    """缓存 init 移入 init_test 的候选；走向文件或数据集版本变化时失效。"""
    del lineage_modified_ns, version
    report = json.loads(report_json) if report_json else None
    state = PipelineState(
        dataset_builds={"full_init": report} if isinstance(report, dict) else {},
    )
    return collect_init_export_sources(Path(data_root), Path(init_root), state)


_DATE_UPDATE_STATUS_LABELS = {
    "new": "新增",
    "shift_missing": "shift 待更新",
    "changed": "待更新",
    "ready": "已最新",
}


@st.cache_data(show_spinner=False)
def _cached_date_update_candidates(
    data_root: str,
    shift_root: str,
    init_root: str,
    report_json: str,
    lineage_modified_ns: int,
    version: int,
) -> tuple[dict[str, object], ...]:
    """缓存日期检测结果；报告、走向文件或写入版本变化时失效。"""
    del lineage_modified_ns, version
    report = json.loads(report_json) if report_json else {}
    return tuple(collect_date_update_candidates(Path(data_root), Path(shift_root), Path(init_root), report))


def _load_stale_test_routes(context: PageContext) -> list[tuple[str, str]]:
    """找出仍登记为 test、但测试目录已缺失且不在默认名单的文件夹。"""
    configured = set(_load_configured_default_sources(context) or [])
    if not configured:
        return []
    folders = load_folder_lineage(context.data_root).get("folders", {})
    if not isinstance(folders, dict):
        return []
    stale: list[tuple[str, str]] = []
    for source_id, record in folders.items():
        if not isinstance(record, dict) or source_id in configured:
            continue
        outputs = record.get("outputs")
        test_path = outputs.get("test") if isinstance(outputs, dict) else None
        if not test_path:
            continue
        target = Path(str(test_path))
        if not target.is_dir() or not has_arc_data(target):
            stale.append((str(source_id), str(test_path)))
    return sorted(stale, key=lambda item: build_natural_key(item[0]))


def _render_stale_test_routes(context: PageContext) -> None:
    """清理指向缺失目录的历史 test 分流，避免日期更新永远停在待更新。"""
    stale = _load_stale_test_routes(context)
    if not stale:
        return
    with st.expander(f"发现 {len(stale)} 个陈旧 test 分流（目录不存在）", expanded=False):
        st.caption(
            "这些文件夹仍登记为 init_test，但测试目录已不存在，导致日期检测一直显示待更新。"
            "清理只改分流记录，不删除 init 数据。"
        )
        for source_id, target in stale[:20]:
            st.write(f"- {source_id} → `{target}`")
        if len(stale) > 20:
            st.caption(f"另有 {len(stale) - 20} 个未显示。")
        confirmed = st.checkbox(
            "我确认清除这些 test 分流",
            key="stale_test_route_confirm",
        )
        if st.button(
            "清除陈旧 test 分流",
            disabled=not confirmed,
            width="stretch",
            key="stale_test_route_clear",
        ):
            result = clear_test_routes(
                context.data_root,
                [source_id for source_id, _target in stale],
            )
            invalidate_ui_cache(*INVALIDATE_DATASET)
            set_flash("dataset", f"已清除 {result['count']} 个 test 分流；init 数据未改动。")
            st.rerun()


def _render_date_dataset_update(context: PageContext, state, catalog: DatasetMapping | None) -> None:
    """自动检测新增与变化日期，也可手动勾选；只重建所选日期的组。"""
    report = state.dataset_builds.get("full_init")
    if not isinstance(report, dict) or not context.init_root.is_dir():
        return
    with st.expander("按日期更新", expanded=False):
        render_flash("date_update")
        lineage_file = lineage_path(context.data_root)
        candidates = _cached_date_update_candidates(
            str(context.data_root.resolve()),
            str(context.shift_root.resolve()),
            str(context.init_root.resolve()),
            json.dumps(report, ensure_ascii=False, sort_keys=True, default=str),
            lineage_file.stat().st_mtime_ns if lineage_file.is_file() else 0,
            ui_cache_version("dataset"),
        )
        if not candidates:
            st.info("当前没有可更新的日期：请先完成日期仿射校正并生成 shift_data。")
            return
        st.caption(
            "自动检测新增、待同步和 shift 文件变化的日期并默认勾选；已有日期沿用原 init/init_test 分流，"
            "同时重建两处产物；新增日期可在下方选择进入 init_test 的文件夹，其余进入 init。"
            "其他日期编号保持不变。已有 Profile 需另行重新生成。"
        )
        st.dataframe(
            [
                {
                    "日期": item["date"],
                    "状态": _DATE_UPDATE_STATUS_LABELS.get(str(item["status"]), str(item["status"])),
                    "来源文件夹": item["folder_count"],
                    "init / init_test": (
                        "待选" if item["is_new"]
                        else f"{item['init_folder_count']} / {item['test_folder_count']}"
                    ),
                    "光谱": item["file_count"],
                    "待同步": len(item["pending_folders"]),
                    "文件变化": len(item["drifted_folders"]),
                    "shift 缺失": len(item["shift_missing_folders"]),
                }
                for item in candidates
            ],
            hide_index=True,
            width="stretch",
        )
        _render_stale_test_routes(context)
        detected = [str(item["date"]) for item in candidates if item["status"] in {"new", "changed"}]
        signature = tuple(
            (
                item["date"], item["status"], item["folder_count"],
                item["pending_folders"], item["drifted_folders"], item["shift_missing_folders"],
            )
            for item in candidates
        )
        if st.session_state.get("date_update_signature") != signature:
            st.session_state["date_update_dates"] = detected
            st.session_state["date_update_signature"] = signature
        selected_dates = st.multiselect(
            "选择要更新的日期",
            [str(item["date"]) for item in candidates],
            key="date_update_dates",
        )
        new_date_names = {str(item["date"]) for item in candidates if item["is_new"]}
        selected_new_dates = [date_name for date_name in selected_dates if date_name in new_date_names]
        selected_test: list[str] = []
        if selected_new_dates:
            test_candidates = _list_test_sources(context, selected_new_dates)
            _sync_test_selection(
                context,
                test_candidates,
                "date_update_test_selection",
                "date_update_test_selection_initialized",
            )
            _render_default_test_selection_button(
                context,
                test_candidates,
                "date_update_test_selection",
            )
            selected_test = st.multiselect(
                "新增日期中选择进入 init_test 的文件夹",
                test_candidates,
                format_func=lambda value: value.replace("/", " / "),
                key="date_update_test_selection",
            )
        if not st.button(
            "从 shift_data 更新所选日期",
            disabled=catalog is None or not selected_dates,
            width="stretch",
        ):
            return
        try:
            with st.status("正在按日期更新…", expanded=True) as status:
                progress = st.progress(0.0, text="正在重建所选日期的组…")

                def report_progress(date_name: str, processed: int, total: int) -> None:
                    """按当前构建批次的复制进度刷新进度条。"""
                    progress.progress(
                        progress_ratio(processed, total),
                        text=f"[{processed}/{total}] 正在同步 {date_name}…",
                    )

                result = run_date_dataset_update(
                    context.data_root,
                    context.shift_root,
                    context.init_root,
                    catalog,
                    state,
                    context.state_file,
                    selected_dates,
                    selected_test,
                    hierarchy=context.hierarchy,
                    progress_callback=report_progress,
                )
                progress.progress(1.0, text="init 与 init_test 写入完成")
                status.update(label="按日期更新完成", state="complete", expanded=False)
            invalidate_ui_cache(*INVALIDATE_DATASET)
            message = (
                f"已更新 {len(result['updated_dates'])} 个日期"
                f"（新增 {len(result['new_dates'])}，同步 {len(result['changed_dates'])}），"
                f"共 {result['spectrum_count']} 条光谱。"
            )
            if result["plot_error"]:
                message += f" prefix-plot 更新失败：{result['plot_error']}"
            set_flash("date_update", message)
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error("按日期更新", error)


def _render_dataset_transfer(context: PageContext, state, catalog: DatasetMapping | None, to_test_enable: bool) -> None:
    """双向迁移只提交所选目录，普通控件交互不写入数据。"""
    key = "init_export" if to_test_enable else "test_return"
    title = "init 快速移入 init_test" if to_test_enable else "init_test 快速移回 init"
    destination = "init_test" if to_test_enable else "init"
    with st.expander(title, expanded=False):
        render_flash(key)
        if to_test_enable:
            lineage_file = lineage_path(context.data_root)
            sources = _cached_init_export_sources(
                str(context.data_root.resolve()),
                str(context.init_root.resolve()),
                _report_cache_key(state),
                lineage_file.stat().st_mtime_ns if lineage_file.is_file() else 0,
                ui_cache_version("dataset"),
            )
        else:
            sources = {source: source for source in collect_test_return_sources(context.init_root, state)}
        if not sources:
            st.info(f"当前没有带来源记录、可迁移的 {'init' if to_test_enable else 'init_test'} 文件夹。")
            return
        selected = st.selectbox("选择移入 init_test 的 init 文件夹" if to_test_enable else "选择移回的 init_test 文件夹",
                                list(sources), format_func=lambda value: sources[value], key=f"{key}_source")
        prefix = parse_folder_prefix(selected.rsplit("/", 1)[-1], uppercase_enable=True)
        genus = next((name for name, prefixes in catalog.genera.items() if prefix in prefixes), None) if catalog is not None else None
        if genus is None:
            st.info(f"种前缀 {prefix} 未映射到属，请先补充分类名单。")
        else:
            caption = "整夹移入 init_test；普通来源保留现有文件夹名。" if to_test_enable else f"整夹移入 init/{genus}，按 {prefix} 和采集日期追加编号。"
            st.caption(caption + "仅更新对应审核图，已有 Profile 不变。")
        label = "将所选 init 直接移入 init_test" if to_test_enable else "将所选 init_test 直接移回 init"
        if not st.button(label, key=f"{key}_confirm", disabled=genus is None, width="stretch"):
            return
        try:
            with st.spinner("正在迁移所选文件夹并更新分流记录……"):
                run_transfer = run_init_to_test if to_test_enable else run_test_to_init
                result = run_transfer(context.data_root, context.init_root, catalog, state, context.state_file, selected)
            invalidate_ui_cache(*INVALIDATE_DATASET)
            st.session_state["test_export_added_source" if to_test_enable else "test_return_removed_source"] = selected
            set_flash(
                key,
                f"{selected} 已直接{'移入' if to_test_enable else '移回'} {destination}"
                f"（{result['spectrum_count']} 条）：{Path(result['target_dir']).name}。",
            )
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error(f"迁移到 {destination}", error)


def _render_affine_compensation(context: PageContext, state, catalog: DatasetMapping) -> None:
    """输入只表示本次 cosmic 原轴增量，提交成功或切换文件夹后归零。"""
    st.subheader("补充频移")
    if st.session_state.pop("folder_offset_reset", False):
        st.session_state["folder_offset_delta"] = 0.0
    render_flash("folder_offset")
    folders = list(
        _cached_date_source_folders(
            str(context.data_root.resolve()),
            tuple(sorted(state.affine_completed_dates)),
            (
                ui_cache_version("cleaning"),
                ui_cache_version("affine"),
                ui_cache_version("dataset"),
            ),
        )
    )
    if not folders or not context.init_root.is_dir():
        st.info("请先完成仿射校正并生成全量 init。")
        return
    selected = st.selectbox("选择文件夹", folders, key="folder_offset_source")
    if st.session_state.get("folder_offset_previous_source") != selected:
        st.session_state["folder_offset_delta"] = 0.0
        st.session_state["folder_offset_previous_source"] = selected
    delta = st.number_input("本次频移量（cm⁻¹）", step=0.1, format="%.1f", key="folder_offset_delta")
    if st.button("应用本次频移并同步数据", key="apply_affine_compensation", width="stretch"):
        try:
            result = run_folder_offset(
                context.data_root,
                context.shift_root,
                context.init_root,
                catalog,
                state,
                context.state_file,
                selected,
                float(delta),
                context.fig_root,
            )
            if not result["changed_folders"]:
                st.info("本次频移量为 0，未改动光谱。")
                return
            invalidate_ui_cache(*INVALIDATE_SPECTRA)
            st.session_state["folder_offset_reset"] = True
            destination = "对应 init_test 文件夹" if result.get("test_updated") else "对应 init 文件夹及审核图"
            set_flash("folder_offset", f"{selected} 已同步平移活动谱及归档谱，并更新{destination}。")
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error("频移补偿", error)


@st.cache_data(show_spinner=False)
def _cached_date_source_folders(
    root: str,
    date_names: tuple[str, ...],
    cache_version: int | tuple[int, int, int],
) -> tuple[str, ...]:
    """缓存日期下可用的来源文件夹；调用方用页面版本控制失效。"""
    del cache_version
    base = Path(root)
    folders: list[str] = []
    for date_name in date_names:
        date_dir = base / date_name
        if not date_dir.is_dir():
            continue
        for folder in sorted(date_dir.iterdir(), key=lambda item: build_natural_key(item.name)):
            if folder.is_dir() and folder.name not in SOURCE_RESERVED_NAMES and has_arc_data(folder):
                folders.append(f"{date_name}/{folder.name}")
    return tuple(folders)


def _list_test_sources(context: PageContext, date_names: list[str]) -> list[str]:
    """列出指定日期中可放入 init_test 的测试文件夹。"""
    return list(
        _cached_date_source_folders(
            str(context.shift_root.resolve()),
            tuple(sorted(date_names, key=build_natural_key)),
            ui_cache_version("dataset"),
        )
    )


# 默认选择逻辑版本号，修改逻辑时递增以强制重新初始化
_TEST_SELECTION_LOGIC_VERSION = 15


def _load_configured_default_sources(context: PageContext) -> list[str] | None:
    """读取当前数据集的推荐默认名单。"""
    db_path = database_path(context.metadata_dir)
    if not db_path.is_file():
        return None
    try:
        folders = load_test_defaults(db_path)
    except UI_ACTION_ERRORS:
        return None
    return folders or None


def _default_test_sources(context: PageContext, candidates: list[str]) -> list[str]:
    """按当前数据集的默认名单过滤候选文件夹。"""
    configured = _load_configured_default_sources(context)
    if configured is None:
        return []
    available = set(candidates)
    return sorted(
        [candidate for candidate in configured if candidate in available],
        key=build_natural_key,
    )


def _sync_test_selection(
    context: PageContext,
    candidates: list[str],
    selection_key: str,
    initialized_key: str,
) -> None:
    """按当前候选集补齐默认选择，同时保留用户显式移除的默认项。"""
    version_key = f"{selection_key}_version"
    defaults_key = f"{selection_key}_defaults"
    suppressed_key = f"{selection_key}_suppressed_defaults"
    version_changed = st.session_state.get(version_key) != _TEST_SELECTION_LOGIC_VERSION
    initialized = st.session_state.get(initialized_key, False)
    previous_defaults = set(st.session_state.get(defaults_key, []))
    current_candidates = set(candidates)
    default_sources = _default_test_sources(context, candidates)

    # 首次初始化或默认规则升级时直接采用当前规则的默认选择。
    if not initialized or version_changed:
        selected = default_sources
        suppressed_defaults: set[str] = set()
    else:
        selected = [
            value
            for value in st.session_state.get(selection_key, [])
            if value in candidates
        ]
        suppressed_defaults = set(st.session_state.get(suppressed_key, []))
        # 只记录仍然存在但被用户从选择中移除的旧默认项；候选已消失时，
        # 允许候选变化后由新的默认文件夹接替它。
        suppressed_defaults.update(
            value
            for value in previous_defaults & current_candidates
            if value not in selected
        )
        suppressed_defaults.difference_update(selected)

    defaults = set(default_sources)
    selected_set = set(selected)
    selected_set.update(defaults - suppressed_defaults)
    st.session_state[selection_key] = sorted(selected_set, key=build_natural_key)
    st.session_state[defaults_key] = sorted(defaults, key=build_natural_key)
    st.session_state[suppressed_key] = sorted(suppressed_defaults, key=build_natural_key)
    st.session_state[version_key] = _TEST_SELECTION_LOGIC_VERSION
    st.session_state[initialized_key] = True


def _render_default_test_selection_button(
    context: PageContext,
    candidates: list[str],
    selection_key: str,
) -> None:
    """提供显式的默认选择恢复入口，避免误用多选框的全选操作。"""
    if not st.button(
        "恢复默认选择",
        key=f"{selection_key}_reset_defaults",
        help="按 SQLite 中保存的推荐默认名单重新选择。",
    ):
        return
    defaults = _default_test_sources(context, candidates)
    st.session_state[selection_key] = defaults
    st.session_state[f"{selection_key}_defaults"] = defaults
    st.session_state[f"{selection_key}_suppressed_defaults"] = []
    st.rerun()


def _build_report_stale(context: PageContext, report: object) -> bool:
    """构建报告中的层级与当前数据集配置不一致时视为待重建。"""
    if not isinstance(report, dict):
        return False
    signature = report.get("hierarchy_signature")
    return signature is not None and signature != context.dataset_spec.hierarchy_signature()


def _render_flat_label_counts(context: PageContext) -> None:
    """按层级一级类别汇总 init 的组数与光谱数。"""
    if not context.init_root.is_dir():
        return
    counts: dict[str, list[int]] = {}
    for folder in sorted(context.init_root.rglob("*"), key=lambda path: build_natural_key(path.as_posix())):
        if not folder.is_dir() or not CANONICAL_FOLDER_PATTERN.fullmatch(folder.name):
            continue
        try:
            label = context.hierarchy.resolve_values(folder.name)["level_1"]
        except (KeyError, ValueError):
            label = folder.parent.name if folder.parent != context.init_root else folder.name
        entry = counts.setdefault(label, [0, 0])
        entry[0] += 1
        entry[1] += count_arc_data(folder)
    rows = [
        {"类别": label, "来源组数": values[0], "光谱数": values[1]}
        for label, values in sorted(counts.items(), key=lambda item: build_natural_key(item[0]))
    ]
    with st.expander("类别数量统计", expanded=False):
        st.dataframe(rows, hide_index=True, width="stretch")


def render_dataset_acquisition_page(context: PageContext) -> None:
    state = load_page_state(context)
    st.title("数据集获取")
    render_flash("dataset")

    try:
        catalog = context.dataset_mapping()
    except UI_ACTION_ERRORS as error:
        catalog = None
        render_action_error("Profile 配置加载", error)

    st.subheader("全量 init")
    full_summary = cached_dataset_summary(
        str(context.init_root.resolve()), ui_cache_version("dataset")
    )
    full_report = state.dataset_builds.get("full_init")
    full_init_ready = (
        bool(full_summary["exists"])
        and isinstance(full_report, dict)
        and full_report.get("status") == "ready"
        and not _build_report_stale(context, full_report)
    )
    test_summary = cached_dataset_summary(
        str(context.init_test_root.resolve()), ui_cache_version("dataset")
    )
    class_count = (
        len([path for path in context.init_root.iterdir() if path.is_dir()])
        if context.init_root.is_dir() else 0
    )
    metrics = st.columns(5)
    metrics[0].metric(
        "状态",
        "待重建" if full_summary["exists"] and not full_init_ready else (
            "已生成" if full_summary["exists"] else "未生成"
        ),
    )
    metrics[1].metric("类别数", f"{class_count} 个")
    metrics[2].metric("来源组数", f"{full_summary['folder_count']} 个")
    metrics[3].metric("init 光谱", f"{full_summary['file_count']} 条")
    metrics[4].metric("init_test 光谱", f"{test_summary['file_count']} 条")
    if context.hierarchy.is_flat:
        _render_flat_label_counts(context)
    else:
        _render_taxon_counts(context.init_root)
    eligible_dates = sorted(state.affine_completed_dates, key=build_natural_key)
    supplement_plan = (
        scan_supplement_plan(context.supplement_root, context.init_root, catalog)
        if not context.hierarchy.is_flat and catalog is not None
        else None
    )
    supplement_ready = context.hierarchy.is_flat or (
        supplement_plan is not None and not supplement_plan.errors
    )
    test_candidates = _list_test_sources(context, eligible_dates)
    test_selection_key = "full_init_test_selection"
    initialized_key = "full_init_test_selection_initialized"
    returned_source = st.session_state.pop("test_return_removed_source", None)
    if returned_source is not None:
        st.session_state[test_selection_key] = [value for value in st.session_state.get(test_selection_key, []) if value != returned_source]
    exported_source = st.session_state.pop("test_export_added_source", None)
    if exported_source is not None:
        st.session_state[test_selection_key] = sorted(set(st.session_state.get(test_selection_key, [])) | {exported_source}, key=build_natural_key)
    _sync_test_selection(
        context,
        test_candidates,
        test_selection_key,
        initialized_key,
    )
    _render_default_test_selection_button(
        context,
        test_candidates,
        test_selection_key,
    )
    selected_test = st.multiselect(
        "选择进入 init_test 的来源文件夹",
        test_candidates,
        format_func=lambda value: value.replace("/", " / "),
        key=test_selection_key,
    )
    confirm_supplement = True
    if not context.hierarchy.is_flat:
        _render_supplement_plan(supplement_plan, context.supplement_root)
        confirm_supplement = st.checkbox(
            "我确认按上述补充数据重建全量 init",
            key="full_init_supplement_confirm",
            disabled=not supplement_ready,
        )
    if st.button(
        "从 shift_data 重建全量 init",
        disabled=not supplement_ready or not confirm_supplement,
        width="stretch",
    ):
        if not context.hierarchy.is_flat:
            validation_errors = validate_supplement_plan(supplement_plan)
            if validation_errors:
                st.error("补充数据校验失败：" + "；".join(validation_errors))
                return
        try:
            with st.status("正在重建全量 init…", expanded=True) as status:
                progress = st.progress(0.0, text="正在校验日期门禁并收集来源文件夹…")

                def report_copy_progress(date_name: str, processed: int, total: int) -> None:
                    """按已复制光谱数刷新全量 init 构建进度。"""
                    progress.progress(
                        progress_ratio(processed, total),
                        text=f"[{processed}/{total} 条光谱] 正在处理 {date_name}…",
                    )

                if context.hierarchy.is_flat:
                    report = build_full_init(
                        context.shift_root,
                        context.init_root,
                        catalog,
                        eligible_dates,
                        test_root=context.init_test_root,
                        selected_test_folders=selected_test,
                        hierarchy=context.hierarchy,
                        progress_callback=report_copy_progress,
                    )
                else:
                    report = build_full_init_with_supplements(
                        context.shift_root,
                        context.init_root,
                        catalog,
                        eligible_dates,
                        supplement_plan,
                        test_root=context.init_test_root,
                        selected_test_folders=selected_test,
                        hierarchy=context.hierarchy,
                        progress_callback=report_copy_progress,
                    )
                progress.progress(1.0, text="全量 init 写入完成")
                status.update(label="全量 init 重建完成", state="complete", expanded=False)
            state.dataset_builds["full_init"] = annotate_dataset_build(report, context.dataset_spec)
            invalidate_ui_cache(*INVALIDATE_DATASET)
            write_page_state(context, state)
            st.session_state.pop("full_init_supplement_confirm", None)
            if context.hierarchy.is_flat:
                set_flash(
                    "dataset",
                    f"全量 init 生成完成：{report['count']['file_count']} 条光谱，"
                    f"{report['test_folder_count']} 个测试组。",
                )
            else:
                supplement = report["supplement"]
                supplement_note = (
                    f"已写入 {supplement['folder_count']} 个补充文件夹（{supplement['file_count']} 条光谱）"
                    if supplement["folder_count"]
                    else "未写入补充数据"
                )
                set_flash(
                    "dataset",
                    f"全量 init 生成完成，{supplement_note}；已有 Profile 保留，可按需重新生成。",
                )
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error("全量 init 生成", error)

    _render_date_dataset_update(context, state, catalog)
    _render_dataset_transfer(context, state, catalog, False)
    _render_dataset_transfer(context, state, catalog, True)

    if catalog is None:
        st.info("请先修复本地分类 Excel 文件。")
        return

    _render_prefix_plot_section(
        context.init_root,
        context.fig_root,
        "full_init",
        InputConfig(bad_bands=context.bad_bands),
        flat=context.hierarchy.is_flat,
    )
    _render_affine_compensation(context, state, catalog)

    st.subheader("Profile 数据集")
    profile = render_profile_picker(
        context,
        label="选择 Profile",
        key="dataset_profile_name",
        profile_ids=list(context.dataset_spec.profiles),
        empty_message="当前数据集没有 Profile 子目录。",
        empty_kind="info",
    )
    if profile is None:
        return
    profile_name = profile.profile_id
    profile_dir = context.profile_dir(profile)
    profile_root = profile_dir / profile.root_init
    profile_summary = cached_dataset_summary(
        str(profile_root.resolve()), ui_cache_version("dataset")
    )
    profile_test_summary = cached_dataset_summary(
        str((profile_dir / profile.root_test).resolve()), ui_cache_version("dataset")
    )
    _render_count_metrics(profile_summary, profile_test_summary)
    profile_reports = state.dataset_builds.get("profiles", {})
    if not isinstance(profile_reports, dict):
        profile_reports = {}

    full_init_report = state.dataset_builds.get("full_init")
    full_init_report = full_init_report if isinstance(full_init_report, dict) else {}
    test_exports = {
        str(entry["target"]): str(entry["source"])
        for entry in full_init_report.get("test_exports", [])
        if isinstance(entry, dict) and entry.get("target") and entry.get("source")
    }
    configured_defaults = set(_load_configured_default_sources() or [])
    prefix_to_genus = {
        prefix: genus
        for genus, prefixes in catalog.genera.items()
        for prefix in prefixes
    }
    profile_genera = set(catalog.datasets.get(profile_name, ()))
    profile_default_sources = {
        source
        for source in configured_defaults
        if prefix_to_genus.get(
            parse_folder_prefix(source.split("/", 1)[1], uppercase_enable=True)
        ) in profile_genera
    }
    allowed_test_targets = {
        target
        for target, source in test_exports.items()
        if source in profile_default_sources
    }
    strict_default_enable = st.checkbox(
        "测试集严格按默认名单",
        value=bool(profile_default_sources) and bool(test_exports),
        disabled=not profile_default_sources or not test_exports,
        key=f"profile_strict_test_defaults_{profile_name}",
        help="只复制默认名单中属于当前 Profile 的测试夹，忽略全局 init_test 的历史遗留夹。",
    )
    if profile_default_sources and test_exports:
        st.caption(
            f"默认名单中属于 {profile_name} 的测试源 {len(profile_default_sources)} 个；"
            f"严格模式候选 {len(allowed_test_targets)} 个测试夹，忽略历史遗留测试夹。"
        )
    elif not profile_default_sources:
        st.warning(f"默认名单中没有 {profile_name} 的测试源；将保留现有 init_test。")

    if st.button(
        "从 init 生成数据集",
        type="primary",
        disabled=not full_init_ready,
        width="stretch",
    ):
        try:
            with st.status(f"正在生成 {profile_name} 数据集…", expanded=True) as status:
                status.write("正在读取全量 init 并生成 Profile 子集，同时从全局 init_test 派生 Profile 测试集。")
                report = build_profile_dataset(
                    context.init_root,
                    profile_root,
                    profile_name,
                    catalog,
                    allowed_test_targets=(
                        allowed_test_targets if strict_default_enable else None
                    ),
                )
                status.update(label=f"{profile_name} 数据集生成完成", state="complete", expanded=False)
            profile_reports[profile_name] = report
            state.dataset_builds["profiles"] = profile_reports
            invalidate_ui_cache(*INVALIDATE_DATASET)
            write_page_state(context, state)
            set_flash("dataset", f"{profile_name} 数据集生成完成。")
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error(f"{profile_name} 数据集生成", error)
