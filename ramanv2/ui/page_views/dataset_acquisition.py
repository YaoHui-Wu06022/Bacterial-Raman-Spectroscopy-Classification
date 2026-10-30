"""全量 init 和固定 Profile 数据集获取页面。"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from ramanv2.data.count import summarize_dataset
from ramanv2.data.builders.init import (
    build_full_init,
    build_init_datasets,
    build_profile_dataset,
)
from ramanv2.data.plots.prefix import build_prefix_plots
from ramanv2.pipeline.calibration import run_folder_offset
from ramanv2.data.catalog import DatasetMapping, ensure_dataset_mapping
from ramanv2.common.naming import (
    NON_TRAINING_FOLDER_NAMES,
    PIPELINE_RESERVED_FOLDER_NAMES,
    build_natural_key,
    parse_folder_prefix,
)
from ramanv2.data.profiles import PIPELINE_PROFILE_IDS, TEST_ROOT_NAME, get_dataset_dir, get_profile
from ramanv2.pipeline.dataset import collect_test_return_sources, collect_dataset_source_folders, collect_init_export_sources, run_test_to_init, run_folder_dataset_update, run_init_to_test
from ramanv2.pipeline.supplement import (
    SupplementPlan,
    apply_supplement_plan,
    scan_supplement_plan,
    validate_supplement_plan,
)
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.cache import cached_dataset_summary, cached_dataset_taxon_counts
from ramanv2.ui.state import (
    invalidate_ui_cache,
    load_page_state,
    save_page_state,
    ui_cache_version,
)


# 候选源文件夹：流水线保留目录与校准球都不作为普通来源。
SOURCE_RESERVED_NAMES = PIPELINE_RESERVED_FOLDER_NAMES | NON_TRAINING_FOLDER_NAMES


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
    metrics[1].metric("种数量", f"{summary['species_count']} 个")
    metrics[2].metric("光谱数量", f"{summary['file_count']} 条")


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
        view_clicked = controls[0].button("查看已有图信息", key=f"{key}_view", width="stretch")
        update_clicked = controls[1].button("更新图信息", key=f"{key}_update", width="stretch")
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
                save_page_state(context, state)
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
            st.image(str(selected), caption=f"{genus_name} / {prefix_name}", width="stretch")


def _render_supplement_plan(
    context: PageContext, catalog: DatasetMapping | None
) -> SupplementPlan | None:
    """展示补充数据预检表；有错误时返回 None，阻止全量重建。"""
    if catalog is None:
        return None
    plan = scan_supplement_plan(context.supplement_root, context.init_root, catalog)
    st.markdown("**补充数据**")
    st.caption(f"来源目录：{context.supplement_root}；重建后按文件夹写入 dataset/init/<属>/<文件夹>。")
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
    return plan if not plan.errors else None


def _render_folder_dataset_update(context: PageContext, state) -> None:
    """已有日期恢复光谱后，按原走向从 shift 更新一个目标组。"""
    with st.expander("按文件夹更新 init/init_test", expanded=False):
        message = st.session_state.pop("dataset_update_message", None)
        if message:
            st.success(message)
        folders = collect_dataset_source_folders(context.data_root, context.shift_root, state)
        if not folders:
            st.info("当前没有已有输出走向的 shift 文件夹，请先生成全量 init。")
            return
        st.caption("活动来源从 shift_data 更新；已整夹删除的来源同步移除旧产物。沿用原输出目录和测试分流，其他组编号不变。")
        selected = st.selectbox(
            "选择更新文件夹", folders, key="dataset_update_source",
            format_func=lambda value: value if (context.data_root / value).is_dir() else f"{value}（已整夹删除）",
        )
        is_deleted = not (context.data_root / selected).exists()
        if is_deleted:
            st.info("此来源已整夹删除；同步将移除它对应的旧 init/init_test 文件夹，cosmic 删除归档保留。")
        if not st.button("同步所选文件夹删除" if is_deleted else "从 shift_data 更新所选文件夹", key="dataset_update_confirm", width="stretch"):
            return
        try:
            with st.spinner(f"正在同步 {selected}……"):
                result = run_folder_dataset_update(context.data_root, context.shift_root, context.init_root, state, context.state_file, selected)
            for name in ("dataset", "training", "inference", "analysis"):
                invalidate_ui_cache(name)
            destination = "init_test" if result["test_updated"] else "init 及对应前缀图"
            if result.get("deleted"):
                target_name = "init_test" if result["test_updated"] else "init"
                st.session_state["dataset_update_message"] = f"{selected} 已同步整夹删除，移除对应旧 {target_name} 文件夹（{result['removed_file_count']} 条光谱），其他组编号不变。"
            else:
                st.session_state["dataset_update_message"] = f"{selected} 已更新 {destination}（{result['spectrum_count']} 条光谱）。"
            st.rerun()
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            st.error(f"文件夹更新失败：{error}")


def _render_test_return(context: PageContext, state, catalog: DatasetMapping | None) -> None:
    """渲染 init_test 到 init 的快速迁移。"""
    _render_dataset_transfer(context, state, catalog, False)


def _render_init_export(context: PageContext, state, catalog: DatasetMapping | None) -> None:
    """渲染 init 到 init_test 的快速迁移。"""
    _render_dataset_transfer(context, state, catalog, True)


def _render_dataset_transfer(context: PageContext, state, catalog: DatasetMapping | None, to_test_enable: bool) -> None:
    """双向迁移只提交所选目录，普通控件交互不写入数据。"""
    key = "init_export" if to_test_enable else "test_return"
    title = "init 快速移入 init_test" if to_test_enable else "init_test 快速移回 init"
    destination = "init_test" if to_test_enable else "init"
    with st.expander(title, expanded=False):
        message = st.session_state.pop(f"{key}_message", None)
        if message:
            st.success(message)
        sources = collect_init_export_sources(context.data_root, context.init_root, state) if to_test_enable else {
            source: source for source in collect_test_return_sources(context.init_root, state)
        }
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
            for name in ("dataset", "training", "inference", "analysis"):
                invalidate_ui_cache(name)
            st.session_state["test_export_added_source" if to_test_enable else "test_return_removed_source"] = selected
            st.session_state[f"{key}_message"] = f"{selected} 已直接{'移入' if to_test_enable else '移回'} {destination}（{result['spectrum_count']} 条）：{Path(result['target_dir']).name}。"
            st.rerun()
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            st.error(f"迁移到 {destination} 失败：{error}")


def _render_affine_compensation(context: PageContext, state, catalog: DatasetMapping) -> None:
    """输入只表示本次 cosmic 原轴增量，提交成功或切换文件夹后归零。"""
    st.subheader("补充频移")
    if st.session_state.pop("folder_offset_reset", False):
        st.session_state["folder_offset_delta"] = 0.0
    message = st.session_state.pop("folder_offset_message", None)
    if message:
        st.success(message)
    dates = state.affine_completed_dates
    folders = [
        f"{date}/{folder.name}"
        for date in sorted(dates)
        for folder in sorted((context.data_root / date).glob("*"), key=lambda item: build_natural_key(item.name))
        if folder.is_dir() and folder.name not in SOURCE_RESERVED_NAMES and any(folder.rglob("*.arc_data"))
    ]
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
            result = run_folder_offset(context.data_root, context.shift_root, context.init_root, catalog, state, context.state_file, selected, float(delta), context.init_root.parent / "fig_init")
            if not result["changed_folders"]:
                st.info("本次频移量为 0，未改动光谱。")
                return
            for name in ("cleaning", "affine", "dataset"):
                invalidate_ui_cache(name)
            st.session_state["folder_offset_reset"] = True
            destination = "对应 init_test 文件夹" if result.get("test_updated") else "对应 init 文件夹及审核图"
            st.session_state["folder_offset_message"] = f"{selected} 已同步平移活动谱及归档谱，并更新{destination}。"
            st.rerun()
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            st.error(f"频移补偿失败：{error}")


@st.cache_data(show_spinner=False)
def _cached_test_sources(shift_root: str, date_names: tuple[str, ...], version: int) -> tuple[str, ...]:
    """缓存日期下的候选目录扫描。"""
    del version
    root = Path(shift_root)
    candidates: list[str] = []
    for date_name in date_names:
        date_dir = root / date_name
        if not date_dir.is_dir():
            continue
        for folder in sorted(date_dir.iterdir(), key=lambda item: build_natural_key(item.name)):
            if folder.is_dir() and folder.name not in SOURCE_RESERVED_NAMES and any(folder.glob("*.arc_data")):
                candidates.append(f"{date_name}/{folder.name}")
    return tuple(candidates)


def _list_test_sources(context: PageContext, date_names: list[str]) -> list[str]:
    """列出指定日期中可放入 init_test 的测试文件夹。"""
    return list(
        _cached_test_sources(
            str(context.shift_root.resolve()),
            tuple(sorted(date_names, key=build_natural_key)),
            ui_cache_version("dataset"),
        )
    )


# 默认选择逻辑版本号，修改逻辑时递增以强制重新初始化
_TEST_SELECTION_LOGIC_VERSION = 5


def _default_test_sources(candidates: list[str]) -> list[str]:
    """默认选择每个种的倒数第二个文件夹（恰好 4 个）或倒数第三个（>= 5 个）；少于 4 个则不选择。"""
    from collections import defaultdict
    species_folders: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        folder_name = candidate.rsplit("/", 1)[-1]
        prefix = parse_folder_prefix(folder_name, uppercase_enable=True)
        species_folders[prefix].append(candidate)
    selected: list[str] = []
    for prefix, folders in species_folders.items():
        if len(folders) < 4:
            continue
        sorted_folders = sorted(folders, key=build_natural_key)
        selected.append(sorted_folders[-3] if len(folders) >= 5 else sorted_folders[-2])
    return sorted(selected, key=build_natural_key)


def _legacy_default_test_sources(candidates: list[str]) -> list[str]:
    """返回旧版“第二个文件夹”规则，用于迁移已有会话选择。"""
    from collections import defaultdict

    species_folders: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        folder_name = candidate.rsplit("/", 1)[-1]
        prefix = parse_folder_prefix(folder_name, uppercase_enable=True)
        species_folders[prefix].append(candidate)
    selected = []
    for folders in species_folders.values():
        if len(folders) >= 4:
            selected.append(sorted(folders, key=build_natural_key)[1])
    return sorted(selected, key=build_natural_key)


def _sync_test_selection(
    candidates: list[str],
    selection_key: str,
    initialized_key: str,
) -> None:
    """按当前候选集补齐默认选择，同时保留用户显式移除的默认项。"""
    version_key = f"{selection_key}_version"
    previous_candidates_key = f"{selection_key}_candidates"
    defaults_key = f"{selection_key}_defaults"
    suppressed_key = f"{selection_key}_suppressed_defaults"
    version_changed = st.session_state.get(version_key) != _TEST_SELECTION_LOGIC_VERSION
    initialized = st.session_state.get(initialized_key, False)
    previous_defaults = set(st.session_state.get(defaults_key, []))
    current_candidates = set(candidates)

    # 首次初始化时直接采用当前候选集的默认值；逻辑版本升级时保留已有选择，
    # 再按当前候选集补齐默认值，避免旧版本状态阻断新规则。
    if not initialized:
        selected = _default_test_sources(candidates)
        suppressed_defaults: set[str] = set()
    else:
        selected = [
            value
            for value in st.session_state.get(selection_key, [])
            if value in candidates
        ]
        suppressed_defaults = set(st.session_state.get(suppressed_key, []))
        if version_changed:
            selected = [value for value in selected if value not in previous_defaults]
        if not version_changed:
            # 只记录仍然存在但被用户从选择中移除的旧默认项；候选已消失时，
            # 允许排序变化后由新的第二个文件夹接替它。
            suppressed_defaults.update(
                value
                for value in previous_defaults & current_candidates
                if value not in selected
            )
        suppressed_defaults.difference_update(selected)

    defaults = set(_default_test_sources(candidates))
    selected_set = set(selected)
    selected_set.update(defaults - suppressed_defaults)
    st.session_state[selection_key] = sorted(selected_set, key=build_natural_key)
    st.session_state[defaults_key] = sorted(defaults, key=build_natural_key)
    st.session_state[suppressed_key] = sorted(suppressed_defaults, key=build_natural_key)
    st.session_state[version_key] = _TEST_SELECTION_LOGIC_VERSION
    st.session_state[previous_candidates_key] = list(candidates)
    st.session_state[initialized_key] = True


def _render_default_test_selection_button(candidates: list[str], selection_key: str) -> None:
    """提供显式的默认选择恢复入口，避免误用多选框的全选操作。"""
    if not st.button(
        "恢复默认选择",
        key=f"{selection_key}_reset_defaults",
        help="按当前候选文件夹重新应用默认规则：每个前缀 4 个选倒数第二个，>= 5 个选倒数第三个。",
    ):
        return
    defaults = _default_test_sources(candidates)
    st.session_state[selection_key] = defaults
    st.session_state[f"{selection_key}_defaults"] = defaults
    st.session_state[f"{selection_key}_suppressed_defaults"] = []
    st.rerun()


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
    for key in ("source_files", "output_files", "init_source_files", "init_output_files", "test_files", "test_output_files", "test_source_files", "test_output_files"):
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
    merged["test_folder_count"] = len(merged["test_folders"])
    merged["selected_test_folders"] = sorted(
        set(previous.get("selected_test_folders", [])) | set(update.get("selected_test_folders", [])),
        key=build_natural_key,
    )
    merged["test_exports"] = [*previous.get("test_exports", []), *update.get("test_exports", [])]
    merged["count"] = summarize_dataset(init_root)
    merged["mode"] = "full"
    merged["status"] = "ready"
    return merged


def render_dataset_acquisition_page(context: PageContext) -> None:
    state = load_page_state(context)

    st.title("数据集获取")
    st.caption("先生成全量 dataset/init，再选择固定 Profile 生成对应数据集。")
    flash = st.session_state.pop("dataset_flash", None)
    if flash:
        st.success(flash)

    try:
        catalog = _load_catalog(context)
    except (OSError, ValueError, RuntimeError) as error:
        catalog = None
        st.error(f"Profile 配置加载失败：{error}")

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
    st.metric("状态", "待重建" if full_summary["exists"] and not full_init_ready else ("已生成" if full_summary["exists"] else "未生成"))
    _render_count_metrics(full_summary)
    _render_taxon_counts(context.init_root)
    eligible_dates = sorted(state.affine_completed_dates, key=build_natural_key)
    test_candidates = _list_test_sources(context, eligible_dates)
    test_selection_key = "full_init_test_selection"
    initialized_key = "full_init_test_selection_initialized"
    if not st.session_state.get(initialized_key, False) and isinstance(full_report, dict):
        st.session_state[test_selection_key] = list(full_report.get("selected_test_folders", []))
        st.session_state[initialized_key] = True
        st.session_state[f"{test_selection_key}_version"] = _TEST_SELECTION_LOGIC_VERSION - 1
        st.session_state[f"{test_selection_key}_defaults"] = _legacy_default_test_sources(test_candidates)
        st.session_state[f"{test_selection_key}_candidates"] = list(test_candidates)
    returned_source = st.session_state.pop("test_return_removed_source", None)
    if returned_source is not None:
        st.session_state[test_selection_key] = [value for value in st.session_state.get(test_selection_key, []) if value != returned_source]
    exported_source = st.session_state.pop("test_export_added_source", None)
    if exported_source is not None:
        st.session_state[test_selection_key] = sorted(set(st.session_state.get(test_selection_key, [])) | {exported_source}, key=build_natural_key)
    _sync_test_selection(test_candidates, test_selection_key, initialized_key)
    _render_default_test_selection_button(test_candidates, test_selection_key)
    selected_test = st.multiselect(
        "选择进入 init_test 的来源文件夹",
        test_candidates,
        format_func=lambda value: value.replace("/", " / "),
        key=test_selection_key,
    )
    supplement_plan = _render_supplement_plan(context, catalog)
    confirm_supplement = st.checkbox(
        "我确认按上述补充数据重建全量 init",
        key="full_init_supplement_confirm",
        disabled=supplement_plan is None,
    )
    if st.button(
        "从 shift_data 重建全量 init",
        disabled=supplement_plan is None or not confirm_supplement,
        width="stretch",
    ):
        validation_errors = validate_supplement_plan(supplement_plan)
        if validation_errors:
            st.error("补充数据校验失败：" + "；".join(validation_errors))
        else:
            try:
                with st.status("正在重建全量 init…", expanded=True) as status:
                    status.write("正在校验日期门禁并复制普通光谱。")
                    status.write("正在按选择更新 init_test。")
                    report = build_full_init(
                        context.shift_root,
                        context.init_root,
                        catalog,
                        eligible_dates,
                        test_root=context.init_root.parent / TEST_ROOT_NAME,
                        selected_test_folders=selected_test,
                    )
                    status.write("正在写入补充数据文件夹。")
                    try:
                        supplement = apply_supplement_plan(context.init_root, supplement_plan)
                    except (OSError, ValueError, RuntimeError) as error:
                        raise RuntimeError(f"全量 init 已重建，但补充数据写入失败：{error}") from error
                    report["supplement"] = supplement
                    report["count"] = summarize_dataset(context.init_root.resolve())
                    status.update(label="全量 init 重建完成", state="complete", expanded=False)
                state.dataset_builds["full_init"] = report
                invalidate_ui_cache("dataset")
                invalidate_ui_cache("training")
                invalidate_ui_cache("inference")
                invalidate_ui_cache("analysis")
                save_page_state(context, state)
                st.session_state.pop("full_init_supplement_confirm", None)
                supplement_note = (
                    f"已写入 {supplement['folder_count']} 个补充文件夹（{supplement['file_count']} 条光谱）"
                    if supplement["folder_count"]
                    else "未写入补充数据"
                )
                st.session_state["dataset_flash"] = (
                    f"全量 init 生成完成，{supplement_note}；已有 Profile 保留，可按需重新生成。"
                )
                st.rerun()
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"全量 init 生成失败：{error}")

    _render_folder_dataset_update(context, state)
    _render_test_return(context, state, catalog)
    _render_init_export(context, state, catalog)

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
            test_candidates = _list_test_sources(context, incremental_dates)
            test_selection_key = "incremental_init_test_selection"
            _sync_test_selection(
                test_candidates,
                test_selection_key,
                "incremental_init_test_selection_initialized",
            )
            _render_default_test_selection_button(test_candidates, test_selection_key)
            selected_test = st.multiselect(
                "新增来源中选择进入 init_test 的文件夹",
                test_candidates,
                format_func=lambda value: value.replace("/", " / "),
                key=test_selection_key,
            )
            if st.button("增量更新 init", disabled=catalog is None, width="stretch"):
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
                            test_root=context.init_root.parent / TEST_ROOT_NAME,
                            selected_test_folders=selected_test,
                        )
                        status.update(label="增量更新完成", state="complete", expanded=False)
                    state.dataset_builds["full_init"] = _merge_incremental_init_report(
                        full_report,
                        update,
                        context.init_root,
                    )
                    invalidate_ui_cache("dataset")
                    invalidate_ui_cache("training")
                    invalidate_ui_cache("inference")
                    invalidate_ui_cache("analysis")
                    save_page_state(context, state)
                    st.success("init 已完成增量更新。")
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as error:
                    st.error(f"init 增量更新失败：{error}")
        else:
            st.info("当前没有尚未进入 init 的新日期。")

    if catalog is None:
        st.info("请先修复本地病原菌分类 Excel 文件。")
        return

    _render_prefix_plot_section(context, context.init_root, context.init_root.parent / "fig_init", "full_init", state)
    _render_affine_compensation(context, state, catalog)

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
        width="stretch",
    ):
        try:
            with st.status(f"正在生成 {profile_name} 数据集…", expanded=True) as status:
                status.write("正在读取全量 init 并生成 Profile 子集，同时从全局 init_test 派生 Profile 测试集。")
                report = build_profile_dataset(context.init_root, profile_root, profile_name, catalog)
                status.update(label=f"{profile_name} 数据集生成完成", state="complete", expanded=False)
            profile_reports[profile_name] = report
            state.dataset_builds["profiles"] = profile_reports
            invalidate_ui_cache("dataset")
            invalidate_ui_cache("training")
            invalidate_ui_cache("inference")
            invalidate_ui_cache("analysis")
            save_page_state(context, state)
            st.success(f"{profile_name} 数据集生成完成。")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"{profile_name} 数据集生成失败：{error}")



__all__ = ["render_dataset_acquisition_page"]
