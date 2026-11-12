"""数据清洗页面：当前日期、当前文件夹、当前页的增量审核。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st

from ramanv2.audit.heterogeneity import (
    AUTO_MIN_SEPARATION,
    HeterogeneityReport,
    analyze_library,
)
from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.naming import BEAD_FOLDER_NAME, NON_TRAINING_FOLDER_NAMES, build_natural_key
from ramanv2.common.plotting import configure_matplotlib_fonts
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.pipeline.anomaly_removal import (
    check_analysis_gate,
    execute_analysis_removal,
)
from ramanv2.pipeline.cleaning import (
    SPECIAL_FOLDER_NAMES,
    ReviewFolder,
    SpectrumItem,
    commit_date_items,
    commit_folder_items,
    commit_paths,
    complete_folder,
    discard_removed_date,
    discard_removed_folder,
    refresh_completed_dates,
    reconcile_cleaning_records,
    reopen_date_review,
    scan_review_folders,
    sync_folder_catalog,
    toggle_mark_path,
    unmark_paths,
)
from ramanv2.pipeline.deleted_data import (
    DeletedFolder,
    DeletedSpectrum,
    purge_deleted_spectra,
    restore_deleted_folder,
    restore_deleted_spectra,
    scan_deleted_folders,
    scan_deleted_spectra,
)
from ramanv2.pipeline.state import PipelineState
from ramanv2.records.lineage import mark_folders_reviewed
from ramanv2.ui.components.spectrum_grid import render_spectrum_grid
from ramanv2.ui.formatting import progress_ratio
from ramanv2.ui.read_cache import cached_directory_names
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.page_state import (
    INVALIDATE_ALL,
    INVALIDATE_CLEANING,
    INVALIDATE_CLEANING_AFFINE_DATASET,
    INVALIDATE_CLEANING_DATASET,
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


configure_matplotlib_fonts()


PAGE_SIZE = 9
DATE_CACHE_KEY = "cleaning_current_date_cache"
DATE_CACHE_NAME_KEY = "cleaning_current_date_name"


def _page_bounds(total: int, index: int) -> tuple[int, int]:
    """返回总页数与钳制后的页码。"""
    page_count = max((total + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    return page_count, min(max(index, 0), page_count - 1)


def _render_page_controls(page_index: int, page_count: int, *, key_prefix: str, on_change) -> None:
    """渲染统一的上一页/下一页/页码控件。"""
    previous, following, page = st.columns(3)
    if previous.button("上一页", disabled=page_index == 0, key=f"{key_prefix}_previous", width="stretch"):
        on_change(page_index - 1)
    if following.button("下一页", disabled=page_index >= page_count - 1, key=f"{key_prefix}_next", width="stretch"):
        on_change(page_index + 1)
    page.metric("当前页", f"{page_index + 1} / {page_count}")


def _store_session_page(key: str, index: int) -> None:
    """保存会话级页码并刷新页面。"""
    st.session_state[key] = index
    st.rerun()


def _normalize_relative_path(path: object) -> str:
    """把组件回传的路径统一为正斜杠形式。"""
    return str(path).replace("\\", "/")


def _apply_page_selection(page_paths: set[str], marked: set[str], chosen: set[str], *, remove, add) -> bool:
    """把页内选中差集应用到当前选择；无变化时返回 False。"""
    current = marked.intersection(page_paths)
    target = chosen.intersection(page_paths)
    if target == current:
        return False
    remove(current - target)
    add(target - current)
    return True


def _deleted_records(data_root: Path | str, mapping: DatasetMapping) -> list[DeletedSpectrum]:
    """扫描人工删除区，**不缓存**。

    清单曾用 ``@st.cache_data`` 缓存、键里放会话内版本号，但 ``st.cache_data`` 是跨会话
    共享的：新会话（刷新页面、新标签页）版本号重置为 0，会复用旧会话留下的清单，对已被
    移走的文件调用绘图从而报 ``WinError 2``。改用目录指纹做键同样不可靠——Windows 上目录
    mtime 的更新存在非确定性延迟（同一条测试重复运行会时对时错）。``os.scandir`` 重写后
    整树扫描约 65ms，直接每次扫描，换取「列表永远与磁盘一致」。
    """
    return scan_deleted_spectra(Path(data_root), mapping, include_bead=True)


def _deleted_folders(data_root: Path | str) -> list[DeletedFolder]:
    """扫描整夹删除区，同样不缓存（理由见 ``_deleted_records``）。"""
    return scan_deleted_folders(Path(data_root))


@st.cache_data(show_spinner=False)
def build_spectrum_image(path_string: str, modified_ns: int) -> str:
    del modified_ns
    wavenumbers, intensities = read_arc_data(Path(path_string))
    if not wavenumbers.size or not intensities.size:
        raise ValueError("文件没有可读取的两列光谱数据")
    figure, axis = plt.subplots(figsize=(5.1, 1.8), dpi=120)
    try:
        axis.plot(wavenumbers, intensities, color="#2563eb", linewidth=0.9)
        axis.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=7)
        axis.set_ylabel("Intensity", fontsize=7)
        axis.tick_params(axis="both", labelsize=6)
        axis.grid(alpha=0.18, linewidth=0.4)
        figure.tight_layout(pad=0.5)
        output = BytesIO()
        figure.savefig(output, format="png")
    finally:
        plt.close(figure)
    return f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}"


def _clear_date_cache() -> None:
    st.session_state.pop(DATE_CACHE_KEY, None)
    st.session_state.pop(DATE_CACHE_NAME_KEY, None)


def _load_current_date(
    context: PageContext,
    state: PipelineState,
    date_name: str,
    force_enable: bool = False,
) -> dict[str, ReviewFolder]:
    cached = st.session_state.get(DATE_CACHE_KEY)
    cached_date = st.session_state.get(DATE_CACHE_NAME_KEY)
    if not force_enable and cached_date == date_name and isinstance(cached, dict):
        return cached
    folders = scan_review_folders(
        context.data_root,
        [date_name],
        state.cleaning.folder_catalog,
        set(),
    )
    before = {key: set(value) for key, value in state.cleaning.folder_catalog.items()}
    sync_folder_catalog(state.cleaning, folders)
    refresh_completed_dates(state.cleaning)
    if state.cleaning.folder_catalog != before:
        write_page_state(context, state)
    catalog = {folder.key: folder for folder in folders}
    st.session_state[DATE_CACHE_KEY] = catalog
    st.session_state[DATE_CACHE_NAME_KEY] = date_name
    return catalog


def _current_folders(catalog: dict[str, ReviewFolder], state: PipelineState) -> list[ReviewFolder]:
    return [
        folder
        for folder in catalog.values()
        if folder.key not in state.cleaning.completed_folder_keys
    ]


def _folder_items(folder: ReviewFolder) -> list[SpectrumItem]:
    return [
        SpectrumItem(path, folder.date_name, folder.folder_name, path.rsplit("/", 1)[-1])
        for path in folder.relative_paths
    ]


def _spectrum_card(
    read_path: Path,
    relative_path: str,
    label: str,
    status_text: str | None = None,
) -> dict[str, object]:
    """生成选择网格卡片；读取失败时保留错误占位，不中断整页渲染。"""
    card: dict[str, object] = {"relative_path": relative_path, "label": label}
    if status_text is not None:
        card["status_text"] = status_text
    try:
        card["image_data"] = build_spectrum_image(str(read_path), read_path.stat().st_mtime_ns)
    except (OSError, ValueError) as error:
        card["error"] = f"读取失败：{error}"
    return card


def _build_cards(root: Path, items: list[SpectrumItem]) -> list[dict[str, object]]:
    return [
        _spectrum_card(
            root / item.relative_path,
            item.relative_path,
            f"{item.date_name} / {item.folder_name} / {item.file_name}",
        )
        for item in items
    ]


def _is_review_date_available(root: Path, date_name: str) -> bool:
    """判断日期目录是否含至少一个可审核文件夹；空目录不进入审核队列。"""
    date_dir = root / date_name
    if not date_dir.is_dir():
        return False
    return any(
        path.is_dir() and path.name not in SPECIAL_FOLDER_NAMES
        for path in date_dir.iterdir()
    )


def _next_pending_date(dates: list[str], state: PipelineState) -> str | None:
    return next((date_name for date_name in dates if date_name not in state.cleaning.completed_date_names), None)


def _set_next_folder_or_date(
    state: PipelineState,
    dates: list[str],
    catalog: dict[str, ReviewFolder],
) -> None:
    remaining = _current_folders(catalog, state)
    if remaining:
        state.cleaning.current_folder_key = remaining[0].key
        return
    refresh_completed_dates(state.cleaning)
    next_date = _next_pending_date(dates, state)
    state.cleaning.current_date_name = next_date
    state.cleaning.current_folder_key = None
    _clear_date_cache()


def _commit_current_folder(
    context: PageContext,
    state: PipelineState,
    dates: list[str],
    catalog: dict[str, ReviewFolder],
    folder: ReviewFolder,
) -> bool:
    paths = set(folder.relative_paths)
    marked = state.cleaning.marked_paths.intersection(paths)
    result = commit_paths(context.data_root, state, context.state_file, marked)
    if result.missing_paths:
        st.warning(f"有 {len(result.missing_paths)} 条文件不存在，当前夹暂未完成。")
        return False
    complete_folder(state.cleaning, folder)
    mark_folders_reviewed(context.data_root, [folder.key], reviewed=True)
    if not (context.data_root / folder.date_name / folder.folder_name).exists():
        catalog.pop(folder.key, None)
        discard_removed_folder(state.cleaning, folder)
    else:
        catalog[folder.key] = ReviewFolder(folder.key, folder.date_name, folder.folder_name, ())
    _set_next_folder_or_date(state, dates, catalog)
    write_page_state(context, state)
    invalidate_ui_cache(*INVALIDATE_CLEANING)
    return True


def _restore_card_data(root: Path, record: DeletedSpectrum) -> dict[str, object]:
    """将删除目录中的光谱转换为选择网格卡片。"""
    return _spectrum_card(
        root / record.source_relative_path,
        record.relative_path,
        f"{record.date_name} / {record.folder_name} / {record.file_name}",
        "待恢复",
    )


def _render_reopen_section(
    context: PageContext,
    state: PipelineState,
    dates: list[str],
) -> None:
    """让已完成日期重新进入审核队列，便于补入新文件夹后复审。"""
    completed = [name for name in dates if name in state.cleaning.completed_date_names]
    if not completed:
        return
    with st.expander("重新审核已完成日期", expanded=False):
        st.caption(
            "将清空所选日期的审核完成状态和当前标记；该日期下的文件夹会重新进入待审核队列。"
        )
        selected = st.selectbox("选择日期", completed, key="cleaning_reopen_date")
        if st.button("重新审核该日期", key="cleaning_reopen_confirm", width="stretch"):
            folders = scan_review_folders(context.data_root, [selected], None, set())
            reopen_date_review(state.cleaning, selected, folders)
            mark_folders_reviewed(
                context.data_root,
                [folder.key for folder in folders],
                reviewed=False,
            )
            _clear_date_cache()
            invalidate_ui_cache(*INVALIDATE_CLEANING)
            write_page_state(context, state)
            set_flash("cleaning", f"{selected} 已重新进入审核队列。")
            st.rerun()


def _render_restore_section(context: PageContext) -> None:
    """显示人工删除普通光谱恢复入口；小球目录不进入该流程。"""
    try:
        mapping = context.dataset_mapping()
    except UI_ACTION_ERRORS as error:
        render_action_error("Profile 配置加载", error)
        return
    try:
        records = _deleted_records(context.data_root, mapping)
    except UI_ACTION_ERRORS as error:
        render_action_error("删除数据扫描", error)
        return
    
    # 过滤小球已批准日期的记录
    bead_dates = {item.date_name for item in records if item.folder_name == BEAD_FOLDER_NAME}
    approved_bead_dates = {
        date_name
        for date_name in bead_dates
        if (load_calibration_report(context.shift_root, date_name) or {}).get("status")
        == "manual_approved"
    }
    records = tuple(
        item
        for item in records
        if item.folder_name != BEAD_FOLDER_NAME or item.date_name not in approved_bead_dates
    )
    
    # 有可恢复记录时保持展开，无记录时折叠
    has_records = len(records) > 0
    expand_restore = st.session_state.get("restore_expand", False) or has_records
    
    with st.expander("删除数据恢复", expanded=expand_restore):
        if not records:
            st.info("人工删除目录中没有可恢复的普通光谱。")
            return

        species_map = {
            (
                f"小球 / 小球 / {item.folder_name}"
                if item.folder_name == BEAD_FOLDER_NAME
                else f"{item.genus} / {item.prefix} / {item.folder_name}"
            ): (item.genus, item.prefix, item.folder_name)
            for item in records
        }
        species_options = sorted(species_map, key=build_natural_key)
        
        # 如果当前选择的物种不在列表中，自动顺延到下一个可用的物种
        current_species = st.session_state.get("restore_species_label")
        if current_species and current_species not in species_options and species_options:
            # 找到当前物种应该插入的位置，然后选择该位置之后的第一个物种
            # 手动比较 key 来找到位置
            current_key = build_natural_key(current_species)
            insert_pos = 0
            for i, species in enumerate(species_options):
                if build_natural_key(species) >= current_key:
                    insert_pos = i
                    break
            else:
                insert_pos = len(species_options)
            # 如果位置在末尾，循环到第一个；否则选择下一个
            next_species = species_options[0] if insert_pos >= len(species_options) else species_options[insert_pos]
            st.session_state["restore_species_label"] = next_species
            st.info(f"删除区中「{current_species}」已没有光谱，已自动切换到「{next_species}」。")
        
        species_label = st.selectbox("选择物种", species_options, key="restore_species_label")
        _, selected_prefix, selected_folder = species_map[species_label]
        date_options = sorted(
            {item.date_name for item in records if item.prefix == selected_prefix and item.folder_name == selected_folder},
            key=build_natural_key,
        )
        selected_date = st.selectbox("选择日期", date_options, key="restore_date_name")
        visible = [
            item
            for item in records
            if item.date_name == selected_date and item.prefix == selected_prefix and item.folder_name == selected_folder
        ]
        page_key = "restore_page_index"
        scope_key = "restore_scope"
        scope = f"{selected_date}/{selected_folder}"
        if st.session_state.get(scope_key) != scope:
            st.session_state[scope_key] = scope
            st.session_state[page_key] = 0
            st.session_state["restore_selected_paths"] = set()
            # 切换文件夹后必须重新确认，避免沿用上一批的彻底删除确认。
            st.session_state["restore_purge_confirm"] = False
        page_count, page_index = _page_bounds(
            len(visible),
            int(st.session_state.get(page_key, 0)),
        )
        selected_paths = set(st.session_state.get("restore_selected_paths", set()))
        selected_paths.intersection_update(item.relative_path for item in visible)
        st.session_state["restore_selected_paths"] = selected_paths
        st.caption(f"删除文件夹：{selected_folder} · 共 {len(visible)} 条 · 已选择 {len(selected_paths)} 条")
        _render_page_controls(
            page_index,
            page_count,
            key_prefix="restore",
            on_change=lambda index: _store_session_page(page_key, index),
        )

        items = visible[page_index * PAGE_SIZE:(page_index + 1) * PAGE_SIZE]
        page_paths = {item.relative_path for item in items}
        grid_key = f"restore-grid-{scope}-{page_index}"
        render_spectrum_grid(
            [_restore_card_data(context.data_root, item) for item in items],
            selected_paths=selected_paths.intersection(page_paths),
            marked_paths=selected_paths.intersection(page_paths),
            key=grid_key,
            on_change=lambda: _save_restore_selection(grid_key, page_paths),
        )

        confirm_purge = st.checkbox(
            "我确认彻底删除（不可恢复）",
            key="restore_purge_confirm",
            value=False,
        )
        actions = st.columns(3)
        if actions[0].button("清除当前选择", key="restore_clear_selection", width="stretch"):
            st.session_state["restore_selected_paths"] = set()
            st.rerun()
        if actions[1].button(
            "确认恢复",
            key="restore_confirm",
            disabled=not selected_paths,
            width="stretch",
        ):
            try:
                with st.status("正在恢复光谱…", expanded=True) as restore_status:
                    st.write("正在把人工删除的原始谱移回 cosmic_data。")
                    st.write("正在作废受影响日期的 init 记录（shift 由重建入口生成）。")
                    result = restore_deleted_spectra(
                        context.data_root,
                        context.shift_root,
                        context.init_root,
                        context.state_file,
                        mapping,
                        selected_paths,
                    )
                    restore_status.update(
                        label=f"恢复完成：{result['restored_file_count']} 条光谱",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.pop("restore_selected_paths", None)
                st.session_state["restore_expand"] = True
                _clear_date_cache()
                invalidate_ui_cache(*INVALIDATE_CLEANING_AFFINE_DATASET)
                if result.get("status") == "bead_restored":
                    set_flash(
                        "cleaning",
                        f"已恢复 {result['restored_file_count']} 条小球光谱；"
                        "小球谱已变，日期级仿射参数需到「仿射校准」页重新分析。",
                    )
                else:
                    set_flash(
                        "cleaning",
                        f"已恢复 {result['restored_file_count']} 条光谱；只改 cosmic_data，"
                        "shift 和 init 由重建入口更新。"
                        "shift 请在「仿射校准」页重建，init 由「数据集获取」页重建。",
                    )
                st.rerun()
            except UI_ACTION_ERRORS as error:
                render_action_error("恢复", error)
        if actions[2].button(
            "彻底删除",
            key="restore_purge_confirm_button",
            disabled=not selected_paths or not confirm_purge,
            width="stretch",
        ):
            try:
                with st.status("正在彻底删除…", expanded=True) as purge_status:
                    st.write(f"正在永久删除 {len(selected_paths)} 条光谱，无法恢复。")
                    st.write("正在清理变空的删除文件夹。")
                    result = purge_deleted_spectra(
                        context.data_root,
                        context.state_file,
                        selected_paths,
                    )
                    purge_status.update(
                        label=f"彻底删除完成：{result['purged_file_count']} 条光谱",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.pop("restore_selected_paths", None)
                st.session_state.pop("restore_purge_confirm", None)
                st.session_state["restore_expand"] = True
                invalidate_ui_cache(*INVALIDATE_CLEANING_DATASET)
                set_flash(
                    "cleaning",
                    f"已彻底删除 {result['purged_file_count']} 条光谱，"
                    "删除区空文件夹已同步清理。",
                )
                st.rerun()
            except UI_ACTION_ERRORS as error:
                render_action_error("彻底删除", error)


def _render_folder_removal_section(context: PageContext, state: PipelineState) -> None:
    """选择任意活动源文件夹，整夹归档并标记 init 待重建。"""
    # 下拉框只需要目录名，不能为每次选谱扫描全库并重写走向记录。
    dates = cached_directory_names(str(context.data_root.resolve()), ui_cache_version("cleaning"))
    source_ids = [
        f"{date_name}/{folder.name}"
        for date_name in dates
        for folder in sorted((context.data_root / date_name).glob("*"), key=lambda path: build_natural_key(path.name))
        if folder.is_dir() and folder.name not in SPECIAL_FOLDER_NAMES | NON_TRAINING_FOLDER_NAMES
        and not folder.name.startswith(".")
    ]
    # 有可移除文件夹时保持展开
    has_folders = len(source_ids) > 0
    expand_folder_remove = st.session_state.get("folder_remove_expand", False) or has_folders
    with st.expander("文件夹移除", expanded=expand_folder_remove):
        if not source_ids:
            st.info("当前没有可移除的文件夹。")
            return
        source_id = st.selectbox("选择要移除的文件夹", source_ids, key="cleaning_remove_source_folder")
        if not st.button("整夹删除", key="cleaning_remove_folder_confirm", width="stretch"):
            return
        try:
            date_name, folder_name = source_id.split("/", 1)
            commit_folder_items(
                context.data_root,
                state,
                context.state_file,
                ReviewFolder(source_id, date_name, folder_name, ()),
            )
            _clear_date_cache()
            invalidate_ui_cache(*INVALIDATE_CLEANING_DATASET)
            st.session_state.pop("cleaning_remove_source_folder", None)
            st.session_state["folder_remove_expand"] = True
            set_flash("cleaning", f"已整夹删除 {source_id}，init 待重建。")
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error("整夹删除", error)


def _render_folder_restore_section(context: PageContext) -> None:
    """显示整夹删除恢复入口；只移动 folder_delete 中的完整目录。"""
    try:
        folders = _deleted_folders(context.data_root)
    except UI_ACTION_ERRORS as error:
        render_action_error("整夹删除扫描", error)
        return
    # 有可恢复文件夹时保持展开
    has_folders = len(folders) > 0
    expand_folder_restore = st.session_state.get("folder_restore_expand", False) or has_folders
    with st.expander("整夹删除恢复", expanded=expand_folder_restore):
        if not folders:
            st.info("整夹删除目录中没有可恢复的文件夹。")
            return
        labels = [f"{item.date_name} / {item.folder_name}（{item.file_count} 条）" for item in folders]
        selected_label = st.selectbox("选择整夹", labels, key="folder_restore_source")
        selected = folders[labels.index(selected_label)]
        st.caption(
            "整夹恢复只把归档移回 cosmic_data，并让该文件夹回到待清洗队列；"
            "不写 shift。shift 到「仿射校准」页重建，init 由「数据集获取」页重建。"
        )
        if not st.button("恢复整夹到 cosmic_data", key="folder_restore_confirm", width="stretch"):
            return
        try:
            result = restore_deleted_folder(
                context.data_root,
                context.shift_root,
                context.init_root,
                context.state_file,
                None,
                selected.relative_path,
            )
            st.session_state.pop("folder_restore_source", None)
            st.session_state["folder_restore_expand"] = True
            _clear_date_cache()
            invalidate_ui_cache(*INVALIDATE_SPECTRA)
            set_flash(
                "cleaning",
                f"已恢复整夹 {selected.relative_path}（{result['restored_file_count']} 条），"
                "该文件夹已回到待清洗队列，只改 cosmic_data；"
                "shift 请在「仿射校准」页重建（重建会作废该日期审核结论）。",
            )
            st.rerun()
        except UI_ACTION_ERRORS as error:
            render_action_error("整夹恢复", error)


def _save_grid_selection(context: PageContext, grid_key: str, page_paths: set[str]) -> None:
    """在页面重跑前保存选谱事件，让计数、按钮和卡片一次刷新到位。"""
    event = st.session_state.get(grid_key)
    if not isinstance(event, dict) or event.get("event") != "selection":
        return
    state = load_page_state(context)
    chosen = {_normalize_relative_path(path) for path in event.get("selected_paths", [])}

    def remove(paths: set[str]) -> None:
        unmark_paths(state.cleaning, paths)

    def add(paths: set[str]) -> None:
        for path in paths:
            toggle_mark_path(state.cleaning, path)

    if not _apply_page_selection(page_paths, state.cleaning.marked_paths, chosen, remove=remove, add=add):
        return
    write_page_state(context, state)


def _store_folder_page(context: PageContext, state: PipelineState, folder_key: str, index: int) -> None:
    """保存主审核网格的文件夹页码并刷新页面。"""
    state.cleaning.folder_page_indices[folder_key] = index
    write_page_state(context, state)
    st.rerun()


def _save_restore_selection(grid_key: str, page_paths: set[str]) -> None:
    """删除恢复区与主网格共用同一套页内选中差集逻辑。"""
    event = st.session_state.get(grid_key)
    if not isinstance(event, dict) or event.get("event") != "selection":
        return
    selected = set(st.session_state.get("restore_selected_paths", set()))
    chosen = {_normalize_relative_path(path) for path in event.get("selected_paths", [])}
    if not _apply_page_selection(
        page_paths, selected, chosen, remove=selected.difference_update, add=selected.update
    ):
        return
    st.session_state["restore_selected_paths"] = selected


ANALYSIS_EDITOR_KEY = "analysis_removal_editor"
ANALYSIS_REPORT_KEY = "analysis_removal_report"


def _analysis_report() -> HeterogeneityReport | None:
    report = st.session_state.get(ANALYSIS_REPORT_KEY)
    return report if isinstance(report, HeterogeneityReport) else None


def _run_analysis_removal_analysis(context: PageContext, mapping, gate) -> None:
    def update(message: str, done: int, total: int) -> None:
        progress.progress(progress_ratio(done, total), text=f"{message}（{done}/{total}）")

    try:
        with st.status("正在分析全库光谱…", expanded=True) as status:
            progress = st.progress(0.0, text="扫描来源文件夹")
            report = analyze_library(
                context.shift_root,
                context.supplement_root,
                mapping,
                cosmic_folder_keys=set(gate.cosmic_folder_keys),
                progress_callback=update,
            )
            status.update(
                label=(
                    f"分析完成：{len(report.folders)} 个文件夹 / "
                    f"{report.sample_count} 条光谱"
                ),
                state="complete",
                expanded=False,
            )
    except UI_ACTION_ERRORS as error:
        render_action_error("分析", error)
        return
    st.session_state[ANALYSIS_REPORT_KEY] = report
    st.session_state.pop(ANALYSIS_EDITOR_KEY, None)
    st.rerun()


def _render_analysis_removal_section(context: PageContext, state: PipelineState) -> None:
    """全库高异常分析、预览勾选与二次确认剔除，只在清洗完成后出现。"""
    st.divider()
    st.subheader("全库高异常剔除")
    st.caption(
        "分析 shift_data 与 extra_data，用训练同一预处理链识别夹内少数子簇与样本级异常。"
        "剔除会归档到 dataset/analysis_delete，并删除 cosmic/shift/extra 中的对应光谱；"
        "init 不直接删除，只标记待重建。"
    )
    try:
        mapping = context.dataset_mapping()
    except UI_ACTION_ERRORS as error:
        render_action_error("Profile 配置加载", error)
        return
    gate = check_analysis_gate(
        context.data_root,
        context.shift_root,
        context.supplement_root,
        context.init_root,
        mapping,
        state.cleaning.folder_catalog,
        state.cleaning.completed_folder_keys,
        hierarchy=context.hierarchy,
    )
    report = _analysis_report()

    with st.container(border=True):
        if not gate.ready_enable:
            st.warning("当前还不能进行全库分析：")
            for issue in gate.issues[:8]:
                st.write(f"- {issue}")
            if len(gate.issues) > 8:
                st.caption(f"另有 {len(gate.issues) - 8} 条问题未显示，修正后重试。")
            if report is not None and st.button("清除已有分析结果", key="analysis_clear_stale"):
                st.session_state.pop(ANALYSIS_REPORT_KEY, None)
                st.session_state.pop(ANALYSIS_EDITOR_KEY, None)
                st.rerun()
            return

        actions = st.columns([1, 1, 2])
        if actions[0].button(
            "重新分析" if report is not None else "开始分析",
            key="analysis_run",
            width="stretch",
        ):
            _run_analysis_removal_analysis(context, mapping, gate)
        if report is not None and actions[1].button(
            "清除分析结果",
            key="analysis_clear",
            width="stretch",
        ):
            st.session_state.pop(ANALYSIS_REPORT_KEY, None)
            st.session_state.pop(ANALYSIS_EDITOR_KEY, None)
            st.rerun()

        if report is None:
            st.info(
                "点击开始分析。自动勾选只针对“少数样本子簇”的 minority 样本；"
                "双群分裂、整体弥散和单点异常只进入人工复核列表。"
            )
            return

        folder_frame = pd.DataFrame(report.folder_records())
        sample_frame = pd.DataFrame(report.sample_records())
        auto_count = int(sample_frame["auto_candidate"].sum())
        auto_folders = int((folder_frame["auto_candidate_count"] > 0).sum())
        metrics = st.columns(5)
        metrics[0].metric("分析文件夹", len(folder_frame))
        metrics[1].metric("分析光谱", report.sample_count)
        metrics[2].metric("自动候选夹", auto_folders)
        metrics[3].metric("自动候选光谱", auto_count)
        metrics[4].metric("跳过光谱", len(report.errors))
        st.caption(
            f"自动勾选条件：判定为「少数样本子簇」且子簇质心距离 ≥ "
            f"{AUTO_MIN_SEPARATION:.2f}。"
        )
        if report.errors:
            with st.expander(f"跳过的光谱（{len(report.errors)} 条）", expanded=False):
                for error in report.errors[:50]:
                    st.write(error)
                if len(report.errors) > 50:
                    st.caption(f"另有 {len(report.errors) - 50} 条未显示。")

        with st.expander("文件夹级判定", expanded=False):
            display_columns = [
                "folder_key",
                "label",
                "genus",
                "source_kind",
                "n_samples",
                "verdict",
                "minority_count",
                "cluster_separation",
                "dispersion_ratio",
                "drift_rho",
                "auto_candidate_count",
                "recommendation",
            ]
            st.dataframe(
                folder_frame[display_columns].sort_values(
                    ["auto_candidate_count", "n_samples"],
                    ascending=[False, False],
                ),
                hide_index=True,
                width="stretch",
            )

        review_frame = sample_frame[sample_frame["review_candidate"]].copy()
        show_all = st.checkbox(
            "显示全部样本（含未触发复核的样本）",
            value=False,
            key=f"analysis_show_all_{report.generated_at}",
        )
        if show_all:
            review_frame = sample_frame.copy()
        if review_frame.empty:
            st.info("没有进入复核列表的样本。")
            return
        review_frame = review_frame.sort_values(
            ["auto_candidate", "distance_rank"],
            ascending=[False, True],
        )
        editor_frame = pd.DataFrame(
            {
                "剔除": review_frame["auto_candidate"].astype(bool).to_numpy(),
                "文件夹": review_frame["folder_key"].to_numpy(),
                "样本": review_frame["file_name"].to_numpy(),
                "来源": review_frame["source_kind"].to_numpy(),
                "判定": review_frame["verdict"].to_numpy(),
                "子簇": review_frame["cluster"].to_numpy(),
                "子簇距离": review_frame["cluster_separation"].round(3).to_numpy(),
                "到质心": review_frame["distance_to_centroid"].round(3).to_numpy(),
                "排名": review_frame["distance_rank"].to_numpy(),
                "cos_z": review_frame["cos_z"].round(2).to_numpy(),
                "knn_z": review_frame["knn_z"].round(2).to_numpy(),
                "粗糙度z": review_frame["roughness_z"].round(2).to_numpy(),
                "投票": review_frame["outlier_vote"].to_numpy(),
                "自动候选": review_frame["auto_candidate"].to_numpy(),
            },
            index=review_frame["sample_id"].to_numpy(),
        )
        editor_key = f"{ANALYSIS_EDITOR_KEY}_{report.generated_at}"
        edited = st.data_editor(
            editor_frame,
            hide_index=True,
            width="stretch",
            disabled=[column for column in editor_frame.columns if column != "剔除"],
            column_config={
                "剔除": st.column_config.CheckboxColumn("剔除", width="small"),
            },
            key=editor_key,
        )
        selected_ids = [
            str(sample_id)
            for sample_id, remove in zip(edited.index, edited["剔除"].astype(bool))
            if remove
        ]
        selected_frame = sample_frame[sample_frame["sample_id"].isin(selected_ids)]
        cosmic_count = int((selected_frame["source_kind"] == "cosmic").sum())
        extra_count = int((selected_frame["source_kind"] == "extra").sum())
        st.caption(
            f"已选 {len(selected_ids)} 条：cosmic {cosmic_count} 条"
            f"（同时删除对应 shift 派生谱），extra {extra_count} 条；"
            "init 不会直接删除，只标记待重建。"
        )
        if st.button("重置为自动候选", key="analysis_reset_selection"):
            st.session_state.pop(editor_key, None)
            st.rerun()

        confirm = st.checkbox(
            "我确认把选中光谱归档到 dataset/analysis_delete，并从 cosmic/shift/extra 移除；"
            "init 标记待重建",
            value=False,
            key=f"analysis_confirm_{report.generated_at}",
        )
        if st.button(
            "执行剔除",
            key="analysis_execute",
            disabled=not selected_ids or not confirm,
            width="stretch",
        ):
            def update_removal(message: str, done: int, total: int) -> None:
                removal_progress.progress(progress_ratio(done, total), text=f"{message}（{done}/{total}）")

            try:
                with st.status("正在归档并更新派生记录…", expanded=True) as status:
                    removal_progress = st.progress(0.0, text="准备归档")
                    result = execute_analysis_removal(
                        context.data_root,
                        context.shift_root,
                        context.supplement_root,
                        state,
                        context.state_file,
                        report,
                        selected_ids,
                        progress_callback=update_removal,
                    )
                    status.update(
                        label=(
                            f"剔除完成：{result['sample_count']} 条光谱"
                            f"（cosmic {result['cosmic_count']} / extra {result['extra_count']}）"
                        ),
                        state="complete",
                        expanded=False,
                    )
                st.session_state.pop(ANALYSIS_REPORT_KEY, None)
                st.session_state.pop(editor_key, None)
                _clear_date_cache()
                invalidate_ui_cache(*INVALIDATE_ALL)
                set_flash(
                    "cleaning",
                    f"已剔除 {result['sample_count']} 条光谱，归档到 dataset/analysis_delete；"
                    f"清单见 {result['report_path']}；init 已标记待重建。",
                )
                st.rerun()
            except UI_ACTION_ERRORS as error:
                render_action_error("剔除", error)


def _render_auxiliary_sections(context: PageContext, state: PipelineState, *, include_analysis: bool = False) -> None:
    """渲染整夹恢复、文件夹移除、删除恢复与可选的全库异常剔除。"""
    _render_folder_restore_section(context)
    _render_folder_removal_section(context, state)
    _render_restore_section(context)
    if include_analysis and (
        not context.hierarchy.is_flat or context.dataset_spec.label_source
    ):
        _render_analysis_removal_section(context, state)


def render_cleaning_page(context: PageContext) -> None:
    state = load_page_state(context)
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        write_page_state(context, state)
    try:
        if not context.data_root.is_dir():
            raise FileNotFoundError(f"光谱根目录不存在：{context.data_root}")
        dates = list(cached_directory_names(str(context.data_root.resolve()), ui_cache_version("cleaning")))
    except FileNotFoundError as error:
        st.error(str(error))
        return
    # 空日期目录（无任何普通子文件夹）不产生审核项，剔除后可避免 _next_pending_date 反复选中同一日期。
    dates = [name for name in dates if _is_review_date_available(context.data_root, name)]

    st.title("数据清洗")
    if st.button("重新扫描", key="cleaning_rescan"):
        _clear_date_cache()
        invalidate_ui_cache(*INVALIDATE_ALL)
        st.rerun()
    if not dates:
        st.warning("dataset/cosmic_data 下没有找到日期目录。")
        _render_auxiliary_sections(context, state)
        return

    pending_dates = [date_name for date_name in dates if date_name not in state.cleaning.completed_date_names]
    current_date = state.cleaning.current_date_name
    if current_date not in pending_dates:
        current_date = pending_dates[0] if pending_dates else None
        state.cleaning.current_date_name = current_date
        state.cleaning.current_folder_key = None
        if current_date is not None:
            write_page_state(context, state)

    render_flash("cleaning")
    _render_reopen_section(context, state, dates)
    if current_date is None:
        st.success("区间内所有文件夹已审核")
        _render_auxiliary_sections(context, state, include_analysis=True)
        return

    try:
        catalog = _load_current_date(context, state, current_date)
    except FileNotFoundError as error:
        st.error(str(error))
        return
    folders = _current_folders(catalog, state)
    if not folders:
        refresh_completed_dates(state.cleaning)
        state.cleaning.current_date_name = _next_pending_date(dates, state)
        state.cleaning.current_folder_key = None
        _clear_date_cache()
        write_page_state(context, state)
        st.rerun()

    folder_keys = {folder.key for folder in folders}
    if state.cleaning.current_folder_key not in folder_keys:
        state.cleaning.current_folder_key = folders[0].key
        write_page_state(context, state)
    current_index = next(index for index, folder in enumerate(folders) if folder.key == state.cleaning.current_folder_key)
    current_folder = folders[current_index]

    page_count, page_index = _page_bounds(
        len(current_folder.relative_paths),
        state.cleaning.folder_page_indices.get(current_folder.key, 0),
    )
    state.cleaning.folder_page_indices[current_folder.key] = page_index
    current_pending = state.cleaning.marked_paths.intersection(current_folder.relative_paths)
    date_total = len(catalog)
    date_done = sum(key in state.cleaning.completed_folder_keys for key in catalog)

    st.caption(
        f"当前日期 {current_date} · 当前文件夹 {current_folder.folder_name} · 点击光谱卡片切换移除状态"
    )
    metrics = st.columns(5)
    metrics[0].metric("当前日期", current_date)
    metrics[1].metric("当前文件夹", current_folder.folder_name)
    metrics[2].metric("剩余光谱", f"{len(current_folder.relative_paths)} 条")
    metrics[3].metric("当前夹待移除", f"{len(current_pending)} 条")
    metrics[4].metric("日期进度", f"{date_done} / {date_total}")

    with st.container(border=True):
        _render_page_controls(
            page_index,
            page_count,
            key_prefix="cleaning_page",
            on_change=lambda index: _store_folder_page(
                context,
                state,
                current_folder.key,
                index,
            ),
        )
        row = st.columns(2)
        if row[0].button(
            "移除选中光谱",
            disabled=not current_pending,
            help="只移除已标记的光谱，继续审核本文件夹",
            width="stretch",
        ):
            try:
                result = commit_paths(context.data_root, state, context.state_file, current_pending)
            except UI_ACTION_ERRORS as error:
                render_action_error("移除", error)
            else:
                if result.missing_paths:
                    st.warning(f"有 {len(result.missing_paths)} 条文件不存在，未移除。")
                else:
                    set_flash("cleaning", f"已移除 {len(result.moved_paths)} 条光谱。")
                _clear_date_cache()
                invalidate_ui_cache(*INVALIDATE_CLEANING)
                write_page_state(context, state)
                st.rerun()
        if row[1].button(
            "完成当前夹",
            disabled=bool(current_pending),
            help="标记本文件夹已审核并进入下一个文件夹",
            width="stretch",
        ):
            _commit_current_folder(context, state, dates, catalog, current_folder)
            st.rerun()
        row = st.columns(2)
        if row[0].button("文件夹移除", width="stretch"):
            result = commit_folder_items(context.data_root, state, context.state_file, current_folder)
            if result.missing_paths:
                st.warning(f"有 {len(result.missing_paths)} 条文件不存在，当前夹暂未完成。")
            else:
                catalog.pop(current_folder.key, None)
                discard_removed_folder(state.cleaning, current_folder)
                _set_next_folder_or_date(state, dates, catalog)
                _clear_date_cache()
                invalidate_ui_cache(*INVALIDATE_CLEANING_DATASET)
                set_flash(
                    "cleaning",
                    f"已整夹删除 {current_folder.folder_name}，init 待重建。",
                )
                write_page_state(context, state)
                st.rerun()
        if row[1].button("日期移除", width="stretch"):
            # 日期移除必须重新读取磁盘，不能使用审核过程中已把完成文件夹路径清空的页面缓存。
            date_catalog = _load_current_date(
                context,
                state,
                current_date,
                force_enable=True,
            )
            result = commit_date_items(
                context.data_root,
                state,
                context.state_file,
                tuple(date_catalog.values()),
            )
            if result.missing_paths:
                st.warning(f"有 {len(result.missing_paths)} 条文件不存在。")
            discard_removed_date(state.cleaning, current_date)
            # 整日期移除后，旧的校正和 init 结果不再满足流水线门禁。
            state.affine_completed_dates.discard(current_date)
            state.cleaning.current_date_name = _next_pending_date(dates, state)
            state.cleaning.current_folder_key = None
            _clear_date_cache()
            invalidate_ui_cache(*INVALIDATE_CLEANING_DATASET)
            write_page_state(context, state)
            st.rerun()

    items = _folder_items(current_folder)[page_index * PAGE_SIZE:(page_index + 1) * PAGE_SIZE]
    page_paths = {item.relative_path for item in items}
    page_marked = state.cleaning.marked_paths.intersection(page_paths)
    grid_key = f"pipeline-grid-{current_folder.key}-{page_index}"
    render_spectrum_grid(
        _build_cards(context.data_root, items),
        selected_paths=page_marked,
        marked_paths=page_marked,
        key=grid_key,
        on_change=lambda: _save_grid_selection(context, grid_key, page_paths),
    )

    _render_auxiliary_sections(context, state)
    st.caption("完成所有文件夹审核后，可在本页进行全库高异常剔除。")
