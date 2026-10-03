"""数据清洗页面：当前日期、当前文件夹、当前页的增量审核。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import streamlit as st

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.naming import build_natural_key
from ramanv2.common.plotting import configure_matplotlib_fonts
from ramanv2.data.catalog import ensure_dataset_mapping
from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.pipeline.cleaning import (
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
    scan_review_folders,
    seed_default_marks,
    sync_folder_catalog,
    toggle_mark_path,
    unmark_paths,
)
from ramanv2.pipeline.restoration import DeletedSpectrum, restore_deleted_spectra, scan_deleted_spectra
from ramanv2.pipeline.state import PipelineState
from ramanv2.ui.components.spectrum_grid import render_spectrum_grid
from ramanv2.ui.cache import cached_directory_names
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.state import (
    invalidate_ui_cache,
    load_page_state as _load_state,
    save_page_state as _save,
    ui_cache_version,
)


configure_matplotlib_fonts()


PAGE_SIZE = 9
DATE_CACHE_KEY = "cleaning_current_date_cache"
DATE_CACHE_NAME_KEY = "cleaning_current_date_name"


@st.cache_data(show_spinner=False)
def _cached_deleted_records(
    data_root: str,
    workbook_path: str,
    mapping_path: str,
    mapping_modified_ns: int,
    version: int,
    include_bead: bool,
) -> tuple[DeletedSpectrum, ...]:
    """缓存删除恢复清单，写入或重新扫描后由版本号失效。"""
    del mapping_modified_ns, version
    mapping = ensure_dataset_mapping(workbook_path, mapping_path)
    return tuple(scan_deleted_spectra(Path(data_root), mapping, include_bead=include_bead))


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
        _save(context, state)
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


def _build_cards(root: Path, items: list[SpectrumItem]) -> list[dict[str, object]]:
    cards = []
    for item in items:
        path = root / item.relative_path
        card = {
            "relative_path": item.relative_path,
            "label": f"{item.date_name} / {item.folder_name} / {item.file_name}",
        }
        try:
            card["image_data"] = build_spectrum_image(str(path), path.stat().st_mtime_ns)
        except (OSError, ValueError) as error:
            card["error"] = f"读取失败：{error}"
        cards.append(card)
    return cards


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
    if not (context.data_root / folder.date_name / folder.folder_name).exists():
        catalog.pop(folder.key, None)
        discard_removed_folder(state.cleaning, folder)
    else:
        catalog[folder.key] = ReviewFolder(folder.key, folder.date_name, folder.folder_name, ())
    _set_next_folder_or_date(state, dates, catalog)
    _save(context, state)
    invalidate_ui_cache("cleaning")
    return True


def _restore_card_data(root: Path, record: DeletedSpectrum) -> dict[str, object]:
    """将删除目录中的光谱转换为选择网格卡片。"""
    path = root / record.source_relative_path
    card = {
        "relative_path": record.relative_path,
        "label": f"{record.date_name} / {record.folder_name} / {record.file_name}",
        "status_text": "待恢复",
    }
    try:
        card["image_data"] = build_spectrum_image(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError) as error:
        card["error"] = f"读取失败：{error}"
    return card


def _render_restore_section(context: PageContext) -> None:
    """显示人工删除普通光谱恢复入口；小球目录不进入该流程。"""
    with st.expander("删除数据恢复", expanded=False):
        try:
            mapping_modified_ns = (
                context.dataset_mapping_file.stat().st_mtime_ns
                if context.dataset_mapping_file.is_file()
                else 0
            )
            records = _cached_deleted_records(
                str(context.data_root.resolve()),
                str(context.classification_workbook.resolve()),
                str(context.dataset_mapping_file.resolve()),
                mapping_modified_ns,
                ui_cache_version("cleaning"),
                True,
            )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"删除数据扫描失败：{error}")
            return
        if not records:
            st.info("人工删除目录中没有可恢复的普通光谱。")
            return
        records = tuple(
            item
            for item in records
            if item.folder_name != "小球"
            or (
                (load_calibration_report(context.shift_root, item.date_name) or {}).get("status")
                != "manual_approved"
            )
        )
        if not records:
            st.info("当前人工删除目录中没有可恢复的光谱。")
            return
        try:
            mapping = ensure_dataset_mapping(context.classification_workbook, context.dataset_mapping_file)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Profile 配置加载失败：{error}")
            return

        species_map = {
            (
                f"小球 / 小球 / {item.folder_name}"
                if item.folder_name == "小球"
                else f"{item.genus} / {item.prefix} / {item.folder_name}"
            ): (item.genus, item.prefix, item.folder_name)
            for item in records
        }
        species_options = sorted(species_map, key=build_natural_key)
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
        page_count = max((len(visible) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        page_key = "restore_page_index"
        scope_key = "restore_scope"
        scope = f"{selected_date}/{selected_folder}"
        if st.session_state.get(scope_key) != scope:
            st.session_state[scope_key] = scope
            st.session_state[page_key] = 0
            st.session_state["restore_selected_paths"] = set()
        page_index = min(int(st.session_state.get(page_key, 0)), page_count - 1)
        selected_paths = set(st.session_state.get("restore_selected_paths", set()))
        selected_paths.intersection_update(item.relative_path for item in visible)
        st.session_state["restore_selected_paths"] = selected_paths
        st.caption(f"删除文件夹：{selected_folder} · 共 {len(visible)} 条 · 已选择 {len(selected_paths)} 条")

        toolbar = st.columns(3)
        if toolbar[0].button("上一页", disabled=page_index == 0, key="restore_previous_page", use_container_width=True):
            st.session_state[page_key] = page_index - 1
            st.rerun()
        if toolbar[1].button("下一页", disabled=page_index >= page_count - 1, key="restore_next_page", use_container_width=True):
            st.session_state[page_key] = page_index + 1
            st.rerun()
        toolbar[2].metric("当前页", f"{page_index + 1} / {page_count}")

        items = visible[page_index * PAGE_SIZE:(page_index + 1) * PAGE_SIZE]
        page_paths = {item.relative_path for item in items}
        event = render_spectrum_grid(
            [_restore_card_data(context.data_root, item) for item in items],
            selected_paths=selected_paths.intersection(page_paths),
            marked_paths=selected_paths.intersection(page_paths),
            key=f"restore-grid-{scope}-{page_index}",
        )
        if isinstance(event, dict):
            event_id = str(event.get("event_id", ""))
            if event_id and event_id != st.session_state.get("restore_grid_event"):
                st.session_state["restore_grid_event"] = event_id
                if event.get("event") == "selection":
                    chosen = {str(path).replace("\\", "/") for path in event.get("selected_paths", [])}
                    selected_paths.difference_update(page_paths)
                    selected_paths.update(chosen)
                    st.session_state["restore_selected_paths"] = selected_paths
                    st.rerun()

        actions = st.columns(2)
        if actions[0].button("清除当前选择", key="restore_clear_selection", use_container_width=True):
            st.session_state["restore_selected_paths"] = set()
            st.rerun()
        if actions[1].button(
            "确认恢复",
            key="restore_confirm",
            disabled=not selected_paths,
            use_container_width=True,
        ):
            try:
                with st.status("正在恢复光谱并重新执行文件夹 audit…", expanded=True) as restore_status:
                    st.write("正在恢复原始光谱并生成 shift_data。")
                    st.write("正在把旧 audit 删除谱合并回当前文件夹。")
                    st.write("正在执行新的文件夹 audit，并同步 init。")
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
                st.session_state.pop("pipeline_state", None)
                st.session_state["pipeline_state"] = _load_state(context)
                st.session_state.pop("restore_selected_paths", None)
                _clear_date_cache()
                invalidate_ui_cache("cleaning")
                invalidate_ui_cache("dataset")
                if result.get("status") == "bead_restored":
                    st.success(f"已恢复 {result['restored_file_count']} 条小球光谱，请到仿射校准页重新分析对应日期。")
                else:
                    st.success(f"已恢复 {result['restored_file_count']} 条光谱，并完成受影响文件夹 audit 和 init 同步。")
                st.rerun()
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"恢复失败：{error}")


def render_cleaning_page(context: PageContext) -> None:
    state = _load_state(context)
    if reconcile_cleaning_records(context.data_root, state.cleaning):
        _save(context, state)
    try:
        if not context.data_root.is_dir():
            raise FileNotFoundError(f"光谱根目录不存在：{context.data_root}")
        dates = list(cached_directory_names(str(context.data_root.resolve()), ui_cache_version("cleaning")))
    except FileNotFoundError as error:
        st.error(str(error))
        return

    st.sidebar.text(f"数据目录：{context.data_root}")
    st.sidebar.text(f"状态文件：{context.state_file}")
    if st.sidebar.button("重新扫描", use_container_width=True):
        _clear_date_cache()
        invalidate_ui_cache()
        st.rerun()
    if not dates:
        st.warning("dataset/cosmic_data 下没有找到日期目录。")
        return

    pending_dates = [date_name for date_name in dates if date_name not in state.cleaning.completed_date_names]
    current_date = state.cleaning.current_date_name
    if current_date not in pending_dates:
        current_date = pending_dates[0] if pending_dates else None
        state.cleaning.current_date_name = current_date
        state.cleaning.current_folder_key = None
        if current_date is not None:
            _save(context, state)

    st.title("数据清洗")
    if current_date is None:
        st.success("区间内所有文件夹已审核")
        _render_restore_section(context)
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
        _save(context, state)
        st.rerun()

    folder_keys = {folder.key for folder in folders}
    if state.cleaning.current_folder_key not in folder_keys:
        state.cleaning.current_folder_key = folders[0].key
        _save(context, state)
    current_index = next(index for index, folder in enumerate(folders) if folder.key == state.cleaning.current_folder_key)
    current_folder = folders[current_index]

    if current_folder.key not in state.cleaning.seeded_folder_keys:
        seed_default_marks([current_folder], state.cleaning)
        _save(context, state)

    page_count = max((len(current_folder.relative_paths) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    page_index = min(state.cleaning.folder_page_indices.get(current_folder.key, 0), page_count - 1)
    state.cleaning.folder_page_indices[current_folder.key] = page_index
    current_pending = state.cleaning.marked_paths.intersection(current_folder.relative_paths)
    date_total = len(catalog)
    date_done = sum(key in state.cleaning.completed_folder_keys for key in catalog)

    st.caption(f"当前日期 {current_date} · 当前文件夹 {current_folder.folder_name} · 点击光谱卡片切换移除状态")
    metrics = st.columns(5)
    metrics[0].metric("当前日期", current_date)
    metrics[1].metric("当前文件夹", current_folder.folder_name)
    metrics[2].metric("剩余光谱", f"{len(current_folder.relative_paths)} 条")
    metrics[3].metric("当前夹待移除", f"{len(current_pending)} 条")
    metrics[4].metric("日期进度", f"{date_done} / {date_total}")

    with st.container(border=True):
        row = st.columns(2)
        if row[0].button("上一页", disabled=page_index == 0, use_container_width=True):
            state.cleaning.folder_page_indices[current_folder.key] = page_index - 1
            _save(context, state)
            st.rerun()
        if row[1].button("下一页", use_container_width=True):
            if page_index < page_count - 1:
                state.cleaning.folder_page_indices[current_folder.key] = page_index + 1
                _save(context, state)
            else:
                _commit_current_folder(context, state, dates, catalog, current_folder)
            st.rerun()
        row = st.columns(2)
        if row[0].button("文件夹移除", use_container_width=True):
            result = commit_folder_items(context.data_root, state, context.state_file, current_folder)
            if result.missing_paths:
                st.warning(f"有 {len(result.missing_paths)} 条文件不存在，当前夹暂未完成。")
            else:
                catalog.pop(current_folder.key, None)
                discard_removed_folder(state.cleaning, current_folder)
                _set_next_folder_or_date(state, dates, catalog)
                _save(context, state)
                st.rerun()
        if row[1].button("日期移除", use_container_width=True):
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
            # 整日期移除后，旧的校正、audit 和 init 结果不再满足流水线门禁。
            state.affine_completed_dates.discard(current_date)
            state.audit_completed_dates.discard(current_date)
            state.dataset_builds.pop("latest", None)
            state.cleaning.current_date_name = _next_pending_date(dates, state)
            state.cleaning.current_folder_key = None
            _clear_date_cache()
            invalidate_ui_cache("cleaning")
            invalidate_ui_cache("dataset")
            _save(context, state)
            st.rerun()
        st.metric("当前页", f"{page_index + 1} / {page_count}")

    items = _folder_items(current_folder)[page_index * PAGE_SIZE:(page_index + 1) * PAGE_SIZE]
    page_paths = {item.relative_path for item in items}
    page_marked = state.cleaning.marked_paths.intersection(page_paths)
    event = render_spectrum_grid(
        _build_cards(context.data_root, items),
        selected_paths=page_marked,
        marked_paths=page_marked,
        key=f"pipeline-grid-{current_folder.key}-{page_index}",
    )
    if isinstance(event, dict):
        event_id = str(event.get("event_id", ""))
        if event_id and event_id != st.session_state.get("pipeline_grid_event"):
            st.session_state["pipeline_grid_event"] = event_id
            if event.get("event") == "selection":
                chosen = {str(path).replace("\\", "/") for path in event.get("selected_paths", [])}
                for path in page_marked - chosen:
                    unmark_paths(state.cleaning, [path])
                for path in chosen - page_marked:
                    toggle_mark_path(state.cleaning, path)
                _save(context, state)
                st.rerun()

    _render_restore_section(context)


__all__ = ["render_cleaning_page"]
