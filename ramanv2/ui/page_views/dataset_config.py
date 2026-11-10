"""数据集配置页：选择数据集文件夹并维护 dataset.json。"""

from __future__ import annotations

import re

import streamlit as st

from ramanv2.data.datasets import write_dataset_config
from ramanv2.ui.page_context import PageContext
from ramanv2.ui.page_state import (
    INVALIDATE_ALL,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    render_action_error,
)


_BAD_BAND_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-~,，]\s*(\d+(?:\.\d+)?)\s*$")


def _parse_bad_bands(text: str) -> tuple[tuple[float, float], ...]:
    """解析“900-940, 1000-1010”形式的坏区。"""
    items = [item for item in re.split(r"[;；\n]", text or "") if item.strip()]
    bands = []
    for item in items:
        match = _BAD_BAND_PATTERN.fullmatch(item)
        if match is None:
            raise ValueError(f"坏区格式无效：{item}（示例：900-940）")
        bands.append((float(match.group(1)), float(match.group(2))))
    return tuple(bands)


def _format_bad_bands(bands: tuple[tuple[float, float], ...]) -> str:
    return ", ".join(f"{start:g}-{end:g}" for start, end in bands)


def render_dataset_config_page(context: PageContext) -> None:
    """选择数据集目录并保存显示名、数据集 ID 与坏区。"""
    st.title("数据集配置")
    st.caption(
        "先把数据放到 dataset/<文件夹>/cosmic_data，再在这里选择文件夹；"
        "显示名默认使用文件夹名。"
    )
    if st.button("刷新数据集列表", key="dataset_config_refresh"):
        st.rerun()

    registry = context.registry
    specs = registry.list_datasets()
    if not specs:
        st.warning("dataset 下没有包含 cosmic_data 的数据集目录。")
        return

    dataset_ids = [spec.dataset_id for spec in specs]
    current_id = context.dataset_spec.dataset_id
    if current_id not in dataset_ids:
        current_id = dataset_ids[0]
    selected_id = st.selectbox(
        "数据集文件夹",
        dataset_ids,
        index=dataset_ids.index(current_id),
        format_func=lambda value: next(
            (
                f"{spec.directory.name}（{spec.dataset_id}）"
                + ("" if spec.config_path.is_file() else " · 未配置")
                for spec in specs
                if spec.dataset_id == value
            ),
            value,
        ),
        key="dataset_config_folder",
    )
    spec = registry.resolve(selected_id)
    st.caption(f"文件夹路径：{spec.dataset_dir}")

    configured = spec.config_path.is_file()
    default_display_name = spec.display_name if configured else spec.directory.name
    default_dataset_id = spec.dataset_id if configured else spec.directory.name
    default_bands = _format_bad_bands(spec.bad_bands)
    with st.form("dataset_config_form"):
        display_name = st.text_input("显示名", value=default_display_name)
        dataset_id = st.text_input(
            "数据集 ID（ASCII，用于命令与配置）",
            value=default_dataset_id,
        )
        bad_bands_text = st.text_input(
            "坏区（多个用逗号分隔，留空表示无）",
            value=default_bands,
        )
        submitted = st.form_submit_button("保存数据集配置", width="stretch")

    if submitted:
        try:
            dataset_id = dataset_id.strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_id):
                raise ValueError("数据集 ID 只能包含字母、数字、下划线或短横线")
            bands = _parse_bad_bands(bad_bands_text)
            write_dataset_config(
                spec.dataset_dir,
                display_name,
                bands,
                dataset_id=dataset_id,
                aliases=spec.aliases,
                profiles=spec.profiles,
                workbook_columns=spec.workbook_columns,
                profile_rules=spec.profile_rules,
                is_default=spec.is_default,
            )
        except UI_ACTION_ERRORS as error:
            render_action_error("保存", error)
            return
        st.session_state["dataset_id"] = dataset_id
        invalidate_ui_cache(*INVALIDATE_ALL)
        st.rerun()
