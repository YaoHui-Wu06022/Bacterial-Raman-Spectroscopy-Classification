"""Raman Web 顶部流程导航。"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import streamlit as st

from ramanv2.common.paths import DATASET_ROOT
from ramanv2.data.datasets import DatasetRegistry, load_dataset_registry
from ramanv2.ui.page_context import PageContext, build_page_context
from ramanv2.ui.page_state import load_page_state, write_page_state


PageRenderer = Callable[[PageContext], None]

PAGE_ALIASES = {"层级训练": "训练配置"}
PAGE_TARGETS = {
    "数据清洗": ("ramanv2.ui.page_views.cleaning", "render_cleaning_page"),
    "仿射校准": ("ramanv2.ui.page_views.affine_calibration", "render_affine_calibration_page"),
    "数据集获取": ("ramanv2.ui.page_views.dataset_acquisition", "render_dataset_acquisition_page"),
    "训练配置": ("ramanv2.ui.page_views.model_training", "render_training_page"),
    "结果分析": ("ramanv2.ui.page_views.result_analysis", "render_result_analysis_page"),
    "样本推理": ("ramanv2.ui.page_views.sample_inference", "render_sample_inference_page"),
    "数据集配置": ("ramanv2.ui.page_views.dataset_config", "render_dataset_config_page"),
}
PAGE_NAMES = tuple(PAGE_TARGETS)
_DATASET_SESSION_KEYS = {"dataset_id", "active_page", "dataset_selector"}


def _load_page_renderer(page_name: str) -> PageRenderer:
    """按当前页面导入渲染函数，避免启动时加载全部页面依赖。"""
    module_name, function_name = PAGE_TARGETS[page_name]
    return getattr(importlib.import_module(module_name), function_name)


def _select_dataset_id(registry: DatasetRegistry) -> str:
    """在页面顶部选择数据集；切换时清理其他数据集的会话状态。"""
    specs = registry.list_datasets()
    dataset_ids = [spec.dataset_id for spec in specs]
    labels = {spec.dataset_id: spec.display_name for spec in specs}
    current = st.session_state.get("dataset_id", registry.default_dataset_id)
    if current not in dataset_ids:
        current = registry.default_dataset_id
    selected = st.selectbox(
        "数据集",
        dataset_ids,
        index=dataset_ids.index(current),
        format_func=lambda value: labels[value],
        key="dataset_selector",
    )
    if selected != current:
        for key in list(st.session_state.keys()):
            if key not in _DATASET_SESSION_KEYS:
                del st.session_state[key]
    st.session_state["dataset_id"] = selected
    return selected


def main() -> None:
    st.set_page_config(page_title="Raman 光谱流程", layout="wide")
    registry = load_dataset_registry(DATASET_ROOT)
    dataset_id = _select_dataset_id(registry)
    context = build_page_context(dataset_id, registry)
    if not context.has_dataset_config:
        _load_page_renderer("数据集配置")(context)
        return
    # 状态文件是跨页面和跨浏览器会话的唯一来源，避免使用旧会话对象覆盖新状态。
    state = load_page_state(context)
    current = st.session_state.get("active_page", state.active_page)
    current = PAGE_ALIASES.get(current, current)
    if current not in PAGE_NAMES:
        current = PAGE_NAMES[0]
    page_name = st.segmented_control(
        "页面",
        PAGE_NAMES,
        default=current,
        selection_mode="single",
        label_visibility="collapsed",
        key="active_page_control",
    ) or current
    st.session_state["active_page"] = page_name
    if state.active_page != page_name:
        state.active_page = page_name
        write_page_state(context, state)
    renderer = _load_page_renderer(page_name)
    renderer(context)
