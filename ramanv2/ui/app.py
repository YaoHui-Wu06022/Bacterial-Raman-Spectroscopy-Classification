"""Raman Web 顶部流程导航。"""

from __future__ import annotations

import streamlit as st

from ramanv2.pipeline.state import load_pipeline_state, save_pipeline_state
from ramanv2.ui.page_context import build_page_context
from ramanv2.ui.pages.affine_calibration import render_affine_calibration_page
from ramanv2.ui.pages.cleaning import render_cleaning_page
from ramanv2.ui.pages.dataset_acquisition import render_dataset_acquisition_page
from ramanv2.ui.pages.hierarchical_training import render_hierarchical_training_page
from ramanv2.ui.pages.result_analysis import render_result_analysis_page
from ramanv2.ui.pages.sample_inference import render_sample_inference_page


PAGE_NAMES = ("数据清洗", "仿射校准", "数据集获取", "层级训练", "结果分析", "样本推理")


def main() -> None:
    st.set_page_config(page_title="Raman 光谱流程", layout="wide")
    context = build_page_context()
    # 状态文件是跨页面和跨浏览器会话的唯一来源，避免使用旧会话对象覆盖新状态。
    state = load_pipeline_state(context.state_file)
    st.session_state["pipeline_state"] = state
    current = st.session_state.get("active_page", state.active_page)
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
        save_pipeline_state(context.state_file, state)

    if page_name == "数据清洗":
        render_cleaning_page(context)
    elif page_name == "仿射校准":
        render_affine_calibration_page(context)
    elif page_name == "数据集获取":
        render_dataset_acquisition_page(context)
    elif page_name == "层级训练":
        render_hierarchical_training_page()
    elif page_name == "结果分析":
        render_result_analysis_page()
    else:
        render_sample_inference_page()


__all__ = ["PAGE_NAMES", "main"]
