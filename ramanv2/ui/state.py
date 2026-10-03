"""Streamlit 页面共用的流水线状态缓存。"""

from __future__ import annotations

import streamlit as st

from ramanv2.pipeline.state import PipelineState, load_pipeline_state, save_pipeline_state
from ramanv2.pipeline.lineage import sync_pipeline_state_from_lineage
from ramanv2.ui.page_context import PageContext


UI_CACHE_NAMES = ("cleaning", "dataset", "inference", "training", "analysis")


def ui_cache_version(name: str) -> int:
    """返回一个会话内缓存版本，用于让写入操作精准失效缓存。"""
    if name not in UI_CACHE_NAMES:
        raise ValueError(f"未知 UI 缓存：{name}")
    versions = st.session_state.setdefault("ui_cache_versions", {})
    return int(versions.get(name, 0))


def invalidate_ui_cache(name: str | None = None) -> None:
    """递增指定页面缓存版本；不清理其他页面的缓存。"""
    names = UI_CACHE_NAMES if name is None else (name,)
    versions = st.session_state.setdefault("ui_cache_versions", {})
    for cache_name in names:
        if cache_name not in UI_CACHE_NAMES:
            raise ValueError(f"未知 UI 缓存：{cache_name}")
        versions[cache_name] = int(versions.get(cache_name, 0)) + 1


def load_page_state(context: PageContext) -> PipelineState:
    """每次页面运行都从磁盘读取共享状态，避免旧会话覆盖新状态。"""
    state = load_pipeline_state(context.state_file)
    if context.data_root.is_dir() and (context.data_root.parent / ".raman_pipeline" / "folder_lineage.json").is_file():
        sync_pipeline_state_from_lineage(context.data_root, state)
    st.session_state["pipeline_state"] = state
    return state


def save_page_state(context: PageContext, state: PipelineState) -> None:
    """持久化页面共享状态并刷新当前会话缓存。"""
    save_pipeline_state(context.state_file, state)
    st.session_state["pipeline_state"] = state


__all__ = ["UI_CACHE_NAMES", "invalidate_ui_cache", "load_page_state", "save_page_state", "ui_cache_version"]
