"""Streamlit 页面共用的流水线状态缓存。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ramanv2.pipeline.state import PipelineState, load_pipeline_state, save_pipeline_state
from ramanv2.pipeline.lineage import sync_pipeline_state_from_lineage
from ramanv2.ui.page_context import PageContext


UI_CACHE_NAMES = ("cleaning", "affine", "dataset", "inference", "training", "analysis")


@st.cache_resource(show_spinner=False)
def _cache_version_registry() -> dict[str, int]:
    """进程级缓存版本表。

    ``st.cache_data`` 的结果是跨会话共享的，版本号必须同作用域：若存于
    ``st.session_state``，新会话（刷新页面、新标签页）会把版本号重置回 0，
    从而命中旧会话留下的陈旧缓存。
    """
    return {}


def ui_cache_version(name: str) -> int:
    """返回进程内缓存版本，用于让写入操作精准失效缓存。"""
    if name not in UI_CACHE_NAMES:
        raise ValueError(f"未知 UI 缓存：{name}")
    return int(_cache_version_registry().get(name, 0))


def invalidate_ui_cache(name: str | None = None) -> None:
    """递增指定页面缓存版本；不清理其他页面的缓存。"""
    names = UI_CACHE_NAMES if name is None else (name,)
    registry = _cache_version_registry()
    for cache_name in names:
        if cache_name not in UI_CACHE_NAMES:
            raise ValueError(f"未知 UI 缓存：{cache_name}")
        registry[cache_name] = int(registry.get(cache_name, 0)) + 1


@st.cache_data(show_spinner=False)
def _cached_lineage_date_sets(data_root: str, lineage_modified_ns: int) -> dict[str, tuple[str, ...]]:
    """按 folder_lineage.json 的 mtime 缓存派生日期集合，避免每次页面运行重解析 2.5MB JSON。

    走向文件变化时 mtime 随之更新，缓存自动失效。
    """
    del lineage_modified_ns
    derived = PipelineState()
    sync_pipeline_state_from_lineage(Path(data_root), derived)
    return {
        "cleaning": tuple(sorted(derived.cleaning.completed_date_names)),
        "affine": tuple(sorted(derived.affine_completed_dates)),
        "init": tuple(sorted(derived.init_generated_dates)),
    }


def load_page_state(context: PageContext) -> PipelineState:
    """每次页面运行都从磁盘读取共享状态，避免旧会话覆盖新状态。"""
    state = load_pipeline_state(context.state_file)
    lineage_file = context.data_root.parent / ".raman_pipeline" / "folder_lineage.json"
    if context.data_root.is_dir() and lineage_file.is_file():
        derived = _cached_lineage_date_sets(
            str(context.data_root.resolve()),
            lineage_file.stat().st_mtime_ns,
        )
        state.cleaning.completed_date_names = set(derived["cleaning"])
        state.affine_completed_dates = set(derived["affine"])
        state.init_generated_dates = set(derived["init"])
    st.session_state["pipeline_state"] = state
    return state


def save_page_state(context: PageContext, state: PipelineState) -> None:
    """持久化页面共享状态并刷新当前会话缓存；与磁盘内容一致时跳过写盘。"""
    payload = state.to_payload()
    payload.pop("updated_at", None)
    current = load_pipeline_state(context.state_file).to_payload()
    current.pop("updated_at", None)
    if payload == current:
        st.session_state["pipeline_state"] = state
        return
    save_pipeline_state(context.state_file, state)
    st.session_state["pipeline_state"] = state


__all__ = ["UI_CACHE_NAMES", "invalidate_ui_cache", "load_page_state", "save_page_state", "ui_cache_version"]
