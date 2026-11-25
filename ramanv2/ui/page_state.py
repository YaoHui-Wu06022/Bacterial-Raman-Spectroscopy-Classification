"""Streamlit 页面共用的流水线状态缓存。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ramanv2.pipeline.state import PipelineState, load_pipeline_state, write_pipeline_state
from ramanv2.pipeline.lineage import lineage_path, load_pipeline_progress
from ramanv2.ui.page_context import PageContext


UI_CACHE_NAMES = ("cleaning", "affine", "dataset", "inference", "training", "analysis")

# 页面动作可能抛出的异常；统一后不再每个调用点各写一套元组。
UI_ACTION_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError)

# 缓存失效语义：页面只描述“发生了什么”，不手写缓存名组合。
INVALIDATE_CLEANING = ("cleaning",)
INVALIDATE_CLEANING_DATASET = ("cleaning", "dataset")
INVALIDATE_CLEANING_AFFINE_DATASET = ("cleaning", "affine", "dataset")
INVALIDATE_DATASET = ("dataset", "training", "inference", "analysis")
INVALIDATE_TRAINING = ("training", "inference", "analysis")
INVALIDATE_ANALYSIS = ("analysis",)
INVALIDATE_INFERENCE = ("inference", "dataset")
INVALIDATE_SPECTRA = UI_CACHE_NAMES
INVALIDATE_ALL = UI_CACHE_NAMES

_FLASH_KEY = "ui_flash"


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


def invalidate_ui_cache(*names: str) -> None:
    """递增给定页面的缓存版本；不传名称时清理全部页面。"""
    registry = _cache_version_registry()
    for cache_name in names or UI_CACHE_NAMES:
        if cache_name not in UI_CACHE_NAMES:
            raise ValueError(f"未知 UI 缓存：{cache_name}")
        registry[cache_name] = int(registry.get(cache_name, 0)) + 1


def set_flash(slot: str, message: str, kind: str = "success") -> None:
    """记录一次性提示，供下一次 rerun 在指定位置展示。"""
    flashes = st.session_state.get(_FLASH_KEY)
    if not isinstance(flashes, dict):
        st.session_state[_FLASH_KEY] = flashes = {}
    flashes[slot] = (kind, message)


def render_flash(slot: str) -> None:
    """取出并展示指定位置的一次性提示。"""
    flashes = st.session_state.get(_FLASH_KEY)
    if not isinstance(flashes, dict):
        return
    entry = flashes.pop(slot, None)
    if not entry:
        return
    kind, message = entry
    getattr(st, kind)(message)


def render_action_error(label: str, error: Exception) -> None:
    """按统一格式显示一次页面动作失败。"""
    st.error(f"{label}失败：{error}")


@st.cache_data(show_spinner=False)
def _cached_pipeline_progress(
    data_root: str, state_path: str, state_modified_ns: int, lineage_modified_ns: int,
) -> PipelineState:
    """缓存同一 SQLite 版本下的流水线门禁。"""
    return load_pipeline_progress(data_root, state_path)


def load_page_state(context: PageContext) -> PipelineState:
    """读取业务门禁；数据库变更后自动失效页面缓存。"""
    state_file = context.state_file
    lineage_file = lineage_path(context.data_root)
    return _cached_pipeline_progress(
        str(context.data_root.resolve()),
        str(state_file),
        state_file.stat().st_mtime_ns if state_file.is_file() else 0,
        lineage_file.stat().st_mtime_ns if lineage_file.is_file() else 0,
    )


def write_page_state(context: PageContext, state: PipelineState) -> None:
    """持久化页面共享状态并刷新当前会话缓存；与磁盘内容一致时跳过写盘。"""
    payload = state.to_payload()
    payload.pop("updated_at", None)
    current = load_pipeline_state(context.state_file).to_payload()
    current.pop("updated_at", None)
    if payload == current:
        return
    write_pipeline_state(context.state_file, state)
