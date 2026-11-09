"""页面共用的 Profile 选择控件。"""

from __future__ import annotations

import streamlit as st

from ramanv2.data.profiles import DatasetProfile
from ramanv2.ui.page_context import PageContext


def render_profile_picker(
    context: PageContext,
    *,
    label: str,
    key: str,
    profile_ids: list[str] | None = None,
    empty_message: str = "当前没有可用 Profile。",
    empty_kind: str = "warning",
) -> DatasetProfile | None:
    """渲染 Profile 选择器；单 Profile 时直接返回，不显示下拉框。"""
    available = (
        list(profile_ids)
        if profile_ids is not None
        else [profile.profile_id for profile in context.profiles]
    )
    if not available:
        getattr(st, empty_kind)(empty_message)
        return None
    if len(available) == 1:
        return context.profile(available[0])
    default_profile = context.default_profile_id
    selected = st.selectbox(
        label,
        available,
        index=available.index(default_profile) if default_profile in available else 0,
        key=key,
    )
    return context.profile(selected)
