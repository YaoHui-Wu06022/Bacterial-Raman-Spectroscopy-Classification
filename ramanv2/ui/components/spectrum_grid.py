"""3×3 光谱选择组件，负责点击选择事件回传。"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "spectrum_selection_grid",
    path=str(Path(__file__).with_name("spectrum_selection_grid")),
)


def render_spectrum_grid(
    cards: list[dict[str, object]],
    selected_paths: set[str],
    marked_paths: set[str],
    key: str,
) -> dict[str, object] | None:
    """渲染光谱网格并返回最近一次点击事件。"""
    return _COMPONENT(
        cards=cards,
        selected_paths=sorted(selected_paths),
        marked_paths=sorted(marked_paths),
        key=key,
        default={"event": "none", "event_id": "initial"},
    )
