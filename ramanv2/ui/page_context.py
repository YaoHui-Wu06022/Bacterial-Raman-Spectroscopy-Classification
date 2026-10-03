"""Streamlit 页面共享的路径上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.core.paths import PROJECT_ROOT
from ramanv2.pipeline.state import pipeline_metadata_dir, pipeline_state_path


@dataclass(frozen=True)
class PageContext:
    """页面之间共享的固定数据目录，不保存页面交互状态。"""

    data_root: Path
    shift_root: Path
    init_root: Path
    state_file: Path
    metadata_dir: Path
    classification_workbook: Path
    dataset_mapping_file: Path


def build_page_context() -> PageContext:
    data_root = PROJECT_ROOT / "dataset" / "cosmic_data"
    return PageContext(
        data_root=data_root,
        shift_root=PROJECT_ROOT / "dataset" / "shift_data",
        init_root=PROJECT_ROOT / "dataset" / "init",
        state_file=pipeline_state_path(PROJECT_ROOT),
        metadata_dir=pipeline_metadata_dir(PROJECT_ROOT),
        classification_workbook=PROJECT_ROOT / "dataset" / "病原菌分类与规范简称.xlsx",
        dataset_mapping_file=pipeline_metadata_dir(PROJECT_ROOT) / "profile_catalog.json",
    )


__all__ = ["PageContext", "build_page_context"]
