"""Streamlit 页面中可安全复用的只读扫描缓存。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ramanv2.common.arc_data import has_arc_data
from ramanv2.common.file_ops import read_json
from ramanv2.common.naming import build_natural_key
from ramanv2.core.config_snapshot import SHARED_CONFIG_NAME
from ramanv2.core.run_context import (
    HIERARCHY_META_FILE_NAME,
    TRAIN_SPLIT_FILE_NAME,
    VALIDATION_SPLIT_FILE_NAME,
)
from ramanv2.data.calibration.affine import load_calibration_report
from ramanv2.data.count import collect_taxon_counts, summarize_dataset
from ramanv2.data.hierarchy_config import HierarchySpec
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.pipeline.cleaning import scan_date_names


_EXPERIMENT_MARKERS = (
    SHARED_CONFIG_NAME,
    TRAIN_SPLIT_FILE_NAME,
    VALIDATION_SPLIT_FILE_NAME,
    HIERARCHY_META_FILE_NAME,
)
_EXPERIMENT_MIN_MARKERS = (HIERARCHY_META_FILE_NAME,)


@st.cache_data(show_spinner=False)
def cached_dataset_summary(root_dir: str, version: int) -> dict[str, int | bool]:
    """缓存数据集统计；版本由页面写入操作主动递增。"""
    del version
    return summarize_dataset(Path(root_dir))


@st.cache_data(show_spinner=False)
def cached_dataset_taxon_counts(root_dir: str, version: int) -> dict[str, list[dict[str, str | int]]]:
    """缓存属/种明细；构建、补偿及删除同步后沿用 dataset 版本失效。"""
    del version
    return collect_taxon_counts(Path(root_dir))


@st.cache_data(show_spinner=False)
def cached_directory_names(root_dir: str, version: int) -> tuple[str, ...]:
    """缓存日期目录名称，外部变更由重新扫描按钮刷新。"""
    del version
    root = Path(root_dir)
    if not root.is_dir():
        return ()
    return tuple(scan_date_names(root))


@st.cache_resource(show_spinner=False)
def cached_dataset_index(train_dir: str, version: int, hierarchy_json: str) -> DatasetIndex | None:
    """缓存统一种标签的只读目录索引，不加载全部强度数组。"""
    del version
    path = Path(train_dir)
    if not path.is_dir():
        return None
    try:
        return DatasetIndex(
            path,
            hierarchy=HierarchySpec.from_payload(json.loads(hierarchy_json)),
            load_intensity_enable=False,
        )
    except (OSError, RuntimeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def cached_json_file(path_string: str, modified_ns: int, size: int) -> dict[str, object] | None:
    """缓存页面只读 JSON；修改时间或大小变化会自动失效。"""
    del modified_ns, size
    path = Path(path_string)
    try:
        payload = read_json(path)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


@st.cache_data(show_spinner=False)
def cached_file_bytes(path_string: str, modified_ns: int, size: int) -> bytes:
    """缓存下载文件内容，避免页面重跑时重复读取大压缩包。"""
    del modified_ns, size
    return Path(path_string).read_bytes()


@st.cache_data(show_spinner=False)
def cached_text_file(path_string: str, modified_ns: int, size: int) -> str:
    """缓存日志/摘要等只读文本；修改时间或大小变化会自动失效。"""
    del modified_ns, size
    try:
        return Path(path_string).read_text(encoding="utf-8")
    except OSError:
        return ""


@st.cache_data(show_spinner=False)
def cached_calibration_reports(
    shift_root: str,
    report_fingerprint: tuple[tuple[str, int, int], ...],
) -> dict[str, dict[str, object] | None]:
    """缓存日期级仿射报告；报告文件的 mtime/size 指纹变化时自动失效。"""
    root = Path(shift_root)
    return {
        date_name: load_calibration_report(root, date_name)
        for date_name, _, _ in report_fingerprint
    }


@st.cache_data(show_spinner=False)
def cached_run_dirs(level_dir: str, version: int) -> tuple[str, ...]:
    """缓存结果分析页的 run 目录递归扫描。"""
    del version
    root = Path(level_dir)
    if not root.is_dir():
        return ()
    return tuple(
        str(path.resolve())
        for path in sorted(
            (path for path in root.rglob("run_*") if path.is_dir()),
            key=lambda path: build_natural_key(path.as_posix()),
        )
    )


@st.cache_data(show_spinner=False)
def cached_experiment_dirs(
    experiment_parent: str,
    require_complete: bool,
    version: int,
) -> tuple[str, ...]:
    """缓存一个输出命名空间下的实验目录；完整模式要求切分与配置快照齐全。"""
    del version
    root = Path(experiment_parent)
    if not root.is_dir():
        return ()
    markers = _EXPERIMENT_MARKERS if require_complete else _EXPERIMENT_MIN_MARKERS
    result = [
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and all((path / marker).is_file() for marker in markers)
    ]
    return tuple(
        str(path)
        for path in sorted(result, key=lambda item: build_natural_key(item.name), reverse=True)
    )


@st.cache_data(show_spinner=False)
def cached_test_folder_names(root_dir: str, version: int) -> tuple[str, ...]:
    """缓存 init_test 的文件夹扫描。"""
    del version
    root = Path(root_dir)
    if not root.is_dir():
        return ()
    return tuple(
        path.name
        for path in sorted(root.iterdir(), key=lambda item: build_natural_key(item.name))
        if path.is_dir() and has_arc_data(path)
    )
