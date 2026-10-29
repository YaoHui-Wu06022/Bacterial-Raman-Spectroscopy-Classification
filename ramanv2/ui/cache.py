"""Streamlit 页面中可安全复用的只读扫描缓存。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from ramanv2.data.count import collect_taxon_counts, summarize_dataset
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.common.naming import build_natural_key


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
    dates = []
    for path in root.iterdir():
        if not path.is_dir() or len(path.name) != 8 or not path.name.isdigit():
            continue
        try:
            parsed = datetime.strptime(path.name, "%Y%m%d")
        except ValueError:
            continue
        dates.append((parsed, path.name))
    return tuple(name for _, name in sorted(dates))


@st.cache_resource(show_spinner=False)
def cached_dataset_index(train_dir: str, version: int) -> DatasetIndex | None:
    """缓存统一种标签的只读目录索引，不加载全部强度数组。"""
    del version
    path = Path(train_dir)
    if not path.is_dir():
        return None
    try:
        return DatasetIndex(path, load_intensity_enable=False)
    except (OSError, RuntimeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def cached_json_file(path_string: str, modified_ns: int, size: int) -> dict[str, object] | None:
    """缓存页面只读 JSON；修改时间或大小变化会自动失效。"""
    del modified_ns, size
    path = Path(path_string)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


@st.cache_data(show_spinner=False)
def cached_file_bytes(path_string: str, modified_ns: int, size: int) -> bytes:
    """缓存下载文件内容，避免页面重跑时重复读取大压缩包。"""
    del modified_ns, size
    return Path(path_string).read_bytes()


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
def cached_experiment_dirs(project_root: str, profile_id: str, version: int) -> tuple[str, ...]:
    """缓存样本推理和结果分析共用的实验目录扫描。"""
    del version
    root = Path(project_root) / "output" / profile_id
    result: set[Path] = set()
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir() and (path / "hierarchy_meta.json").is_file():
                result.add(path.resolve())
    return tuple(str(path) for path in sorted(result, key=lambda path: build_natural_key(path.name), reverse=True))


@st.cache_data(show_spinner=False)
def cached_result_experiment_dirs(output_root: str, version: int) -> tuple[str, ...]:
    """缓存结果分析页的完整实验目录扫描。"""
    del version
    root = Path(output_root)
    result: set[Path] = set()
    markers = ("shared_config.yaml", "hierarchy_meta.json", "train_split.json", "val_split.json")
    if root.is_dir():
        for profile_dir in root.iterdir():
            if not profile_dir.is_dir():
                continue
            for path in profile_dir.iterdir():
                if path.is_dir() and all((path / marker).is_file() for marker in markers):
                    result.add(path.resolve())
    return tuple(str(path) for path in sorted(result, key=lambda path: str(path).lower(), reverse=True))


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
        if path.is_dir() and any(path.glob("*.arc_data"))
    )


__all__ = [
    "cached_dataset_index",
    "cached_dataset_summary",
    "cached_dataset_taxon_counts",
    "cached_directory_names",
    "cached_test_folder_names",
    "cached_experiment_dirs",
    "cached_file_bytes",
    "cached_json_file",
    "cached_run_dirs",
    "cached_result_experiment_dirs",
]
