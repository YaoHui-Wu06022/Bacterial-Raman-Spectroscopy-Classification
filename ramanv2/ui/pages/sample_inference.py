"""样本推理、CS 文件夹筛选和测试输入缓存页面。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from ramanv2.common.naming import build_natural_key, parse_test_folder_prefix
from ramanv2.core.paths import DATASET_ROOT, PROJECT_ROOT
from ramanv2.core.runtime import resolve_device
from ramanv2.data.profiles import get_dataset_dir, list_profiles
from ramanv2.inference.cache import build_test_cache
from ramanv2.inference.labels import build_expected_label_lookup
from ramanv2.inference.predictor import load_predictor
from ramanv2.inference.runner import run_independent_inference
from ramanv2.ui.cache import (
    cached_cs_folder_names,
    cached_experiment_dirs,
    cached_json_file,
)
from ramanv2.ui.state import invalidate_ui_cache, ui_cache_version


CS_ROOT = DATASET_ROOT / "CSdata"


def render_sample_inference_page() -> None:
    """渲染实验选择、测试缓存、样本推理和结果查看区域。"""
    st.title("样本推理")
    profiles = list_profiles()
    if not profiles:
        st.warning("当前没有可用 Profile。")
        return
    profile_ids = [profile.profile_id for profile in profiles]
    profile_id = st.selectbox(
        "选择 Profile",
        profile_ids,
        index=profile_ids.index("GN") if "GN" in profile_ids else 0,
        key="inference_profile",
    )
    experiment_dirs = _list_experiment_dirs(profile_id)
    if not experiment_dirs:
        st.info(f"没有找到 {profile_id} 的有效实验目录。")
        return
    experiment_labels = [str(path.relative_to(PROJECT_ROOT)) for path in experiment_dirs]
    selected_label = st.selectbox("选择实验目录", experiment_labels, key="inference_experiment")
    experiment_dir = experiment_dirs[experiment_labels.index(selected_label)]
    meta_path = experiment_dir / "hierarchy_meta.json"
    meta_stat = meta_path.stat()
    meta = cached_json_file(str(meta_path.resolve()), meta_stat.st_mtime_ns, meta_stat.st_size)
    if not isinstance(meta, dict):
        st.error("实验目录缺少有效的 hierarchy_meta.json。")
        return
    level_names = [
        str(name)
        for name in (meta.get("head_names") or meta.get("level_names") or [])
    ]
    if not level_names:
        st.error("实验元数据中没有可用预测层级。")
        return
    level_name = st.selectbox("选择预测层级", level_names, key="inference_level")
    target_labels = _target_class_names(meta, level_name)
    st.caption(f"实验目录：{experiment_dir}")
    metrics = st.columns(3)
    metrics[0].metric("Profile", profile_id)
    metrics[1].metric("预测层级", level_name)
    metrics[2].metric("模型类别数", len(target_labels))

    folders = _list_cs_folders()
    if not folders:
        st.info("dataset/CSdata 下没有可推理的 CS 文件夹。")
        _render_existing_result(experiment_dir, level_name)
        return
    expected_lookup = build_expected_label_lookup(meta, level_name)
    folder_info = _build_folder_info(folders, expected_lookup, set(target_labels))
    category_options = sorted(
        {str(item["category"]) for item in folder_info},
        key=build_natural_key,
    )
    default_categories = sorted(
        {str(item["category"]) for item in folder_info if item["known"]},
        key=build_natural_key,
    )
    selected_categories = st.multiselect(
        "按 CS 类别筛选",
        category_options,
        default=default_categories,
        key="inference_categories",
    )
    category_set = set(selected_categories)
    category_folders = [
        str(item["folder"])
        for item in folder_info
        if item["category"] in category_set
    ]
    selected_folders = st.multiselect(
        "选择推理文件夹",
        category_folders,
        default=category_folders,
        key="inference_folders",
    )
    if not selected_folders:
        st.warning("请至少选择一个 CS 文件夹。")
        _render_existing_result(experiment_dir, level_name)
        return

    profile = next(profile for profile in profiles if profile.profile_id == profile_id)
    cache_root = get_dataset_dir(profile, PROJECT_ROOT) / profile.root_test
    st.caption(f"测试数据缓存：{cache_root}")
    actions = st.columns(2)
    if actions[0].button("构建/更新测试数据", use_container_width=True):
        _build_cache(experiment_dir, level_name, selected_folders, cache_root)
    if actions[1].button("开始样本推理", type="primary", use_container_width=True):
        _run_inference(experiment_dir, level_name, selected_folders, cache_root)
    _render_existing_result(experiment_dir, level_name)


def _list_experiment_dirs(profile_id: str) -> list[Path]:
    return [
        Path(path)
        for path in cached_experiment_dirs(
            str(PROJECT_ROOT.resolve()), profile_id, ui_cache_version("inference")
        )
    ]


def _list_cs_folders() -> list[str]:
    return list(cached_cs_folder_names(str(CS_ROOT.resolve()), ui_cache_version("inference")))


@st.cache_resource(show_spinner=False)
def _load_cached_predictor(experiment_dir: str, level_name: str, meta_modified_ns: int):
    """在当前会话复用同一实验层级的已加载模型。"""
    del meta_modified_ns
    return load_predictor(experiment_dir, resolve_device("cpu"), level_name)


def _get_cached_predictor(experiment_dir: Path, level_name: str):
    meta_path = experiment_dir / "hierarchy_meta.json"
    return _load_cached_predictor(str(experiment_dir.resolve()), level_name, meta_path.stat().st_mtime_ns)


def _target_class_names(meta: dict[str, Any], level_name: str) -> list[str]:
    names = (meta.get("class_names_by_level") or {}).get(level_name, [])
    return [str(name) for name in names]


def _build_folder_info(
    folders: list[str],
    expected_lookup: dict[str, str],
    target_labels: set[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for folder in folders:
        expected = expected_lookup.get(parse_test_folder_prefix(folder))
        known = expected is not None and expected in target_labels
        result.append(
            {
                "folder": folder,
                "category": expected if known else "未识别类别",
                "expected": expected or "",
                "known": known,
            }
        )
    return result


def _build_cache(
    experiment_dir: Path,
    level_name: str,
    folder_names: list[str],
    cache_root: Path,
) -> None:
    try:
        with st.status("正在更新测试缓存…", expanded=True) as status:
            status.write("正在复用或加载实验模型。")
            predictor = _get_cached_predictor(experiment_dir, level_name)
            status.write("正在检查新增和已修改的 CS 光谱。")
            report = build_test_cache(CS_ROOT, cache_root, folder_names, predictor)
            status.update(label="测试缓存更新完成", state="complete", expanded=False)
        invalidate_ui_cache("inference")
        st.success(
            f"测试数据缓存完成：共 {report['file_count']} 条，"
            f"复用 {report['reused_count']} 条，"
            f"重新处理 {report['rebuilt_count']} 条，"
            f"失败 {report['failed_count']} 条。"
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        st.error(f"测试数据缓存失败：{error}")


def _run_inference(
    experiment_dir: Path,
    level_name: str,
    folder_names: list[str],
    cache_root: Path,
) -> None:
    try:
        with st.status("正在执行样本推理…", expanded=True) as status:
            status.write("正在加载或复用实验模型。")
            predictor = _get_cached_predictor(experiment_dir, level_name)
            status.write("正在检查测试数据缓存。")
            build_report = build_test_cache(CS_ROOT, cache_root, folder_names, predictor)
            status.write("正在执行预测并生成结果报告。")
            result_dir = run_independent_inference(
                experiment_dir,
                level_name,
                predictor=predictor,
                cached_test_root=cache_root,
                folder_names=folder_names,
                evaluate_enable=True,
            )
            status.update(label="样本推理完成", state="complete", expanded=False)
        invalidate_ui_cache("inference")
        st.success(
            f"推理完成：缓存共 {build_report['file_count']} 条，"
            f"复用 {build_report['reused_count']} 条，"
            f"重新处理 {build_report['rebuilt_count']} 条，"
            f"结果已保存到 {result_dir}。"
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        st.error(f"样本推理失败：{error}")


def _render_existing_result(experiment_dir: Path, level_name: str) -> None:
    result_dir = experiment_dir / level_name / "test_result"
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        return
    try:
        summary_stat = summary_path.stat()
        summary = cached_json_file(str(summary_path.resolve()), summary_stat.st_mtime_ns, summary_stat.st_size)
    except OSError:
        summary = None
    if not isinstance(summary, dict):
        st.warning("已有推理结果无法读取。")
        return
    st.subheader("最近一次推理结果")
    metrics = st.columns(5)
    metrics[0].metric("推理文件夹", int(summary.get("folder_count", 0)))
    metrics[1].metric("已评估文件夹", int(summary.get("evaluated_folder_count", 0)))
    metrics[2].metric("文件夹准确率", _format_ratio(summary.get("folder_accuracy")))
    metrics[3].metric("光谱准确率", _format_ratio(summary.get("spectrum_accuracy")))
    metrics[4].metric("未识别类别", int(summary.get("unevaluated_folder_count", 0)))
    rows = summary.get("rows", [])
    if isinstance(rows, list) and rows:
        display_rows = [
            {
                "文件夹": row.get("folder", ""),
                "真实类别": row.get("expected_label", "未评估"),
                "预测类别": row.get("predicted_label", ""),
                "光谱正确率": _format_ratio(row.get("correct_ratio")),
                "文件夹正确": "是" if row.get("folder_correct") else "否",
            }
            for row in rows
            if isinstance(row, dict)
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        for row in rows:
            if not isinstance(row, dict):
                continue
            folder_name = str(row.get("folder", ""))
            if not folder_name:
                continue
            with st.expander(f"查看 {folder_name} 的逐谱结果", expanded=False):
                image_path = result_dir / folder_name / "spectra.png"
                if image_path.is_file():
                    st.image(str(image_path), use_container_width=True)
                file_predictions = row.get("file_predictions")
                if isinstance(file_predictions, list) and file_predictions:
                    st.dataframe(
                        [
                            {
                                "文件": item.get("file", ""),
                                "Top-1": item.get("top1_label", ""),
                                "Top-k": ", ".join(
                                    str(prediction.get("label", ""))
                                    for prediction in item.get("predictions", [])
                                    if isinstance(prediction, dict)
                                ),
                            }
                            for item in file_predictions
                            if isinstance(item, dict)
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
    with st.expander("查看结果目录", expanded=False):
        st.write(str(result_dir))
        summary_text = result_dir / "summary.txt"
        if summary_text.is_file():
            st.code(summary_text.read_text(encoding="utf-8"), language="text")


def _format_ratio(value: object) -> str:
    if value is None:
        return "未评估"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "未评估"


__all__ = ["render_sample_inference_page"]
