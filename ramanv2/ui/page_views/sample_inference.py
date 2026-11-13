"""无标签实际样本推理页面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ramanv2.common.device import resolve_device
from ramanv2.inference.unlabeled_inference import (
    UNLABELED_RESULT_DIR_NAME,
    INFERENCE_DATA_DIR_NAME,
    list_inference_dates,
    run_unlabeled_inference,
)
from ramanv2.modeling.predictor import load_predictor
from ramanv2.modeling.run_manifest import has_model_path
from ramanv2.ui.formatting import relative_label
from ramanv2.ui.page_state import (
    INVALIDATE_INFERENCE,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    render_action_error,
    ui_cache_version,
)
from ramanv2.ui.read_cache import (
    cached_experiment_dirs,
    cached_json_file,
    cached_text_file,
)


def render_sample_inference_page(context) -> None:
    """选择实验、层级和 inference_data 日期，执行无标签实际推理。"""
    st.title("样本推理")
    st.caption("实际推理：使用 inference_data 原始光谱；不依赖标签，不参与测试评估。")
    profile = context.profile()
    experiment_dirs = _list_experiment_dirs(context, profile)
    if not experiment_dirs:
        st.info(f"没有找到 {context.dataset_spec.display_name} 的有效实验目录。")
        return
    experiment_labels = [relative_label(path, context.project_root) for path in experiment_dirs]
    selected_label = st.selectbox("选择实验目录", experiment_labels, key="inference_experiment")
    experiment_dir = experiment_dirs[experiment_labels.index(selected_label)]
    meta_path = experiment_dir / "hierarchy_meta.json"
    meta_stat = meta_path.stat()
    meta = cached_json_file(str(meta_path.resolve()), meta_stat.st_mtime_ns, meta_stat.st_size)
    if not isinstance(meta, dict):
        st.error("实验目录缺少有效的 hierarchy_meta.json。")
        return
    level_names = _predict_level_names(meta)
    if not level_names:
        st.error("实验元数据中没有训练完成的预测层级。")
        return
    level_display = _level_display_names(meta)
    level_name = st.selectbox(
        "选择预测层级",
        level_names,
        format_func=lambda name: level_display.get(name, name),
        key=f"inference_level_{experiment_dir.name}",
    )
    target_labels = _target_class_names(meta, level_name)

    inference_root = context.dataset_spec.dataset_dir / INFERENCE_DATA_DIR_NAME
    date_dirs = list_inference_dates(inference_root)
    st.caption(f"实验目录：{experiment_dir}")
    metrics = st.columns(3)
    metrics[0].metric("数据集", context.dataset_id)
    metrics[1].metric("预测层级", level_display.get(level_name, level_name))
    metrics[2].metric("模型类别数", len(target_labels))
    if not date_dirs:
        st.info(f"{inference_root} 下没有推理日期目录。")
        return
    date_labels = [relative_label(path, context.project_root) for path in date_dirs]
    date_label = st.selectbox("选择推理日期", date_labels, key="inference_date")
    date_dir = date_dirs[date_labels.index(date_label)]
    st.caption(f"输入目录：{date_dir}")

    if st.button("开始样本推理", type="primary", key="inference_run"):
        try:
            with st.status("正在执行实际推理…", expanded=True) as status:
                status.write("正在加载模型。")
                predictor = _get_cached_predictor(experiment_dir, level_name)
                status.write("正在拟合小球仿射并推理全部光谱。")
                report = run_unlabeled_inference(
                    experiment_dir,
                    level_name,
                    date_dir,
                    predictor=predictor,
                )
                status.update(label="实际推理完成", state="complete", expanded=False)
            invalidate_ui_cache(*INVALIDATE_INFERENCE)
            st.success(
                f"完成：{report['folder_count']} 个文件夹，"
                f"成功 {report['succeeded_folder_count']} 个，"
                f"失败 {report['failed_folder_count']} 个；"
                f"结果已保存到 {report['result_dir']}。"
            )
        except UI_ACTION_ERRORS as error:
            render_action_error("样本推理", error)
    _render_unlabeled_result(date_dir, level_display)


def _list_experiment_dirs(context, profile) -> list[Path]:
    return [
        Path(path)
        for path in cached_experiment_dirs(
            str(context.experiment_root(profile).resolve()),
            False,
            ui_cache_version("inference"),
        )
    ]


@st.cache_resource(show_spinner=False)
def _load_cached_predictor(experiment_dir: str, level_name: str, meta_modified_ns: int):
    """复用通过统一种标签版本校验的实验模型。"""
    del meta_modified_ns
    return load_predictor(experiment_dir, resolve_device("cpu"), level_name)


def _get_cached_predictor(experiment_dir: Path, level_name: str):
    meta_path = experiment_dir / "hierarchy_meta.json"
    return _load_cached_predictor(str(experiment_dir.resolve()), level_name, meta_path.stat().st_mtime_ns)


def _predict_level_names(meta: dict[str, Any]) -> list[str]:
    """返回训练完成的业务层级名称。"""
    names = [str(name) for name in (meta.get("head_names") or [])]
    return [name for name in names if _level_has_model(meta, name)]


def _level_has_model(meta: dict[str, Any], level_name: str) -> bool:
    """层级有全局模型或任一父类子模型才算可推理。"""
    global_entry = (meta.get("level_models") or {}).get(level_name)
    if has_model_path(global_entry):
        return True
    parent_entries = (meta.get("parent_models") or {}).get(level_name) or {}
    return any(has_model_path(entry) for entry in parent_entries.values())


def _level_display_names(meta: dict[str, Any]) -> dict[str, str]:
    """返回层级到显示名的映射，例如 level_2 -> 属。"""
    mapping: dict[str, str] = {}
    for item in (meta.get("hierarchy") or {}).get("levels") or []:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            mapping[name] = str(item.get("display_name") or name)
    return mapping


def _target_class_names(meta: dict[str, Any], level_name: str) -> list[str]:
    names = (meta.get("class_names_by_level") or {}).get(level_name, [])
    return [str(name) for name in names]


def _render_unlabeled_result(date_dir: Path, level_display: dict[str, str]) -> None:
    """展示一个日期已有的无标签推理结果。"""
    result_dir = date_dir.parent / UNLABELED_RESULT_DIR_NAME / date_dir.name
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        return
    try:
        stat = summary_path.stat()
        payload = cached_json_file(str(summary_path.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        payload = None
    if not isinstance(payload, dict):
        st.warning("已有推理结果无法读取。")
        return
    st.subheader("最近一次实际推理结果")
    metrics = st.columns(5)
    metrics[0].metric("日期", str(payload.get("date", date_dir.name)))
    metrics[1].metric("文件夹", int(payload.get("folder_count", 0)))
    metrics[2].metric("成功", int(payload.get("succeeded_folder_count", 0)))
    metrics[3].metric("失败", int(payload.get("failed_folder_count", 0)))
    metrics[4].metric("仿射状态", str((payload.get("affine") or {}).get("status", "")))
    target_level = str(payload.get("target_level", ""))
    st.caption(
        f"实验：{payload.get('experiment_dir', '')}  "
        f"层级：{level_display.get(target_level, target_level)}  "
        f"模型类别数：{payload.get('model_class_count', '')}"
    )
    folders = payload.get("folders") or []
    if not isinstance(folders, list) or not folders:
        st.info("该日期没有可展示的样本文件夹结果。")
        return
    st.dataframe(
        [
            {
                "文件夹": folder.get("folder", ""),
                "状态": folder.get("status", ""),
                "预测": folder.get("predicted_label", ""),
                "票数": f"{int(folder.get('majority_count', 0))}/{int(folder.get('total_count', 0))}",
                "最高票占比": f"{float(folder.get('majority_ratio', 0.0)) * 100:.1f}%",
                "平均置信度": f"{float(folder.get('mean_top1_probability', 0.0)) * 100:.1f}%",
                "失败光谱": int(folder.get("failed_count", 0)),
            }
            for folder in folders
            if isinstance(folder, dict)
        ],
        width="stretch",
        hide_index=True,
    )
    for folder in folders:
        if not isinstance(folder, dict):
            continue
        folder_name = str(folder.get("folder", ""))
        if not folder_name:
            continue
        with st.expander(
            f"查看 {folder_name} 的逐谱结果（{folder.get('predicted_label', '')}）",
            expanded=False,
        ):
            predictions_path = result_dir / folder_name / "predictions.csv"
            if predictions_path.is_file():
                st.dataframe(pd.read_csv(predictions_path), width="stretch", hide_index=True)
            summary_text_path = result_dir / folder_name / "summary.txt"
            if summary_text_path.is_file():
                stat = summary_text_path.stat()
                st.code(
                    cached_text_file(str(summary_text_path.resolve()), stat.st_mtime_ns, stat.st_size),
                    language="text",
                )
    with st.expander("查看结果目录", expanded=False):
        st.write(str(result_dir))
        summary_text_path = result_dir / "summary.txt"
        if summary_text_path.is_file():
            stat = summary_text_path.stat()
            st.code(
                cached_text_file(str(summary_text_path.resolve()), stat.st_mtime_ns, stat.st_size),
                language="text",
            )
