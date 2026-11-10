"""样本推理、测试文件夹筛选和测试输入缓存页面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ramanv2.common.naming import build_natural_key
from ramanv2.common.device import resolve_device
from ramanv2.core.run_context import INFERENCE_RESULT_DIR_NAME
from ramanv2.inference.input_cache import build_test_cache
from ramanv2.modeling.predictor import load_predictor
from ramanv2.inference.folder_inference import (
    SPECTRA_FIGURE_NAME,
    SUMMARY_JSON_NAME,
    SUMMARY_TEXT_NAME,
    run_independent_inference,
)
from ramanv2.ui.read_cache import (
    cached_test_folder_names,
    cached_experiment_dirs,
    cached_json_file,
    cached_text_file,
)
from ramanv2.ui.components.pickers import render_profile_picker
from ramanv2.ui.formatting import format_ratio, relative_label
from ramanv2.ui.page_state import (
    INVALIDATE_INFERENCE,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    render_action_error,
    ui_cache_version,
)


def render_sample_inference_page(context) -> None:
    """渲染实验选择、测试缓存、样本推理和结果查看区域。"""
    st.title("样本推理")
    profile = render_profile_picker(context, label="选择 Profile", key="inference_profile")
    if profile is None:
        return
    experiment_dirs = _list_experiment_dirs(context, profile)
    if not experiment_dirs:
        st.info(f"没有找到 {profile.dataset_name} 的有效实验目录。")
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
        st.error("实验元数据中没有可用预测层级。")
        return
    level_name = st.selectbox("选择预测层级", level_names, key="inference_level")
    target_labels = _target_class_names(meta, level_name)
    st.caption(f"实验目录：{experiment_dir}")
    metrics = st.columns(3)
    metrics[0].metric("数据集", context.dataset_id)
    metrics[1].metric("预测层级", level_name)
    metrics[2].metric("模型类别数", len(target_labels))

    folders = list(
        cached_test_folder_names(
            str(context.init_test_root.resolve()),
            ui_cache_version("inference"),
        )
    )
    if not folders:
        st.info(f"{context.init_test_root} 下没有可推理的测试文件夹。")
        _render_existing_result(experiment_dir, level_name)
        return
    folder_info = _build_folder_info(folders, set(target_labels), context.hierarchy)
    category_options = sorted(
        {str(item["category"]) for item in folder_info},
        key=build_natural_key,
    )
    default_categories = sorted(
        {str(item["category"]) for item in folder_info if item["known"]},
        key=build_natural_key,
    )
    selected_categories = st.multiselect(
        "按测试类别筛选",
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
        st.warning("请至少选择一个测试文件夹。")
        _render_existing_result(experiment_dir, level_name)
        return

    st.caption("样本推理会从每个文件夹随机抽取最多 10 条光谱，同一 cell 只取 1 条。")

    cache_root = context.profile_dir(profile) / profile.root_test
    st.caption(f"测试数据缓存：{cache_root}")
    actions = st.columns(2)
    if actions[0].button("构建/更新测试数据", width="stretch"):
        _prepare_test_inputs(
            experiment_dir,
            level_name,
            selected_folders,
            cache_root,
            context.init_test_root,
            infer_enable=False,
        )
    if actions[1].button("开始样本推理", type="primary", width="stretch"):
        _prepare_test_inputs(
            experiment_dir,
            level_name,
            selected_folders,
            cache_root,
            context.init_test_root,
            infer_enable=True,
        )
    _render_existing_result(experiment_dir, level_name)


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
    """返回可训练的业务层级名称。"""
    return [str(name) for name in (meta.get("head_names") or [])]


def _target_class_names(meta: dict[str, Any], level_name: str) -> list[str]:
    names = (meta.get("class_names_by_level") or {}).get(level_name, [])
    return [str(name) for name in names]


def _build_folder_info(
    folders: list[str],
    target_labels: set[str],
    hierarchy,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for folder in folders:
        try:
            expected = hierarchy.expected_label(folder, "level_1")
        except ValueError:
            expected = None
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


def _prepare_test_inputs(
    experiment_dir: Path,
    level_name: str,
    folder_names: list[str],
    cache_root: Path,
    test_root: Path,
    *,
    infer_enable: bool,
) -> None:
    """构建/更新测试缓存；infer_enable 时继续执行样本推理。"""
    label = "样本推理" if infer_enable else "测试数据缓存"
    try:
        with st.status(
            "正在执行样本推理…" if infer_enable else "正在更新测试缓存…",
            expanded=True,
        ) as status:
            status.write("正在复用或加载实验模型。")
            predictor = _get_cached_predictor(experiment_dir, level_name)
            status.write("正在检查新增和已修改的测试光谱。")
            report = build_test_cache(test_root, cache_root, folder_names, predictor)
            result_dir = None
            if infer_enable:
                status.write("正在从每个文件夹随机抽取光谱（每个 cell 至多 1 条）。")
                result_dir = run_independent_inference(
                    experiment_dir,
                    level_name,
                    predictor=predictor,
                    cached_test_root=cache_root,
                    folder_names=folder_names,
                    evaluate_enable=True,
                )
            status.update(
                label="样本推理完成" if infer_enable else "测试缓存更新完成",
                state="complete",
                expanded=False,
            )
        invalidate_ui_cache(*INVALIDATE_INFERENCE)
        tail = (
            f"结果已保存到 {result_dir}。"
            if infer_enable
            else f"失败 {report['failed_count']} 条。"
        )
        st.success(
            f"{'推理完成：缓存共' if infer_enable else '测试数据缓存完成：共'} "
            f"{report['file_count']} 条，复用 {report['reused_count']} 条，"
            f"重新处理 {report['rebuilt_count']} 条，{tail}"
        )
    except UI_ACTION_ERRORS as error:
        render_action_error(label, error)


def _render_existing_result(experiment_dir: Path, level_name: str) -> None:
    result_dir = experiment_dir / level_name / INFERENCE_RESULT_DIR_NAME
    summary_path = result_dir / SUMMARY_JSON_NAME
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
    metrics[2].metric("文件夹准确率", format_ratio(summary.get("folder_accuracy"), "未评估"))
    metrics[3].metric("光谱准确率", format_ratio(summary.get("spectrum_accuracy"), "未评估"))
    metrics[4].metric("未识别类别", int(summary.get("unevaluated_folder_count", 0)))
    rows = summary.get("rows", [])
    if isinstance(rows, list) and rows:
        display_rows = [
            {
                "文件夹": row.get("folder", ""),
                "真实类别": row.get("expected_label", "未评估"),
                "预测类别": row.get("predicted_label", ""),
                "高置信输出": "是" if row.get("high_confidence") else "否",
                "光谱正确率": format_ratio(row.get("correct_ratio"), "未评估"),
                "文件夹正确": "是" if row.get("folder_correct") else "否",
            }
            for row in rows
            if isinstance(row, dict)
        ]
        st.dataframe(display_rows, width="stretch", hide_index=True)
        for row in rows:
            if not isinstance(row, dict):
                continue
            folder_name = str(row.get("folder", ""))
            if not folder_name:
                continue
            with st.expander(f"查看 {folder_name} 的逐谱结果", expanded=False):
                image_path = result_dir / folder_name / SPECTRA_FIGURE_NAME
                if image_path.is_file():
                    st.image(str(image_path), width="stretch")
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
                        width="stretch",
                        hide_index=True,
                    )
    with st.expander("查看结果目录", expanded=False):
        st.write(str(result_dir))
        summary_text = result_dir / SUMMARY_TEXT_NAME
        if summary_text.is_file():
            stat = summary_text.stat()
            st.code(cached_text_file(str(summary_text.resolve()), stat.st_mtime_ns, stat.st_size), language="text")
