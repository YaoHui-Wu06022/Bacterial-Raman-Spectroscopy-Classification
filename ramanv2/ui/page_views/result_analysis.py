"""单个模型槽位的评估、Analysis 与 UMAP 页面。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from ramanv2.analysis.interpret import interpret_level, interpret_slot, umap_slot
from ramanv2.common.hierarchy import load_hierarchy_meta
from ramanv2.common.naming import build_natural_key
from ramanv2.core.run_context import (
    ANALYSIS_LOG_FILE_NAME,
    ANALYSIS_RESULT_DIR_NAME,
    CASCADE_MODE_DIR_NAME,
    CONFUSION_MATRIX_FILE_NAME,
    HIERARCHY_META_FILE_NAME,
    METRICS_FILE_NAME,
    PARENT_KNOWN_MODE_DIR_NAME,
    UMAP_FIGURE_NAME,
    VAL_RESULT_DIR_NAME,
)
from ramanv2.evaluation.model_eval import evaluate_model_run
from ramanv2.evaluation.level_summary import (
    LEVEL_SUMMARY_DIR_NAME,
    summarize_level_results,
)
from ramanv2.modeling.run_manifest import has_model_path
from ramanv2.ui.read_cache import (
    LEVEL_DIR_PATTERN,
    cached_experiment_dirs,
    cached_json_file,
    cached_run_dirs,
    cached_text_file,
)
from ramanv2.ui.formatting import format_ratio, relative_label
from ramanv2.ui.page_state import (
    INVALIDATE_ANALYSIS,
    UI_ACTION_ERRORS,
    invalidate_ui_cache,
    render_action_error,
    ui_cache_version,
)


@dataclass(frozen=True)
class ResultSection:
    """描述一个结果区块的差异；按钮、读取门控与错误处理共用一套。"""

    key: str
    title: str
    read_label: str
    compute_label: str
    compute: Callable[[Path, str], None]
    result_path: Callable[[Path], Path]
    exists: Callable[[Path], bool]
    missing_message: str
    pending_message: str
    error_label: str
    success_message: str
    render: Callable[[Path, Path], None]

    @property
    def read_key(self) -> str:
        return f"result_analysis_{self.key}_read"


def render_result_analysis_page(context) -> None:
    """按实验目录、层级和 run 逐级选择并展示结果。"""
    st.title("结果分析")
    experiments = _list_experiment_dirs(context)
    if not experiments:
        st.info("当前数据集尚未发现包含完整切分和层级元数据的实验目录。")
        return
    experiment_labels = [relative_label(path, context.project_root) for path in experiments]
    experiment_label = st.selectbox(
        "实验目录", experiment_labels,
        index=_selected_index("result_analysis_experiment", experiment_labels),
        key="result_analysis_experiment_select",
    )
    experiment_dir = experiments[experiment_labels.index(experiment_label)]
    st.session_state["result_analysis_experiment"] = experiment_label
    levels = _find_level_dirs(experiment_dir)
    if not levels:
        st.warning("实验目录中没有可用的 level_X。")
        return
    level_names = [path.name for path in levels]
    level_name = st.selectbox(
        "层级", level_names,
        index=_selected_index("result_analysis_level", level_names),
        key="result_analysis_level_select",
    )
    st.session_state["result_analysis_level"] = level_name
    run_dirs = _find_run_dirs(experiment_dir / level_name)
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    global_entry = (meta.get("level_models") or {}).get(level_name) if meta else None
    if run_dirs and not has_model_path(global_entry):
        _render_level_summary(experiment_dir, level_name)
    if not run_dirs:
        st.warning("当前层级没有可用的模型。")
        return
    run_labels = [str(path.relative_to(experiment_dir)) for path in run_dirs]
    run_label = st.selectbox(
        "模型", run_labels,
        index=_selected_index("result_analysis_run", run_labels),
        key="result_analysis_run_select",
    )
    st.session_state["result_analysis_run"] = run_label
    run_dir = run_dirs[run_labels.index(run_label)]
    st.caption(f"当前模型：{run_dir}")
    for section in _RESULT_SECTIONS:
        _render_result_section(run_dir, level_name, section)


def _list_experiment_dirs(context) -> list[Path]:
    """汇总当前数据集所有命名空间的完整实验目录。"""
    paths: list[str] = []
    for root in context.experiment_roots():
        paths.extend(
            cached_experiment_dirs(
                str(root.resolve()),
                True,
                ui_cache_version("analysis"),
            )
        )
    return [Path(path) for path in sorted(set(paths), key=build_natural_key, reverse=True)]


def _find_level_dirs(experiment_dir: Path | str) -> list[Path]:
    """列出实验根下的业务层级目录。"""
    root = Path(experiment_dir)
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and LEVEL_DIR_PATTERN.match(path.name)),
        key=lambda path: build_natural_key(path.name),
    )


def _find_run_dirs(level_dir: Path | str) -> list[Path]:
    """发现全局模型和父类子模型的槽位目录。"""
    root = Path(level_dir)
    if not root.is_dir():
        return []
    return [Path(path) for path in cached_run_dirs(str(root.resolve()), ui_cache_version("analysis"))]


def _render_level_summary(experiment_dir: Path, level_name: str) -> None:
    """渲染层级汇总的生成与查看区域。"""
    st.subheader("层级汇总")
    summary_root = Path(experiment_dir) / level_name / LEVEL_SUMMARY_DIR_NAME
    if st.button(
        "重算层级汇总（两种模式）",
        type="primary",
        width="stretch",
        key=f"level_summary_compute_{level_name}",
    ):
        try:
            with st.spinner("正在拼接两种模式的验证与测试结果…"):
                summarize_level_results(experiment_dir, level_name)
            invalidate_ui_cache(*INVALIDATE_ANALYSIS)
            st.success("层级汇总已更新。")
        except UI_ACTION_ERRORS as error:
            render_action_error("层级汇总", error)
    mode_labels = {
        PARENT_KNOWN_MODE_DIR_NAME: "已知上层（parent_known）",
        CASCADE_MODE_DIR_NAME: "从头级联（cascade）",
    }
    mode = st.selectbox(
        "评估模式",
        [PARENT_KNOWN_MODE_DIR_NAME, CASCADE_MODE_DIR_NAME],
        format_func=lambda value: mode_labels[value],
        key=f"level_summary_mode_{level_name}",
    )
    mode_root = summary_root / mode
    tabs = st.tabs(["验证结果", "测试结果", "Analysis 汇总"])
    for tab, output_name in zip(tabs[:2], ("val_result", "test_result")):
        with tab:
            result_dir = mode_root / output_name
            if not (result_dir / METRICS_FILE_NAME).is_file():
                st.info("尚未生成该模式的层级汇总。")
                continue
            _render_confusion(Path(experiment_dir) / level_name, result_dir)
    with tabs[2]:
        if st.button(
            "重算层级归因",
            type="primary",
            width="stretch",
            key=f"level_analysis_compute_{level_name}",
        ):
            try:
                with st.spinner("正在聚合层级归因…"):
                    interpret_level(experiment_dir, level_name)
                invalidate_ui_cache(*INVALIDATE_ANALYSIS)
                st.success("层级归因已更新。")
            except UI_ACTION_ERRORS as error:
                render_action_error("层级归因", error)
        analysis_dir = summary_root / ANALYSIS_RESULT_DIR_NAME
        if (analysis_dir / "figures").is_dir():
            _render_analysis(Path(experiment_dir) / level_name, analysis_dir)
        else:
            st.info("尚未生成层级归因汇总。")


def _render_result_section(run_dir: Path, level_name: str, section: ResultSection) -> None:
    """渲染一个结果区块：读取/重算按钮、结果门控与具体展示。"""
    st.subheader(section.title)
    actions = st.columns(2)
    if actions[0].button(section.read_label, width="stretch"):
        st.session_state[section.read_key] = str(run_dir)
    if actions[1].button(section.compute_label, type="primary", width="stretch"):
        try:
            section.compute(run_dir, level_name)
            invalidate_ui_cache(*INVALIDATE_ANALYSIS)
            st.session_state[section.read_key] = str(run_dir)
            st.success(section.success_message)
        except UI_ACTION_ERRORS as error:
            render_action_error(section.error_label, error)
    result_path = section.result_path(run_dir)
    if not section.exists(result_path):
        st.info(section.missing_message)
        return
    if st.session_state.get(section.read_key) != str(run_dir):
        st.caption(section.pending_message)
        return
    section.render(run_dir, result_path)


def _render_confusion(run_dir: Path, result_dir: Path) -> None:
    image_path = result_dir / CONFUSION_MATRIX_FILE_NAME
    if image_path.is_file():
        st.image(str(image_path), caption="Confusion matrix", width="stretch")
    metrics_path = result_dir / METRICS_FILE_NAME
    if not metrics_path.is_file():
        return
    stat = metrics_path.stat()
    payload = cached_json_file(str(metrics_path.resolve()), stat.st_mtime_ns, stat.st_size)
    if not isinstance(payload, dict):
        return
    summary = payload.get("summary", {})
    columns = st.columns(3)
    columns[0].metric("Accuracy", format_ratio(summary.get("accuracy"), "—"))
    columns[1].metric("Macro F1", format_ratio(summary.get("macro_f1"), "—"))
    columns[2].metric("Macro Recall", format_ratio(summary.get("macro_recall"), "—"))
    with st.expander("分类报告", expanded=False):
        st.dataframe(payload.get("classes", []), width="stretch")


def _render_analysis(run_dir: Path, analysis_dir: Path) -> None:
    _render_figures(analysis_dir / "figures", "Analysis 图表")
    log_path = analysis_dir / "logs" / ANALYSIS_LOG_FILE_NAME
    if not log_path.is_file():
        return
    stat = log_path.stat()
    with st.expander("Analysis 日志", expanded=False):
        st.code(cached_text_file(str(log_path.resolve()), stat.st_mtime_ns, stat.st_size), language="text")


def _render_umap(run_dir: Path, image_path: Path) -> None:
    st.image(str(image_path), caption="Train / Val UMAP", width="stretch")


def _render_figures(figure_dir: Path, caption: str) -> None:
    figures = sorted(path for path in figure_dir.glob("*.png") if path.name != UMAP_FIGURE_NAME) if figure_dir.is_dir() else []
    if not figures:
        st.info(f"尚无{caption}。")
        return
    names = [path.name for path in figures]
    selected = st.selectbox(caption, names, key=f"result_analysis_figure_{caption}")
    st.image(str(figures[names.index(selected)]), width="stretch")


def _selected_index(key: str, options: list[str]) -> int:
    value = st.session_state.get(key)
    return options.index(value) if value in options else 0


_RESULT_SECTIONS = (
    ResultSection(
        key="confusion",
        title="混淆矩阵",
        read_label="读取已有混淆矩阵",
        compute_label="重新计算混淆矩阵",
        compute=evaluate_model_run,
        result_path=lambda run_dir: run_dir / VAL_RESULT_DIR_NAME,
        exists=Path.is_dir,
        missing_message="尚无评估结果，请点击重新计算。",
        pending_message="已有评估结果；点击读取已有混淆矩阵后查看。",
        error_label="混淆矩阵计算",
        success_message="混淆矩阵已重新计算。",
        render=_render_confusion,
    ),
    ResultSection(
        key="analysis",
        title="Analysis",
        read_label="读取已有 Analysis",
        compute_label="重新计算 Analysis",
        compute=lambda run_dir, level_name: interpret_slot(run_dir, level_name, umap_enable=False),
        result_path=lambda run_dir: run_dir / ANALYSIS_RESULT_DIR_NAME,
        exists=Path.is_dir,
        missing_message="尚无 Analysis 结果，请点击重新计算。",
        pending_message="已有 Analysis 结果；点击读取已有 Analysis 后查看。",
        error_label="Analysis 计算",
        success_message="Analysis 已重新计算，未重新生成 UMAP。",
        render=_render_analysis,
    ),
    ResultSection(
        key="umap",
        title="UMAP",
        read_label="读取已有 UMAP",
        compute_label="重新计算 UMAP",
        compute=umap_slot,
        result_path=lambda run_dir: run_dir / ANALYSIS_RESULT_DIR_NAME / "figures" / UMAP_FIGURE_NAME,
        exists=Path.is_file,
        missing_message="尚无 UMAP 图，请点击重新计算。",
        pending_message="已有 UMAP 图；点击读取已有 UMAP 后查看。",
        error_label="UMAP 计算",
        success_message="UMAP 已重新计算。",
        render=_render_umap,
    ),
)
