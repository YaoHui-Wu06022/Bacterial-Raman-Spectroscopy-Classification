"""单个训练 run 的评估、Analysis 与 UMAP 页面。"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from ramanv2.analysis.runner import run_interpret_run, run_umap_run
from ramanv2.core.paths import PROJECT_ROOT
from ramanv2.evaluation.model_eval import evaluate_model_run
from ramanv2.ui.cache import cached_json_file, cached_result_experiment_dirs, cached_run_dirs
from ramanv2.ui.state import invalidate_ui_cache, ui_cache_version


_LEVEL_PATTERN = re.compile(r"^level_\d+$")


def render_result_analysis_page() -> None:
    """按实验目录、层级和 run 逐级选择并展示结果。"""
    st.title("结果分析")
    experiments = [
        Path(path)
        for path in cached_result_experiment_dirs(
            str((PROJECT_ROOT / "output").resolve()), ui_cache_version("analysis")
        )
    ]
    if not experiments:
        st.info("尚未发现包含完整切分和层级元数据的实验目录。")
        return
    experiment_labels = [str(path.relative_to(PROJECT_ROOT)) for path in experiments]
    experiment_label = st.selectbox(
        "实验目录", experiment_labels,
        index=_selected_index("result_analysis_experiment", experiment_labels),
        key="result_analysis_experiment_select",
    )
    experiment_dir = experiments[experiment_labels.index(experiment_label)]
    st.session_state["result_analysis_experiment"] = experiment_label
    levels = find_level_dirs(experiment_dir)
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
    run_dirs = find_run_dirs(experiment_dir / level_name)
    if not run_dirs:
        st.warning("当前层级没有可用的 run。")
        return
    run_labels = [str(path.relative_to(experiment_dir)) for path in run_dirs]
    run_label = st.selectbox(
        "run", run_labels,
        index=_selected_index("result_analysis_run", run_labels),
        key="result_analysis_run_select",
    )
    st.session_state["result_analysis_run"] = run_label
    run_dir = run_dirs[run_labels.index(run_label)]
    st.caption(f"当前 run：{run_dir}")
    _render_confusion_section(run_dir, level_name)
    _render_analysis_section(run_dir, level_name)
    _render_umap_section(run_dir, level_name)


def find_experiment_dirs(output_root: Path | str) -> list[Path]:
    """发现包含页面所需四个实验级快照的实验目录。"""
    root = Path(output_root)
    if not root.is_dir():
        return []
    markers = ("shared_config.yaml", "hierarchy_meta.json", "train_split.json", "val_split.json")
    return sorted(
        (
            path
            for profile_dir in root.iterdir() if profile_dir.is_dir()
            for path in profile_dir.iterdir()
            if path.is_dir() and all((path / marker).is_file() for marker in markers)
        ), key=lambda path: str(path).lower(), reverse=True,
    )


def find_level_dirs(experiment_dir: Path | str) -> list[Path]:
    """列出实验根下的业务层级目录。"""
    root = Path(experiment_dir)
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and _LEVEL_PATTERN.match(path.name)),
        key=lambda path: _natural_key(path.name),
    )


def find_run_dirs(level_dir: Path | str) -> list[Path]:
    """递归发现全局模型和父类子模型的 run_* 目录。"""
    root = Path(level_dir)
    if not root.is_dir():
        return []
    return [Path(path) for path in cached_run_dirs(str(root.resolve()), ui_cache_version("analysis"))]


def _render_confusion_section(run_dir: Path, level_name: str) -> None:
    st.subheader("混淆矩阵")
    actions = st.columns(2)
    if actions[0].button("读取已有混淆矩阵", use_container_width=True):
        st.session_state["result_analysis_confusion_read"] = str(run_dir)
    if actions[1].button("重新计算混淆矩阵", type="primary", use_container_width=True):
        try:
            evaluate_model_run(run_dir, level_name)
            invalidate_ui_cache("analysis")
            st.session_state["result_analysis_confusion_read"] = str(run_dir)
            st.success("混淆矩阵已重新计算。")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            st.error(f"混淆矩阵计算失败：{error}")
    result_dir = run_dir / "val_result"
    metrics_path = result_dir / "metrics.json"
    image_path = result_dir / "confusion_matrix.png"
    if not result_dir.is_dir():
        st.info("尚无评估结果，请点击重新计算。")
        return
    if st.session_state.get("result_analysis_confusion_read") != str(run_dir):
        st.caption("已有评估结果；点击读取已有混淆矩阵后查看。")
        return
    if image_path.is_file():
        st.image(str(image_path), caption="Confusion matrix", use_container_width=True)
    if metrics_path.is_file():
        stat = metrics_path.stat()
        payload = cached_json_file(str(metrics_path.resolve()), stat.st_mtime_ns, stat.st_size)
        if not isinstance(payload, dict):
            return
        summary = payload.get("summary", {})
        columns = st.columns(3)
        columns[0].metric("Accuracy", _format_metric(summary.get("accuracy")))
        columns[1].metric("Macro F1", _format_metric(summary.get("macro_f1")))
        columns[2].metric("Macro Recall", _format_metric(summary.get("macro_recall")))
        with st.expander("分类报告", expanded=False):
            st.dataframe(payload.get("classes", []), use_container_width=True)


def _render_analysis_section(run_dir: Path, level_name: str) -> None:
    st.subheader("Analysis")
    actions = st.columns(2)
    if actions[0].button("读取已有 Analysis", use_container_width=True):
        st.session_state["result_analysis_analysis_read"] = str(run_dir)
    if actions[1].button("重新计算 Analysis", type="primary", use_container_width=True):
        try:
            run_interpret_run(run_dir, level_name, umap_enable=False)
            invalidate_ui_cache("analysis")
            st.session_state["result_analysis_analysis_read"] = str(run_dir)
            st.success("Analysis 已重新计算，未重新生成 UMAP。")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            st.error(f"Analysis 计算失败：{error}")
    analysis_dir = run_dir / "analysis_result"
    if not analysis_dir.is_dir():
        st.info("尚无 Analysis 结果，请点击重新计算。")
        return
    if st.session_state.get("result_analysis_analysis_read") != str(run_dir):
        st.caption("已有 Analysis 结果；点击读取已有 Analysis 后查看。")
        return
    _render_figures(analysis_dir / "figures", "Analysis 图表")
    log_path = run_dir / "analysis_result" / "logs" / "analysis_log.txt"
    if log_path.is_file():
        with st.expander("Analysis 日志", expanded=False):
            st.code(log_path.read_text(encoding="utf-8"), language="text")


def _render_umap_section(run_dir: Path, level_name: str) -> None:
    st.subheader("UMAP")
    actions = st.columns(2)
    if actions[0].button("读取已有 UMAP", use_container_width=True):
        st.session_state["result_analysis_umap_read"] = str(run_dir)
    if actions[1].button("重新计算 UMAP", type="primary", use_container_width=True):
        try:
            run_umap_run(run_dir, level_name)
            invalidate_ui_cache("analysis")
            st.session_state["result_analysis_umap_read"] = str(run_dir)
            st.success("UMAP 已重新计算。")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            st.error(f"UMAP 计算失败：{error}")
    image_path = run_dir / "analysis_result" / "figures" / "umap_hier_train_val.png"
    if not image_path.is_file():
        st.info("尚无 UMAP 图，请点击重新计算。")
    elif st.session_state.get("result_analysis_umap_read") == str(run_dir):
        st.image(str(image_path), caption="Train / Val UMAP", use_container_width=True)
    else:
        st.caption("已有 UMAP 图；点击读取已有 UMAP 后查看。")


def _render_figures(figure_dir: Path, caption: str) -> None:
    figures = sorted(path for path in figure_dir.glob("*.png") if path.name != "umap_hier_train_val.png") if figure_dir.is_dir() else []
    if not figures:
        st.info(f"尚无{caption}。")
        return
    names = [path.name for path in figures]
    selected = st.selectbox(caption, names, key=f"result_analysis_figure_{caption}")
    st.image(str(figures[names.index(selected)]), use_container_width=True)


def _selected_index(key: str, options: list[str]) -> int:
    value = st.session_state.get(key)
    return options.index(value) if value in options else 0


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _format_metric(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


__all__ = ["find_experiment_dirs", "find_level_dirs", "find_run_dirs", "render_result_analysis_page"]
