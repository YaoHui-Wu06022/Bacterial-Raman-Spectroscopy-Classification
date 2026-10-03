"""层级训练与 Colab 数据包导出页面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ramanv2.core.config import build_config
from ramanv2.core.paths import PROJECT_ROOT
from ramanv2.packaging.data_archive import build_colab_bundle
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.profiles import get_profile, list_profiles, resolve_training_dir
from ramanv2.training.workflow import TrainRequest, run_training
from ramanv2.ui.cache import cached_dataset_index, cached_file_bytes, cached_run_dirs
from ramanv2.ui.state import invalidate_ui_cache, ui_cache_version


BATCH_SIZES = (8, 16, 32, 48, 64, 96, 128)
WORKER_COUNTS = (0, 1, 2, 4, 8)
TARGET_POINTS = (256, 512, 896, 1024, 2048)


def render_hierarchical_training_page() -> None:
    """渲染本地训练与 Colab 导出两个相互独立的操作区。"""
    st.title("层级训练")
    profiles = list_profiles()
    if not profiles:
        st.warning("当前没有可训练的数据集。")
        return

    profile_ids = [profile.profile_id for profile in profiles]
    profile_id = st.selectbox(
        "训练数据集",
        profile_ids,
        index=profile_ids.index("GN") if "GN" in profile_ids else 0,
        key="training_profile",
    )
    profile = get_profile(profile_id)
    train_dir = resolve_training_dir(profile_id)
    index = _load_dataset_index(train_dir)
    if index is None:
        st.info(f"未找到可训练数据：{train_dir}")
    else:
        _render_dataset_summary(index, train_dir)

    st.subheader("训练范围")
    scope = _render_training_scope(index)
    values = _render_config_controls(profile_id)
    run_options = _render_run_options(profile_id)

    if st.button("开始本地训练", type="primary", disabled=index is None, use_container_width=True):
        try:
            request = _build_train_request(values, scope, run_options)
            with st.status("正在准备训练…", expanded=True) as status:
                status.write("正在加载数据索引和训练配置。")
                status.write("训练任务运行中，请等待当前训练任务完成。")
                result = run_training(request)
                status.update(label="训练完成", state="complete", expanded=False)
            st.session_state["last_training_result"] = result
            invalidate_ui_cache("training")
            invalidate_ui_cache("analysis")
            st.success("本地训练完成。")
            st.caption("训练结果和层级元数据已保存到自动生成的实验目录。")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            st.error(f"本地训练失败：{error}")

    _render_colab_export(profiles)


def _load_dataset_index(train_dir: Path) -> DatasetIndex | None:
    return cached_dataset_index(str(train_dir.resolve()), ui_cache_version("training"))


def _render_dataset_summary(index: DatasetIndex, train_dir: Path) -> None:
    st.caption(f"训练目录：{train_dir}")
    metrics = st.columns(4)
    metrics[0].metric("光谱数量", f"{len(index)} 条")
    metrics[1].metric("层级数量", f"{len(index.level_names)} 层")
    metrics[2].metric("末级类别数", f"{len(index.class_names_by_level[-1])} 个")
    metrics[3].metric("父子映射", f"{sum(len(v) for v in index.parent_to_children.values())} 组")


def _render_training_scope(index: DatasetIndex | None) -> dict[str, object]:
    if index is None:
        return {"level_name": "level_1", "train_per_parent_enable": False}
    level_labels = {f"第{number}层": name for number, name in enumerate(index.level_names, 1)}
    selected_level = st.selectbox("训练层级", list(level_labels), key="training_level")
    level_name = level_labels[selected_level]
    parent_level = index.get_parent_level(level_name)
    per_parent_enable = False
    parent_name = None
    if parent_level is not None:
        per_parent_enable = st.checkbox("按父类分别训练子模型", value=True, key="train_per_parent")
        if per_parent_enable:
            parent_names = ["全部父类", *index.get_class_names(parent_level)]
            selected_parent = st.selectbox("训练父类", parent_names, key="training_parent")
            parent_name = None if selected_parent == "全部父类" else selected_parent
    else:
        st.caption("当前层级没有父类，将使用全局模型。")
    return {
        "level_name": level_name,
        "train_per_parent_enable": per_parent_enable,
        "only_parent_name": parent_name,
    }


def _render_config_controls(profile_id: str) -> dict[str, Any]:
    """用中文点击控件生成现有 build_config 所需的局部覆盖值。"""
    defaults = build_config({"profile_id": profile_id})
    training = defaults.training
    input_config = defaults.input
    model = defaults.model
    execution = defaults.execution
    epoch_default = int(training.epochs)
    patience_default = int(training.patience)
    split_default = float(training.train_split)
    batch_default = int(training.batch_size)
    worker_default = int(training.train_loader_num_workers)
    persistent_default = bool(training.loader_persistent_workers)
    amp_default = bool(execution.use_amp)

    values: dict[str, Any] = {
        "profile_id": profile_id,
        "epochs": st.slider("训练轮数", 1, 300, epoch_default, key="training_epochs"),
        "patience": st.slider("早停等待轮数", 1, 150, patience_default, key="training_patience"),
        "train_split": st.slider("训练集比例", 0.50, 0.95, split_default, 0.05, key="training_split"),
        "batch_size": st.selectbox("批大小", BATCH_SIZES, index=BATCH_SIZES.index(batch_default), key="training_batch"),
        "learning_rate": st.selectbox("学习率", (1e-5, 5e-5, 1e-4, 2e-4, 4e-4, 1e-3), index=4, key="training_lr"),
        "weight_decay": st.selectbox("权重衰减", (0.0, 1e-5, 1e-4, 1e-3), index=2, key="training_decay"),
        "train_loader_num_workers": st.selectbox("数据加载进程数", WORKER_COUNTS, index=WORKER_COUNTS.index(worker_default), key="training_workers"),
        "val_loader_num_workers": st.selectbox("验证加载进程数", WORKER_COUNTS, index=WORKER_COUNTS.index(worker_default), key="validation_workers"),
        "use_gpu": st.checkbox("使用显卡", value=execution.use_gpu, key="training_gpu"),
        "use_amp": st.checkbox("使用混合精度", value=amp_default, key="training_amp"),
        "deterministic": st.checkbox("启用确定性训练", value=execution.deterministic, key="training_deterministic"),
        "resume_training": st.checkbox("允许断点恢复", value=execution.resume_training, key="training_resume"),
        "norm_method": st.radio("输入归一化", ("SNV 标准化", "不归一化"), index=0 if input_config.norm_method == "snv" else 1, key="training_norm"),
        "smooth_use": st.checkbox("加入平滑谱", value=input_config.smooth_use, key="training_smooth"),
        "d1_use": st.checkbox("加入一阶导谱", value=input_config.d1_use, key="training_d1"),
        "target_points": st.selectbox("输入点数", TARGET_POINTS, index=TARGET_POINTS.index(input_config.target_points), key="training_points"),
        "backbone_type": st.radio("模型主干", ("卷积网络", "直接输入"), index=0 if model.backbone_type == "cnn" else 1, key="training_backbone"),
        "cnn_block_type": st.selectbox("卷积模块", ("ResNet", "ResNeXt"), index=1 if model.cnn_block_type == "resnext" else 0, key="training_block"),
        "encoder_type": st.selectbox("序列编码器", ("Transformer", "LSTM", "不使用"), index={"transformer": 0, "lstm": 1, "none": 2}.get(model.encoder_type, 0), key="training_encoder"),
        "pooling_type": st.selectbox("特征池化", ("注意力池化", "统计池化"), index=0 if model.pooling_type == "attn" else 1, key="training_pooling"),
    }
    with st.expander("更多训练设置", expanded=False):
        values["loader_pin_memory"] = st.checkbox("固定内存加载", value=training.loader_pin_memory, key="training_pin")
        values["loader_persistent_workers"] = st.checkbox("保持加载进程", value=persistent_default, key="training_persistent")
        values["loader_prefetch_factor"] = st.selectbox("预取批次数", (1, 2, 4, 8), index=1, key="training_prefetch")
        values["grad_clip_norm"] = st.select_slider("梯度裁剪", options=(0.0, 1.0, 2.0, 5.0, 10.0), value=training.grad_clip_norm, key="training_clip")
        values["scheduler_eta_min"] = st.select_slider("最低学习率", options=(0.0, 1e-6, 1e-5, 1e-4), value=training.scheduler_eta_min, key="training_eta")
        values["gamma"] = st.select_slider("Focal Loss 权重", options=(0.0, 0.5, 1.0, 1.2, 2.0), value=training.gamma, key="training_gamma")
        values["early_stop_w_f1"] = st.select_slider("早停 F1 权重", options=(0.0, 0.25, 0.5, 0.6, 0.75, 1.0), value=training.early_stop_w_f1, key="training_stop_f1")
        values["early_stop_w_acc"] = st.select_slider("早停准确率权重", options=(0.0, 0.25, 0.4, 0.5, 0.75, 1.0), value=training.early_stop_w_acc, key="training_stop_acc")
        values["se_use"] = st.checkbox("启用通道注意力", value=model.se_use, key="training_se")
    loss_columns = st.columns(3)
    values["use_align_loss"] = loss_columns[0].checkbox(
        "Align Loss",
        value=training.use_align_loss,
        key="training_align",
    )
    values["use_supcon_loss"] = loss_columns[1].checkbox(
        "SupCon Loss",
        value=training.use_supcon_loss,
        key="training_supcon",
    )
    values["use_ema"] = loss_columns[2].checkbox(
        "EMA",
        value=training.use_ema,
        key="training_ema",
    )
    values["norm_method"] = "snv" if values["norm_method"] == "SNV 标准化" else "none"
    values["backbone_type"] = "cnn" if values["backbone_type"] == "卷积网络" else "direct"
    values["cnn_block_type"] = "resnext" if values["cnn_block_type"] == "ResNeXt" else "resnet"
    values["encoder_type"] = {"Transformer": "transformer", "LSTM": "lstm", "不使用": "none"}[values["encoder_type"]]
    values["pooling_type"] = "attn" if values["pooling_type"] == "注意力池化" else "stat"
    return values


def _render_run_options(profile_id: str) -> dict[str, Path | str | None]:
    """扫描已有运行目录，以选择方式提供断点恢复。"""
    root = PROJECT_ROOT / "output" / profile_id
    choices = [
        Path(path)
        for path in reversed(cached_run_dirs(str(root.resolve()), ui_cache_version("training")))
    ] if root.is_dir() else []
    labels = ["不恢复"] + [str(path.relative_to(PROJECT_ROOT)) for path in choices]
    selected = st.selectbox("恢复已有训练", labels, key="training_resume_run")
    return {
        "experiment_dir": None,
        "run_name": None,
        "resume_run_dir": None if selected == "不恢复" else choices[labels.index(selected) - 1],
    }


def _build_train_request(
    values: dict[str, Any],
    scope: dict[str, object],
    run_options: dict[str, Path | str | None],
) -> TrainRequest:
    config = build_config(values)
    return TrainRequest(
        config=config,
        level_name=str(scope["level_name"]),
        only_parent_name=scope.get("only_parent_name"),
        train_per_parent_enable=bool(scope.get("train_per_parent_enable", False)),
        experiment_dir=run_options.get("experiment_dir"),
        run_name=run_options.get("run_name"),
        resume_run_dir=run_options.get("resume_run_dir"),
    )


def _render_colab_export(profiles) -> None:
    st.divider()
    st.subheader("Colab 数据包")
    profile_ids = [profile.profile_id for profile in profiles]
    profile_id = st.selectbox(
        "导出数据集",
        profile_ids,
        index=profile_ids.index("GN") if "GN" in profile_ids else 0,
        key="colab_export_profile",
    )
    output_dir = PROJECT_ROOT / "output" / "colab" / profile_id
    st.caption(f"输出目录：{output_dir}")
    if st.button("生成 data.zip 和 ramanv2.zip", use_container_width=True):
        try:
            with st.spinner("正在生成 Colab 压缩包…"):
                report = build_colab_bundle(profile_id, output_dir)
            st.success("Colab 压缩包生成完成。")
            data_report = report.get("data_archive", {})
            package_report = report.get("package_archive", {})
            if isinstance(data_report, dict) and isinstance(package_report, dict):
                st.write(f"样本数量：{report.get('sample_count', 0)}")
                st.write(f"数据包路径：{data_report.get('output_path', '')}")
                st.write(f"源码包路径：{package_report.get('output_path', '')}")
            _render_download_buttons(report)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            st.error(f"Colab 压缩包生成失败：{error}")


def _render_download_buttons(report: dict[str, object]) -> None:
    data_report = report.get("data_archive", {})
    package_report = report.get("package_archive", {})
    for label, item, key in (
        ("下载 data.zip", data_report, "download_data_zip"),
        ("下载 ramanv2.zip", package_report, "download_ramanv2_zip"),
    ):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("output_path", "")))
        if path.is_file():
            stat = path.stat()
            data = cached_file_bytes(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            st.download_button(label, data, file_name=path.name, key=key)


__all__ = ["render_hierarchical_training_page"]
