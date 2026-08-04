"""PCA-SVM 验证集 baseline。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ramanv2.data.input import InputPreprocessor
from ramanv2.evaluation.context import (
    EvaluationContext,
    RunEntry,
    load_evaluation_context,
    resolve_level_name,
    resolve_result_dir,
    resolve_run_entry,
)
from ramanv2.evaluation.report import write_baseline_report, write_pca_scatter


@dataclass(frozen=True)
class BaselineSpec:
    """PCA-SVM 的可复现实验参数。"""

    all_channels_enable: bool = False
    pca_components: float = 0.95
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    random_state: int = 42


def evaluate_baseline_run(
    source_dir: Path | str,
    level_name: str,
    spec: BaselineSpec = BaselineSpec(),
) -> Path:
    """在一个明确 run 的任务类别空间中拟合并评估 PCA-SVM。"""
    context = load_evaluation_context(source_dir)
    run_entry = resolve_run_entry(context)
    level_name = resolve_level_name(context, level_name)
    if run_entry.level_name != level_name:
        raise ValueError(
            f"run 属于 {run_entry.level_name}，不能按 {level_name} 评估"
        )
    class_ids = _resolve_run_class_ids(context, run_entry)
    train_indices = _select_class_indices(context, context.train_indices, level_name, class_ids)
    validation_indices = _select_class_indices(context, context.validation_indices, level_name, class_ids)
    class_names = [context.dataset_index.get_class_names(level_name)[index] for index in class_ids]
    train_features, train_labels, validation_labels, predictions, pca = _fit_predict(
        context,
        level_name,
        train_indices,
        validation_indices,
        class_ids,
        spec,
    )
    result_dir = resolve_result_dir(context, level_name, "baseline", "run", run_entry)
    output_dir = write_baseline_report(result_dir, class_names, validation_labels, predictions)
    _write_pca_summary(output_dir, pca)
    write_pca_scatter(output_dir, train_features, train_labels, class_names)
    return output_dir


def evaluate_baseline_parent_routed(
    source_dir: Path | str,
    level_name: str,
    spec: BaselineSpec = BaselineSpec(),
) -> Path:
    """按真实父类分别拟合 PCA-SVM 并汇总目标层全局指标。"""
    context = load_evaluation_context(source_dir)
    level_name = resolve_level_name(context, level_name)
    parent_level = context.dataset_index.get_parent_level(level_name)
    if parent_level is None:
        return _evaluate_global_baseline(context, level_name, spec, "parent-routed")
    level_index = context.dataset_index.head_name_to_idx[level_name]
    parent_index = context.dataset_index.head_name_to_idx[parent_level]
    mapping = (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    labels: list[int] = []
    predictions: list[int] = []
    for parent_text, child_values in sorted(mapping.items(), key=lambda item: int(item[0])):
        parent_id = int(parent_text)
        child_ids = [int(item) for item in child_values]
        train_indices = _select_parent_indices(
            context.dataset_index.level_labels,
            context.train_indices,
            parent_index,
            level_index,
            parent_id,
        )
        validation_indices = _select_parent_indices(
            context.dataset_index.level_labels,
            context.validation_indices,
            parent_index,
            level_index,
            parent_id,
        )
        if len(child_ids) == 1:
            valid_labels = context.dataset_index.level_labels[validation_indices, level_index]
            valid_labels = valid_labels[valid_labels >= 0]
            labels.extend(valid_labels.astype(int).tolist())
            predictions.extend([child_ids[0]] * len(valid_labels))
            continue
        _features, _train_labels, validation_labels, predicted_labels, _pca = _fit_predict(
            context,
            level_name,
            train_indices,
            validation_indices,
            child_ids,
            spec,
        )
        labels.extend(child_ids[int(label)] for label in validation_labels)
        predictions.extend(child_ids[int(label)] for label in predicted_labels)
    result_dir = resolve_result_dir(context, level_name, "baseline", "parent-routed")
    return write_baseline_report(
        result_dir,
        context.dataset_index.get_class_names(level_name),
        labels,
        predictions,
    )


def _evaluate_global_baseline(
    context: EvaluationContext,
    level_name: str,
    spec: BaselineSpec,
    mode: str,
) -> Path:
    """在一个完整目标层类别空间中拟合并评估 PCA-SVM。"""
    class_ids = list(range(context.dataset_index.num_classes_by_level[level_name]))
    train_features, train_labels, validation_labels, predictions, pca = _fit_predict(
        context,
        level_name,
        context.train_indices,
        context.validation_indices,
        class_ids,
        spec,
    )
    result_dir = resolve_result_dir(context, level_name, "baseline", mode)
    output_dir = write_baseline_report(
        result_dir,
        context.dataset_index.get_class_names(level_name),
        validation_labels,
        predictions,
    )
    _write_pca_summary(output_dir, pca)
    write_pca_scatter(
        output_dir,
        train_features,
        train_labels,
        context.dataset_index.get_class_names(level_name),
    )
    return output_dir


def _fit_predict(
    context: EvaluationContext,
    level_name: str,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    class_ids: Sequence[int],
    spec: BaselineSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """提取特征后完成标准化、PCA、SVM 训练与验证预测。"""
    train_features, train_labels = _extract_features(
        context,
        train_indices,
        level_name,
        class_ids,
        spec.all_channels_enable,
    )
    validation_features, validation_labels = _extract_features(
        context,
        validation_indices,
        level_name,
        class_ids,
        spec.all_channels_enable,
    )
    scaler = StandardScaler()
    train_standard = scaler.fit_transform(train_features)
    validation_standard = scaler.transform(validation_features)
    pca = PCA(n_components=spec.pca_components, random_state=spec.random_state)
    train_pca = pca.fit_transform(train_standard)
    validation_pca = pca.transform(validation_standard)
    svm = SVC(C=spec.svm_c, kernel=spec.svm_kernel, gamma=spec.svm_gamma)
    svm.fit(train_pca, train_labels)
    return train_pca, train_labels, validation_labels, svm.predict(validation_pca), pca


def _extract_features(
    context: EvaluationContext,
    indices: np.ndarray,
    level_name: str,
    class_ids: Sequence[int],
    all_channels_enable: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """从确定性模型输入提取指定类别空间的二维特征和局部标签。"""
    level_index = context.dataset_index.head_name_to_idx[level_name]
    local_by_global = {class_id: local_id for local_id, class_id in enumerate(class_ids)}
    preprocessor = InputPreprocessor(context.input_spec, "cpu")
    features: list[np.ndarray] = []
    labels: list[int] = []
    for sample_index in indices:
        global_label = int(context.dataset_index.level_labels[sample_index, level_index])
        if global_label not in local_by_global:
            continue
        values = preprocessor.preprocess_intensity(
            context.dataset_index.get_raw_intensity(int(sample_index))
        )[0].cpu().numpy()
        features.append(values.reshape(-1) if all_channels_enable else values[0])
        labels.append(local_by_global[global_label])
    if not features:
        raise RuntimeError("筛选后没有有效样本")
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def _select_class_indices(
    context: EvaluationContext,
    indices: np.ndarray,
    level_name: str,
    class_ids: Sequence[int],
) -> np.ndarray:
    """筛选属于指定目标层类别空间的样本索引。"""
    level_index = context.dataset_index.head_name_to_idx[level_name]
    labels = context.dataset_index.level_labels[indices, level_index]
    return indices[np.isin(labels, list(class_ids))]


def _select_parent_indices(
    labels: np.ndarray,
    indices: np.ndarray,
    parent_index: int,
    level_index: int,
    parent_id: int,
) -> np.ndarray:
    """筛选属于一个父类且目标层标签有效的样本索引。"""
    values = labels[indices]
    mask = (values[:, parent_index] == parent_id) & (values[:, level_index] >= 0)
    return indices[mask]


def _resolve_run_class_ids(context: EvaluationContext, run_entry: RunEntry) -> list[int]:
    """解析一个 global 或 parent run 的目标层全局类别标识。"""
    if run_entry.parent_id is None:
        return list(range(context.dataset_index.num_classes_by_level[run_entry.level_name]))
    return [int(item) for item in run_entry.values.get("child_ids") or []]


def _write_pca_summary(result_dir: Path, pca: PCA) -> None:
    """在 baseline 指标文本前写入 PCA 成分和解释方差。"""
    path = result_dir / "metrics.txt"
    report = path.read_text(encoding="utf-8")
    summary = (
        f"PCA components: {pca.n_components_}\n"
        "Explained variance ratio:\n"
        f"{np.array2string(pca.explained_variance_ratio_, precision=4)}\n\n"
    )
    path.write_text(summary + report, encoding="utf-8")
