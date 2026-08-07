"""输入通道和类别波段的 Integrated Gradients 归因。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ramanv2.data.input import InputPreprocessor


@dataclass(frozen=True)
class IntegratedGradientsResult:
    """一个模型任务的输入归因及类别平均谱统计。"""

    channel_importance: np.ndarray
    band_importance: np.ndarray
    sample_counts: np.ndarray
    mean_spectra: np.ndarray
    mean_counts: np.ndarray
    sample_count: int


def collect_task_inputs(
    context,
    task,
    split_name: str,
    device: torch.device,
    sample_indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将任务范围内 train 或 val 的原始谱转换为模型输入张量。"""
    indices = task.train_indices if split_name == "train" else task.validation_indices
    if sample_indices is not None:
        indices = np.asarray(sample_indices, dtype=np.int64)
    level_index = context.dataset_index.head_name_to_idx[task.level_name]
    local_ids = {class_id: index for index, class_id in enumerate(task.class_ids)}
    preprocessor = InputPreprocessor(context.input_spec, device)
    values = []
    labels = []
    for sample_index in indices:
        class_id = int(context.dataset_index.level_labels[sample_index, level_index])
        if class_id not in local_ids:
            continue
        intensity = context.dataset_index.get_raw_intensity(int(sample_index))
        values.append(preprocessor.preprocess_intensity(intensity)[0])
        labels.append(local_ids[class_id])
    if not values:
        raise RuntimeError("分析范围没有有效样本")
    return torch.stack(values), torch.tensor(labels, dtype=torch.long, device=device)


def select_balanced_task_sample_indices(
    context,
    task,
    split_name: str,
    total_limit: int,
    max_per_class: int,
) -> np.ndarray:
    """在预处理前按类别均衡抽取样本索引，避免处理未参与 IG 的全部光谱。"""
    if total_limit < 1:
        raise ValueError("归因样本总数上限必须至少为 1")
    if max_per_class < 1:
        raise ValueError("每类归因样本上限必须至少为 1")
    source_indices = (
        task.train_indices if split_name == "train" else task.validation_indices
    )
    level_index = context.dataset_index.head_name_to_idx[task.level_name]
    local_ids = {class_id: index for index, class_id in enumerate(task.class_ids)}
    local_labels = np.asarray(
        [
            local_ids.get(int(context.dataset_index.level_labels[index, level_index]), -1)
            for index in source_indices
        ],
        dtype=np.int64,
    )
    valid_positions = np.flatnonzero(local_labels >= 0)
    valid_labels = local_labels[valid_positions]
    class_ids = np.unique(valid_labels)
    if class_ids.size == 0:
        raise RuntimeError("分析范围没有有效样本")
    if class_ids.size > total_limit:
        raise ValueError("归因样本总数上限小于实际类别数")
    quota = min(max_per_class, total_limit // class_ids.size)
    selected_positions = np.sort(np.concatenate(
        [
            valid_positions[np.flatnonzero(valid_labels == class_id)[:quota]]
            for class_id in class_ids
        ]
    ))
    return np.asarray(source_indices[selected_positions], dtype=np.int64)


def select_balanced_class_inputs(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    total_limit: int,
    max_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在归因预算内按类别稳定抽取输入，避免索引排序造成类别缺失。"""
    if inputs.size(0) != labels.size(0):
        raise ValueError("归因输入和标签数量不一致")
    if total_limit < 1:
        raise ValueError("归因样本总数上限必须至少为 1")
    if max_per_class < 1:
        raise ValueError("每类归因样本上限必须至少为 1")
    class_ids = torch.unique(labels, sorted=True)
    if len(class_ids) > total_limit:
        raise ValueError("归因样本总数上限小于实际类别数")
    quota = min(max_per_class, total_limit // len(class_ids))
    selected = []
    for class_id in class_ids:
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        selected.append(class_indices[:quota])
    selected_indices = torch.cat(selected).sort().values
    return inputs.index_select(0, selected_indices), labels.index_select(0, selected_indices)


def compute_integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    ig_steps: int,
    max_per_class: int,
    class_count: int,
) -> IntegratedGradientsResult:
    """计算输入通道、每类波段重要性与每类平均主通道光谱。"""
    if inputs.size(0) != labels.size(0):
        raise ValueError("归因输入和标签数量不一致")
    if ig_steps < 1:
        raise ValueError("ig_steps 必须至少为 1")
    baseline = inputs[: min(batch_size, len(inputs))].mean(dim=0, keepdim=True)
    channel_total = torch.zeros(inputs.size(1), device=inputs.device)
    if class_count < int(labels.max().item()) + 1:
        raise ValueError("归因类别数小于标签范围")
    band_total = np.zeros((class_count, inputs.size(2)), dtype=np.float64)
    sample_counts = np.zeros(class_count, dtype=np.int64)
    for start in range(0, len(inputs), batch_size):
        values = inputs[start : start + batch_size]
        targets = labels[start : start + batch_size]
        gradients = torch.zeros_like(values)
        for alpha in torch.linspace(0, 1, ig_steps, device=inputs.device):
            point = (baseline + alpha * (values - baseline)).detach().requires_grad_(True)
            logits = model(point)
            score = logits.gather(1, targets[:, None]).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            gradients += point.grad.detach()
        attribution = (values - baseline) * gradients / float(ig_steps)
        channel_total += attribution.abs().mean(dim=(0, 2))
        bands = attribution.abs().mean(dim=1).detach().cpu().numpy()
        for index, class_id in enumerate(targets.detach().cpu().tolist()):
            if sample_counts[class_id] >= max_per_class:
                continue
            band_total[class_id] += bands[index]
            sample_counts[class_id] += 1
    batch_count = max(1, (len(inputs) + batch_size - 1) // batch_size)
    channel = (channel_total / batch_count).detach().cpu().numpy()
    channel /= channel.sum() + 1e-8
    band = band_total / np.maximum(sample_counts[:, None], 1)
    mean_spectra, mean_counts = _compute_mean_spectra(inputs, labels, class_count, max_per_class)
    return IntegratedGradientsResult(
        channel_importance=channel,
        band_importance=band,
        sample_counts=sample_counts,
        mean_spectra=mean_spectra,
        mean_counts=mean_counts,
        sample_count=len(inputs),
    )


def _compute_mean_spectra(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
    max_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按类别聚合模型主输入通道，用于波段图中的平均谱。"""
    mean_total = np.zeros((class_count, inputs.size(2)), dtype=np.float64)
    mean_counts = np.zeros(class_count, dtype=np.int64)
    spectra = inputs[:, 0].detach().cpu().numpy()
    for index, class_id in enumerate(labels.detach().cpu().tolist()):
        if mean_counts[class_id] >= max_per_class:
            continue
        mean_total[class_id] += spectra[index]
        mean_counts[class_id] += 1
    return mean_total / np.maximum(mean_counts[:, None], 1), mean_counts
