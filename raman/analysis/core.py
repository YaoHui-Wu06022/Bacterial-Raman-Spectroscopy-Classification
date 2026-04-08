# 训练结果分析核心逻辑（供根目录 analyze.py 调用）

import os
import json
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from torch.utils.data import DataLoader, Subset, Dataset

from raman.config_io import load_experiment
from raman.data import RamanDataset, resolve_dataset_stage
from raman.model import RamanClassifier1D, SEBlock1D
from raman.training import (
    AutoHierarchicalBatchSampler,
    build_label_map_np,
    split_by_lowest_level_ratio,
    load_split_files
)
from .utils import (
    _compute_baseline_mean_spectrum,
    _needs_cudnn_rnn_guard,
    compute_input_channel_importance_IG,
    collect_analyzable_layers,
    LayerGradCAMAnalyzer,
    merge_scores_by_group,
    collect_embeddings,
    collect_embeddings_train_test,
    plot_embedding_hierarchical,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path):
    """将相对路径解析到项目根目录，绝对路径保持不变。"""
    if path is None:
        return path
    path = Path(path)
    if path.is_absolute():
        return os.fspath(path)
    return os.fspath((PROJECT_ROOT / path).resolve())


def _load_hierarchy_meta(exp_dir):
    """读取层级训练元数据，解析 parent 模型映射。"""
    meta_path = os.path.join(exp_dir, "hierarchy_meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    parent_models_raw = meta.get("parent_models", {})
    parent_models = {}
    for level, mapping in parent_models_raw.items():
        parent_models[level] = {}
        for k, v in mapping.items():
            entry = dict(v)
            entry["child_ids"] = [int(c) for c in entry.get("child_ids", [])]
            parent_models[level][int(k)] = entry

    meta["parent_models"] = parent_models
    meta["level_models"] = meta.get("level_models", {})
    return meta


def resolve_analysis_level(dataset, level_name, config):
    """解析分析层级，None 时回退到当前训练层级。"""
    if level_name is None:
        level_name = (
            getattr(config, "current_train_level", None)
            or "leaf"
        )
    if level_name not in dataset.head_names:
        raise ValueError(
            f"Unknown analysis level: {level_name}. Available: {dataset.head_names}"
        )
    return level_name


class LabelMapDataset(Dataset):
    # 将某一层级的标签映射为局部索引（用于父类内子模型分析）
    def __init__(self, base_dataset, level_idx, label_map_np):
        self.base_dataset = base_dataset
        self.level_idx = level_idx
        self.label_map_np = label_map_np

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, labels, hier = self.base_dataset[idx]
        labels = np.array(labels, copy=True)
        if labels[self.level_idx] >= 0:
            labels[self.level_idx] = self.label_map_np[labels[self.level_idx]]
        else:
            labels[self.level_idx] = -1
        return x, labels, hier


@dataclass
class HeatmapConfig:
    # 波段重要性热图（IG）相关配置
    num_batches: int = 10          # 采样多少个 batch
    steps: int = 32                # IG 积分步数
    max_per_class: int = 50        # 每类最多样本数
    target_mode: str = "true"      # true=真实标签，pred=预测标签
    row_norm: str = "max"          # 行归一化方式：max / sum / none
    use_train_loader: bool = True  # True=训练集，False=测试集
    topk_per_class: int = 5        # 每类 top-k 波段导出


@dataclass
class AnalysisOverrides:
    """统一收拢分析入口的覆盖项。"""

    exp_dir: str | None = None
    mode: str = "single"
    analysis_level: str | None = None
    parent_idx: int | str | None = None
    use_train_aug: bool = True
    inherit_missing_levels: bool = False
    fallback_to_single: bool = True


def _ensure_heatmap_cfg(cfg):
    # None 时返回默认热图配置，避免外部传空
    return cfg if cfg is not None else HeatmapConfig()


def _normalize_parent_idx(parent_idx):
    """统一 parent_idx 输入格式（None/int/'all'）。"""
    if parent_idx is None:
        return None
    if isinstance(parent_idx, str):
        text = parent_idx.strip().lower()
        if text == "all":
            return "all"
        if text.isdigit():
            return int(text)
        raise ValueError(
            f"Unknown PARENT_IDX value: {parent_idx}. Use int or 'all'."
        )
    return int(parent_idx)


def _build_analysis_tasks(
    exp_dir,
    analysis_level,
    head_index,
    full_dataset,
    level_models,
    parent_models,
    parent_idx_setting,
):
    """解析需要分析的模型任务（单模型或按 parent 拆分）。"""
    parent_idx_setting = _normalize_parent_idx(parent_idx_setting)
    parent_entries = parent_models.get(analysis_level, {})
    tasks = []
    auto_all = False

    if parent_idx_setting is None:
        model_name = level_models.get(analysis_level, f"{analysis_level}_model.pt")
        model_path = os.path.join(exp_dir, model_name)
        if os.path.exists(model_path):
            num_classes = full_dataset.num_classes_by_level[analysis_level]
            class_names = full_dataset.class_names_by_level[head_index]
            tasks.append(
                {
                    "parent_idx": None,
                    "model_path": model_path,
                    "child_ids": None,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "tag": analysis_level,
                }
            )
            return tasks, auto_all

        if parent_entries:
            parent_idx_setting = "all"
            auto_all = True
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}. If you trained per-parent, set PARENT_IDX."
            )

    if parent_idx_setting == "all":
        if not parent_entries:
            raise ValueError(
                f"No parent model entries for level={analysis_level}."
            )
        for parent_idx in sorted(parent_entries.keys()):
            entry = parent_entries[parent_idx]
            if entry.get("model_path") is None:
                continue
            model_path = entry["model_path"]
            if not os.path.isabs(model_path):
                model_path = os.path.join(exp_dir, model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Parent model not found: {model_path}")
            child_ids = entry.get("child_ids", [])
            class_names = [
                full_dataset.class_names_by_level[head_index][cid]
                for cid in child_ids
            ]
            tasks.append(
                {
                    "parent_idx": parent_idx,
                    "model_path": model_path,
                    "child_ids": child_ids,
                    "num_classes": len(child_ids),
                    "class_names": class_names,
                    "tag": f"{analysis_level}_parent_{parent_idx}",
                }
            )

        if not tasks:
            raise ValueError(
                f"No parent models to analyze for level={analysis_level}."
            )
        return tasks, auto_all

    parent_idx = int(parent_idx_setting)
    entry = parent_entries.get(parent_idx)
    if entry is None:
        raise ValueError(
            f"No parent model entry for level={analysis_level}, parent={parent_idx}"
        )
    if entry.get("model_path") is None:
        raise ValueError(
            f"Parent {parent_idx} has only one child; no model to analyze."
        )

    model_path = entry["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(exp_dir, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Parent model not found: {model_path}")

    child_ids = entry.get("child_ids", [])
    class_names = [
        full_dataset.class_names_by_level[head_index][cid]
        for cid in child_ids
    ]
    tasks.append(
        {
            "parent_idx": parent_idx,
            "model_path": model_path,
            "child_ids": child_ids,
            "num_classes": len(child_ids),
            "class_names": class_names,
            "tag": f"{analysis_level}_parent_{parent_idx}",
        }
    )
    return tasks, auto_all


def _effective_label_names(dataset, level_name, missing_tag="__missing__"):
    if hasattr(dataset, "_resolve_level_name"):
        level_name = dataset._resolve_level_name(level_name)
    names = []
    seen = set()
    for hier in dataset.hier_names:
        name = hier.get(level_name)
        if name is None or name == missing_tag:
            name = hier.get("leaf")
        if name is None:
            name = missing_tag
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _batch_effective_label_ids(hier, level_name, name_to_idx, missing_tag="__missing__"):
    if hier is None:
        return None
    level_vals = hier.get(level_name, [])
    leaf_vals = hier.get("leaf", [])
    ids = []
    for i, v in enumerate(level_vals):
        if v is None or v == missing_tag:
            v = leaf_vals[i] if i < len(leaf_vals) else v
        if v is None:
            v = missing_tag
        ids.append(name_to_idx.get(v, -1))
    return np.asarray(ids, dtype=np.int64)


def compute_class_band_importance_ig(
    model,
    loader,
    device,
    num_classes,
    head_index,
    steps=32,
    num_batches=6,
    baseline_mode="mean",
    target_mode="true",
    max_per_class=None,
    level_name=None,
    label_name_to_idx=None,
    missing_tag="__missing__",
):
    """计算每个类别的波段重要性（IG 平均）。"""
    model.eval()
    disable_cudnn = _needs_cudnn_rnn_guard(model)
    prev_cudnn = torch.backends.cudnn.enabled
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    if baseline_mode == "zero":
        baseline = None
    elif baseline_mode == "mean":
        baseline = _compute_baseline_mean_spectrum(loader, device, num_batches=num_batches)
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    total = None
    counts = np.zeros(num_classes, dtype=np.int64)
    it = iter(loader)

    for _ in range(num_batches):
        try:
            x, y, hier = next(it)
        except StopIteration:
            break

        x = x.to(device)
        y = y.to(device)

        if y.ndim == 2:
            y = y[:, head_index]

        if label_name_to_idx is not None and level_name is not None:
            group_ids = _batch_effective_label_ids(hier, level_name, label_name_to_idx, missing_tag)
            if group_ids is None:
                continue
            group_ids_t = torch.from_numpy(group_ids).to(y.device)
            valid_mask = group_ids_t >= 0
        else:
            group_ids_t = y
            valid_mask = y >= 0
        if not valid_mask.any():
            continue

        if baseline_mode == "zero":
            b = torch.zeros_like(x)
        else:
            b = baseline.expand(x.size(0), -1, -1)

        with torch.no_grad():
            logits0 = model(x)
            if target_mode == "pred":
                target = logits0.argmax(dim=1)
            else:
                # If true label missing, fallback to pred.
                pred = logits0.argmax(dim=1)
                target = torch.where(y >= 0, y, pred)

        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        total_grad = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, steps, device=device)

        for alpha in alphas:
            x_step = (b + alpha * (x - b)).detach().requires_grad_(True)
            logits = model(x_step)
            score_each = logits.gather(1, safe_target.view(-1, 1)).squeeze(1)
            score = (score_each * valid_mask.float()).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            total_grad += x_step.grad.detach()

        avg_grad = total_grad / float(steps)
        ig = (x - b) * avg_grad
        ig_band = ig.abs().mean(dim=1)

        ig_np = ig_band.detach().cpu().numpy()
        valid_np = valid_mask.detach().cpu().numpy()

        if total is None:
            total = np.zeros((num_classes, ig_np.shape[1]), dtype=np.float32)

        for i in range(ig_np.shape[0]):
            if not valid_np[i]:
                continue
            cls = int(group_ids_t[i].item())
            if cls < 0 or cls >= num_classes:
                continue
            if max_per_class is not None and counts[cls] >= max_per_class:
                continue
            total[cls] += ig_np[i]
            counts[cls] += 1

    if total is None:
        if disable_cudnn:
            torch.backends.cudnn.enabled = prev_cudnn
        raise RuntimeError("No valid samples for band importance heatmap.")

    avg = total / np.maximum(counts[:, None], 1)
    if disable_cudnn:
        torch.backends.cudnn.enabled = prev_cudnn
    return avg, counts

def compute_class_mean_spectrum(
    loader,
    device,
    num_classes,
    head_index,
    max_per_class=None,
    level_name=None,
    label_name_to_idx=None,
    missing_tag="__missing__",
):
    """统计每个类别的平均光谱（默认使用第0个通道）。"""
    sums = None
    counts = np.zeros(num_classes, dtype=np.int64)

    for x, y, hier in loader:
        x = x.to(device)
        y = y.to(device)

        if y.ndim == 2:
            y = y[:, head_index]

        if label_name_to_idx is not None and level_name is not None:
            group_ids = _batch_effective_label_ids(hier, level_name, label_name_to_idx, missing_tag)
            if group_ids is None:
                continue
            group_ids_t = torch.from_numpy(group_ids).to(y.device)
            valid_mask = group_ids_t >= 0
        else:
            group_ids_t = y
            valid_mask = y >= 0
        if not valid_mask.any():
            continue

        # 兼容单通道/多通道输入
        if x.ndim == 3:
            spectra = x[:, 0, :]
        else:
            spectra = x

        spectra_np = spectra.detach().cpu().numpy()
        labels_np = group_ids_t.detach().cpu().numpy()
        valid_np = valid_mask.detach().cpu().numpy()

        if sums is None:
            sums = np.zeros((num_classes, spectra_np.shape[1]), dtype=np.float32)

        for i in range(spectra_np.shape[0]):
            if not valid_np[i]:
                continue
            cls = int(labels_np[i])
            if cls < 0 or cls >= num_classes:
                continue
            if max_per_class is not None and counts[cls] >= max_per_class:
                continue
            sums[cls] += spectra_np[i]
            counts[cls] += 1

    if sums is None:
        raise RuntimeError("No valid samples for mean spectrum.")

    mean = sums / np.maximum(counts[:, None], 1)
    return mean, counts

def _normalize_bad_bands(bad_bands):
    # 统一坏段格式并保证区间有序
    if not bad_bands:
        return []
    out = []
    for band in bad_bands:
        if band is None or len(band) < 2:
            continue
        b0 = float(band[0])
        b1 = float(band[1])
        if b1 < b0:
            b0, b1 = b1, b0
        out.append((b0, b1))
    return out


def _get_bad_bands(config):
    # 兼容 BAD_BANDS / bad_bands 两种写法
    if hasattr(config, "BAD_BANDS"):
        return _normalize_bad_bands(config.BAD_BANDS)
    if hasattr(config, "bad_bands"):
        return _normalize_bad_bands(config.bad_bands)
    return []


def _apply_bad_bands(wn, bad_bands):
    # 删除坏段对应的波数区间
    if not bad_bands:
        return wn
    mask = np.ones_like(wn, dtype=bool)
    for b0, b1 in bad_bands:
        mask &= ~((wn >= b0) & (wn <= b1))
    return wn[mask]


def _estimate_gap_indices(wavenumbers, gap_factor=1.5):
    # 通过相邻步长突变估计“坏段”缺口位置
    if wavenumbers is None or len(wavenumbers) < 2:
        return []
    diffs = np.diff(wavenumbers)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return []
    step = np.median(diffs)
    if step <= 0:
        return []
    gap_idx = np.where(np.diff(wavenumbers) > step * gap_factor)[0]
    return gap_idx.tolist()


def build_wavenumber_axis(length, config):
    """构建波数坐标轴（根据 config.cut_min / delta）。"""
    bad_bands = _get_bad_bands(config)
    if hasattr(config, "cut_min") and hasattr(config, "cut_max"):
        if hasattr(config, "target_points"):
            try:
                target_points = int(config.target_points)
            except Exception:
                target_points = None
            if target_points:
                # 按预处理方式先生成全量，再删掉坏段，确保与实际输入长度一致
                wn_full = np.linspace(config.cut_min, config.cut_max, target_points)
                wn_full = _apply_bad_bands(wn_full, bad_bands)
                if wn_full.shape[0] == length:
                    return wn_full
        if hasattr(config, "delta"):
            return config.cut_min + config.delta * np.arange(length)
        return np.linspace(config.cut_min, config.cut_max, length)
    return np.arange(length)

def plot_band_importance_heatmap(
    importance,
    counts,
    class_names,
    wavenumbers,
    save_path,
    row_norm="max",
    mean_spectra=None,
    bad_bands=None,
):
    """按类别绘制：曲线为平均光谱，曲线下颜色表示波段重要性。"""
    data = importance.copy()
    if row_norm == "max":
        denom = np.max(data, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        data = data / denom
    elif row_norm == "sum":
        denom = np.sum(data, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        data = data / denom

    if mean_spectra is None:
        raise ValueError("mean_spectra is required for spectrum-style heatmap.")

    use_wavenumber_axis = not (wavenumbers is None or len(wavenumbers) != data.shape[1])
    if not use_wavenumber_axis:
        wavenumbers = np.arange(data.shape[1])
        x_label = "Band Index"
    else:
        x_label = "Wavenumber"

    fig_height = max(4, 0.6 * len(class_names))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    gap_indices = _estimate_gap_indices(wavenumbers)
    gap_set = set(gap_indices)

    if bad_bands and use_wavenumber_axis:
        # 在波数坐标上用灰色阴影标记坏段
        for b0, b1 in _normalize_bad_bands(bad_bands):
            ax.axvspan(b0, b1, color="gray", alpha=0.15, zorder=0)

    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for i, name in enumerate(class_names):
        y = mean_spectra[i]
        if y.size == 0:
            continue

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if y_max - y_min < 1e-8:
            y_norm = np.zeros_like(y)
        else:
            y_norm = (y - y_min) / (y_max - y_min)

        height = 0.8
        baseline = i
        y_plot = baseline + y_norm * height

        imp_row = data[i]
        imp_seg = 0.5 * (imp_row[:-1] + imp_row[1:])

        polys = []
        colors = []
        for j in range(len(wavenumbers) - 1):
            if j in gap_set:
                # 缺口处不画填充，避免跨坏段连接
                continue
            x0 = wavenumbers[j]
            x1 = wavenumbers[j + 1]
            polys.append([
                (x0, baseline),
                (x0, y_plot[j]),
                (x1, y_plot[j + 1]),
                (x1, baseline),
            ])
            colors.append(cmap(norm(imp_seg[j])))

        collection = PolyCollection(polys, facecolors=colors, edgecolors='none', alpha=0.9)
        ax.add_collection(collection)
        if gap_indices:
            start = 0
            for gap in gap_indices:
                end = gap + 1
                if end - start > 1:
                    ax.plot(
                        wavenumbers[start:end],
                        y_plot[start:end],
                        color="#1f1f1f",
                        linewidth=1.0,
                    )
                start = end
            if len(wavenumbers) - start > 1:
                ax.plot(
                    wavenumbers[start:],
                    y_plot[start:],
                    color="#1f1f1f",
                    linewidth=1.0,
                )
        else:
            ax.plot(wavenumbers, y_plot, color="#1f1f1f", linewidth=1.0)

    ax.set_yticks([i + 0.4 for i in range(len(class_names))])
    ax.set_yticklabels(class_names, fontsize=8)

    tick_count = 6
    idx = np.linspace(0, len(wavenumbers) - 1, tick_count, dtype=int)
    ax.set_xticks(wavenumbers[idx])
    ax.set_xticklabels([f"{wavenumbers[i]:.0f}" for i in idx])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Class")
    ax.set_title("Band Importance (Mean Spectrum with Colored Area)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Importance (normalized)")

    ax.set_ylim(-0.2, len(class_names) - 0.2 + 1.0)
    ax.set_xlim(wavenumbers[0], wavenumbers[-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def save_topk_bands_csv(
    importance, class_names, wavenumbers, top_k, save_path, row_norm="max"
):
    """导出每类 top-k 波段到 CSV。"""
    data = importance.copy()
    if row_norm == "max":
        denom = np.max(data, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        data = data / denom
    elif row_norm == "sum":
        denom = np.sum(data, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        data = data / denom
    elif row_norm != "none":
        raise ValueError(f"Unknown row_norm: {row_norm}")

    if wavenumbers is None or len(wavenumbers) != data.shape[1]:
        wavenumbers = np.arange(data.shape[1])
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "rank", "band_index", "wavenumber", "importance"])
        for i, name in enumerate(class_names):
            row = data[i]
            if row.size == 0:
                continue
            top_idx = np.argsort(row)[::-1][:top_k]
            for rank, idx in enumerate(top_idx, 1):
                writer.writerow(
                    [name, rank, int(idx), float(wavenumbers[idx]), float(row[idx])]
                )


def _plot_channel_importance(importance, channel_names, save_path):
    """聚合模式下的通道重要性柱状图。"""
    if isinstance(importance, torch.Tensor):
        importance = importance.detach().cpu().numpy()
    else:
        importance = np.asarray(importance)
    plt.figure(figsize=(7, 4))
    xnames = channel_names[: len(importance)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(xnames)))
    plt.bar(xnames, importance, color=colors)
    plt.title("Input Channel Contribution (Aggregated)")
    plt.ylabel("Relative Importance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_layer_importance(scores, save_path):
    """聚合模式下的层级重要性柱状图。"""
    names = list(scores.keys())
    vals = list(scores.values())
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.xticks(rotation=60)
    plt.ylabel("Layer Importance (Aggregated |A x G|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _build_task_loaders(
    task,
    config,
    full_dataset,
    analysis_level,
    head_index,
    train_idx_all,
    test_idx_all,
    base_train_dataset,
    base_test_dataset,
):
    """按 parent 过滤样本并构建 DataLoader。"""
    parent_idx = task["parent_idx"]
    child_ids = task["child_ids"]
    inherit_missing = getattr(config, "inherit_missing_levels", False)

    train_idx = train_idx_all
    test_idx = test_idx_all

    if parent_idx is not None:
        parent_level = full_dataset.get_parent_level(analysis_level)
        if parent_level is None:
            raise ValueError("Top level has no parent; cannot use PARENT_IDX.")
        parent_level_idx = full_dataset.head_name_to_idx[parent_level]

        labels_train = full_dataset.level_labels[train_idx]
        labels_test = full_dataset.level_labels[test_idx]

        train_mask = (labels_train[:, parent_level_idx] == parent_idx)
        test_mask = (labels_test[:, parent_level_idx] == parent_idx)
        if not inherit_missing:
            train_mask = train_mask & (labels_train[:, head_index] >= 0)
            test_mask = test_mask & (labels_test[:, head_index] >= 0)

        train_idx = train_idx[train_mask]
        test_idx = test_idx[test_mask]

        label_map_np = build_label_map_np(
            child_ids,
            full_dataset.num_classes_by_level[analysis_level]
        )

        train_dataset = LabelMapDataset(base_train_dataset, head_index, label_map_np)
        test_dataset = LabelMapDataset(base_test_dataset, head_index, label_map_np)
    else:
        train_dataset = base_train_dataset
        test_dataset = base_test_dataset

    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(test_dataset, test_idx)

    use_hier_sampler = getattr(config, "use_hier_sampler", False)
    if use_hier_sampler and parent_idx is None:
        top_level = getattr(config, "hier_sampler_top", None)
        if top_level is None:
            top_level = full_dataset.head_names[0] if full_dataset.head_names else "leaf"
        sampler = AutoHierarchicalBatchSampler(
            dataset=train_subset,
            batch_size=config.batch_size,
            top_level=top_level,
            leaf_level="leaf",
            min_samples_per_leaf=2
        )
        train_loader = DataLoader(
            train_subset,
            batch_sampler=sampler,
            num_workers=2
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )

    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False
    )

    if len(test_subset) == 0:
        test_loader = train_loader

    return train_loader, test_loader, train_subset, test_subset


def run_aggregate_analysis(
    exp_dir,
    config,
    full_dataset,
    analysis_level,
    head_index,
    tasks,
    train_idx_all,
    test_idx_all,
    base_train_dataset,
    base_test_dataset,
    heatmap_cfg=None,
):
    """跨 parent 聚合分析：用样本数加权合并结果。"""
    heatmap_cfg = _ensure_heatmap_cfg(heatmap_cfg)
    inherit_missing = getattr(config, "inherit_missing_levels", False)
    missing_tag = getattr(full_dataset, "MISSING_TAG", "__missing__")
    if inherit_missing:
        global_class_names = _effective_label_names(full_dataset, analysis_level, missing_tag)
        global_num_classes = len(global_class_names)
        global_name_to_idx = {n: i for i, n in enumerate(global_class_names)}
    else:
        global_num_classes = full_dataset.num_classes_by_level[analysis_level]
        global_class_names = full_dataset.class_names_by_level[head_index]
        global_name_to_idx = None

    # ---------------- 输出目录 ----------------
    analysis_dir = os.path.join(exp_dir, f"{analysis_level}_aggregate_analysis")
    fig_dir = os.path.join(analysis_dir, "figures")
    log_dir = os.path.join(analysis_dir, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "analysis_log.txt")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + os.linesep)

    log(f"Aggregate analysis for {analysis_level} over {len(tasks)} parents.")

    # ---------------- 模型与统计缓存 ----------------
    channel_names = [f"{config.norm_method}"]
    if config.smooth_use:
        channel_names.append("smooth")
    if getattr(config, "raw_use", False):
        channel_names.append("raw")
    if config.d1_use:
        channel_names.append("d1")

    agg_channel = None
    agg_layer = None
    weight_total = 0
    layer_names = None

    band_total = None
    band_counts = np.zeros(global_num_classes, dtype=np.int64)
    mean_total = None
    mean_counts = np.zeros(global_num_classes, dtype=np.int64)

    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"Using device: {device} (config.use_gpu={config.use_gpu}, cuda_available={torch.cuda.is_available()})")

    for task in tasks:
        parent_idx = task["parent_idx"]
        log(f"--- Parent {parent_idx} ---")

        train_loader, test_loader, train_subset, _ = _build_task_loaders(
            task,
            config,
            full_dataset,
            analysis_level,
            head_index,
            train_idx_all,
            test_idx_all,
            base_train_dataset,
            base_test_dataset,
        )
        if len(train_subset) == 0:
            log(f"Skip parent {parent_idx}: no samples after filtering.")
            continue

        model = RamanClassifier1D(
            num_classes=task["num_classes"],
            config=config
        ).to(device)
        state = torch.load(task["model_path"], map_location=device)
        model.load_state_dict(state)
        model.eval()

        # warmup forward
        sample_x, _, _ = next(iter(train_loader))
        sample_x = sample_x.to(device)
        _ = model(sample_x)

        # ----- channel importance -----
        ch_imp = compute_input_channel_importance_IG(
            model,
            train_loader,
            device,
            channel_names=channel_names,
            save_dir=None,
            steps=64,
            num_batches=10,
            baseline_mode="mean",
            target_mode="true",
            head_index=head_index
        )
        weight = len(train_subset)
        if agg_channel is None:
            agg_channel = ch_imp * weight
        else:
            agg_channel += ch_imp * weight

        # ----- layer importance -----
        analyzable, groups = collect_analyzable_layers(model)
        analyzer = LayerGradCAMAnalyzer(model, device)
        for name, layer in analyzable.items():
            analyzer.register_layer(name, layer)
        layer_scores = analyzer.run(
            train_loader,
            save_dir=None,
            num_batches=3,
            target_mode="true",
            head_index=head_index
        )
        merged_scores = merge_scores_by_group(layer_scores, groups)
        if layer_names is None:
            layer_names = list(merged_scores.keys())
            agg_layer = {k: 0.0 for k in layer_names}
        for k in layer_names:
            agg_layer[k] += merged_scores.get(k, 0.0) * weight

        weight_total += weight

        # ----- band importance & mean spectrum -----
        heatmap_loader = train_loader if heatmap_cfg.use_train_loader else test_loader
        band_importance, counts = compute_class_band_importance_ig(
            model,
            heatmap_loader,
            device,
            num_classes=global_num_classes if inherit_missing else task["num_classes"],
            head_index=head_index,
            steps=heatmap_cfg.steps,
            num_batches=heatmap_cfg.num_batches,
            baseline_mode="mean",
            target_mode=heatmap_cfg.target_mode,
            max_per_class=heatmap_cfg.max_per_class,
            level_name=analysis_level if inherit_missing else None,
            label_name_to_idx=global_name_to_idx,
            missing_tag=missing_tag,
        )
        mean_spectra, mean_ct = compute_class_mean_spectrum(
            heatmap_loader,
            device,
            num_classes=global_num_classes if inherit_missing else task["num_classes"],
            head_index=head_index,
            max_per_class=heatmap_cfg.max_per_class,
            level_name=analysis_level if inherit_missing else None,
            label_name_to_idx=global_name_to_idx,
            missing_tag=missing_tag,
        )

        if band_total is None:
            band_total = np.zeros((global_num_classes, band_importance.shape[1]), dtype=np.float32)
            mean_total = np.zeros((global_num_classes, mean_spectra.shape[1]), dtype=np.float32)

        if inherit_missing:
            for idx in range(global_num_classes):
                c = int(counts[idx])
                if c > 0:
                    band_total[idx] += band_importance[idx] * c
                    band_counts[idx] += c
                mc = int(mean_ct[idx])
                if mc > 0:
                    mean_total[idx] += mean_spectra[idx] * mc
                    mean_counts[idx] += mc
        else:
            child_ids = task["child_ids"]
            for local_idx, global_idx in enumerate(child_ids):
                c = int(counts[local_idx])
                if c > 0:
                    band_total[global_idx] += band_importance[local_idx] * c
                    band_counts[global_idx] += c
                mc = int(mean_ct[local_idx])
                if mc > 0:
                    mean_total[global_idx] += mean_spectra[local_idx] * mc
                    mean_counts[global_idx] += mc

    # 单子类 parent：用上一级模型的重要性补齐
    parent_level = full_dataset.get_parent_level(analysis_level)
    if (not inherit_missing) and parent_level is not None and band_total is not None:
        parent_to_children = full_dataset.parent_to_children.get(analysis_level, {})
        single_child_map = {
            int(p): int(children[0])
            for p, children in parent_to_children.items()
            if len(children) == 1
        }

        if single_child_map:
            meta = _load_hierarchy_meta(exp_dir) or {}
            level_models = meta.get("level_models", {})
            parent_model_name = level_models.get(parent_level, f"{parent_level}_model.pt")
            parent_model_path = os.path.join(exp_dir, parent_model_name)

            if os.path.exists(parent_model_path):
                parent_head_index = full_dataset.head_name_to_idx[parent_level]
                num_parent_classes = full_dataset.num_classes_by_level[parent_level]

                dummy_task = {"parent_idx": None, "child_ids": None}
                parent_train_loader, parent_test_loader, _, _ = _build_task_loaders(
                    dummy_task,
                    config,
                    full_dataset,
                    parent_level,
                    parent_head_index,
                    train_idx_all,
                    test_idx_all,
                    base_train_dataset,
                    base_test_dataset,
                )
                parent_loader = (
                    parent_train_loader if heatmap_cfg.use_train_loader else parent_test_loader
                )

                parent_model = RamanClassifier1D(
                    num_classes=num_parent_classes,
                    config=config
                ).to(device)
                state = torch.load(parent_model_path, map_location=device)
                parent_model.load_state_dict(state)
                parent_model.eval()

                parent_importance, parent_counts = compute_class_band_importance_ig(
                    parent_model,
                    parent_loader,
                    device,
                    num_classes=num_parent_classes,
                    head_index=parent_head_index,
                    steps=heatmap_cfg.steps,
                    num_batches=heatmap_cfg.num_batches,
                    baseline_mode="mean",
                    target_mode=heatmap_cfg.target_mode,
                    max_per_class=heatmap_cfg.max_per_class,
                )
                # parent level 的平均光谱（用于单子类 child 继承）
                parent_mean_spectra, parent_mean_ct = compute_class_mean_spectrum(
                    parent_loader,
                    device,
                    num_classes=num_parent_classes,
                    head_index=parent_head_index,
                    max_per_class=heatmap_cfg.max_per_class,
                )

                for p_idx, child_idx in single_child_map.items():
                    if child_idx < 0 or child_idx >= global_num_classes:
                        continue
                    if band_counts[child_idx] > 0:
                        continue
                    c = int(parent_counts[p_idx])
                    if c <= 0:
                        continue
                    # -------- band importance：继承 parent --------
                    band_total[child_idx] += parent_importance[p_idx] * c
                    band_counts[child_idx] += c

                    # -------- mean spectrum：继承 parent --------
                    mc = int(parent_mean_ct[p_idx])
                    if mc > 0:
                        mean_total[child_idx] += parent_mean_spectra[p_idx] * mc
                        mean_counts[child_idx] += mc
            else:
                log(f"Parent model not found for {parent_level}: {parent_model_path}")

    if inherit_missing and mean_total is not None:
        missing_mask = mean_counts == 0
        if missing_mask.any():
            full_train_subset = Subset(base_train_dataset, train_idx_all)
            full_test_subset = Subset(base_test_dataset, test_idx_all)
            full_loader = DataLoader(
                full_train_subset if heatmap_cfg.use_train_loader else full_test_subset,
                batch_size=config.batch_size,
                shuffle=False
            )
            global_mean, global_counts = compute_class_mean_spectrum(
                full_loader,
                device,
                num_classes=global_num_classes,
                head_index=head_index,
                max_per_class=heatmap_cfg.max_per_class,
                level_name=analysis_level,
                label_name_to_idx=global_name_to_idx,
                missing_tag=missing_tag,
            )
            for i in np.where(missing_mask)[0]:
                c = int(global_counts[i])
                if c <= 0:
                    continue
                mean_total[i] = global_mean[i] * c
                mean_counts[i] = c

        # Fill missing band importance by inheriting from parent-level per-parent models
        if band_total is not None:
            missing_band = band_counts == 0
            if missing_band.any():
                parent_level = full_dataset.get_parent_level(analysis_level)
                if parent_level is None:
                    log("Parent level not found; cannot inherit band importance.")
                else:
                    meta = _load_hierarchy_meta(exp_dir) or {}
                    parent_models = meta.get("parent_models", {}).get(parent_level, {})
                    if not parent_models:
                        log(f"No parent models for {parent_level}; cannot inherit band importance.")
                    else:
                        parent_level_idx = full_dataset.head_name_to_idx[parent_level]
                        parent_level_map = full_dataset.label_maps_by_level[parent_level_idx]
                        grand_parent = full_dataset.get_parent_level(parent_level)
                        if grand_parent is None:
                            log(f"{parent_level} has no parent; cannot select per-parent models.")
                        else:
                            grand_parent_idx = full_dataset.head_name_to_idx[grand_parent]
                            grand_parent_map = full_dataset.label_maps_by_level[grand_parent_idx]

                            leaf_to_parent_idx = {}
                            leaf_to_parent_level_id = {}
                            for hier in full_dataset.hier_names:
                                leaf_name = hier.get("leaf")
                                parent_name = hier.get(parent_level)
                                gp_name = hier.get(grand_parent)
                                if leaf_name is None or parent_name is None or gp_name is None:
                                    continue
                                if leaf_name not in leaf_to_parent_idx:
                                    leaf_to_parent_idx[leaf_name] = grand_parent_map.get(gp_name)
                                    leaf_to_parent_level_id[leaf_name] = parent_level_map.get(parent_name)

                            parent_importance_cache = {}

                            for p_idx, entry in parent_models.items():
                                model_path = entry.get("model_path")
                                child_ids = entry.get("child_ids", [])
                                if model_path is None or not child_ids:
                                    continue
                                if not os.path.isabs(model_path):
                                    model_path = os.path.join(exp_dir, model_path)
                                if not os.path.exists(model_path):
                                    log(f"Parent model missing: {model_path}")
                                    continue

                                task = {"parent_idx": int(p_idx), "child_ids": child_ids}
                                parent_train_loader, parent_test_loader, _, _ = _build_task_loaders(
                                    task,
                                    config,
                                    full_dataset,
                                    parent_level,
                                    parent_level_idx,
                                    train_idx_all,
                                    test_idx_all,
                                    base_train_dataset,
                                    base_test_dataset,
                                )
                                parent_loader = (
                                    parent_train_loader if heatmap_cfg.use_train_loader else parent_test_loader
                                )

                                parent_model = RamanClassifier1D(
                                    num_classes=len(child_ids),
                                    config=config
                                ).to(device)
                                state = torch.load(model_path, map_location=device)
                                parent_model.load_state_dict(state)
                                parent_model.eval()

                                parent_importance, _ = compute_class_band_importance_ig(
                                    parent_model,
                                    parent_loader,
                                    device,
                                    num_classes=len(child_ids),
                                    head_index=parent_level_idx,
                                    steps=heatmap_cfg.steps,
                                    num_batches=heatmap_cfg.num_batches,
                                    baseline_mode="mean",
                                    target_mode=heatmap_cfg.target_mode,
                                    max_per_class=heatmap_cfg.max_per_class,
                                )
                                child_to_local = {int(cid): i for i, cid in enumerate(child_ids)}
                                parent_importance_cache[int(p_idx)] = (parent_importance, child_to_local)

                            for i in np.where(missing_band)[0]:
                                class_name = global_class_names[i]
                                p_idx = leaf_to_parent_idx.get(class_name)
                                parent_level_id = leaf_to_parent_level_id.get(class_name)
                                if p_idx is None or parent_level_id is None:
                                    continue
                                cached = parent_importance_cache.get(int(p_idx))
                                if cached is None:
                                    continue
                                parent_importance, child_to_local = cached
                                local_idx = child_to_local.get(int(parent_level_id))
                                if local_idx is None:
                                    continue
                                c = int(max(mean_counts[i], 1))
                                band_total[i] = parent_importance[local_idx] * c
                                band_counts[i] = c

    if weight_total == 0:
        log("No samples available for aggregate analysis.")
        log_file.close()
        return

    # ----- finalize aggregate channel/layer -----
    agg_channel = agg_channel / float(weight_total)
    agg_channel = agg_channel / (agg_channel.sum() + 1e-8)
    _plot_channel_importance(
        agg_channel,
        channel_names,
        os.path.join(fig_dir, "channel_importance_IG_aggregate.png")
    )
    log(f"Saved aggregate channel importance: {os.path.join(fig_dir, 'channel_importance_IG_aggregate.png')}")

    agg_layer = {k: v / float(weight_total) for k, v in agg_layer.items()}
    layer_total = sum(agg_layer.values())
    if layer_total > 0:
        agg_layer = {k: v / layer_total for k, v in agg_layer.items()}
    _plot_layer_importance(
        agg_layer,
        os.path.join(fig_dir, "layer_importance_aggregate.png")
    )
    log(f"Saved aggregate layer importance: {os.path.join(fig_dir, 'layer_importance_aggregate.png')}")

    # ----- finalize aggregate band importance -----
    valid_counts = np.maximum(band_counts[:, None], 1)
    band_avg = band_total / valid_counts
    mean_avg = mean_total / np.maximum(mean_counts[:, None], 1)

    wavenumbers = build_wavenumber_axis(band_avg.shape[1], config)
    bad_bands = _get_bad_bands(config)
    heatmap_path = os.path.join(fig_dir, "band_importance_heatmap_aggregate.png")
    plot_band_importance_heatmap(
        band_avg,
        band_counts,
        global_class_names,
        wavenumbers,
        heatmap_path,
        row_norm=heatmap_cfg.row_norm,
        mean_spectra=mean_avg,
        bad_bands=bad_bands,
    )
    log(f"Saved aggregate band importance heatmap: {heatmap_path}")

    topk_path = os.path.join(
        fig_dir, f"band_top{heatmap_cfg.topk_per_class}_per_class_aggregate.csv"
    )
    save_topk_bands_csv(
        band_avg,
        global_class_names,
        wavenumbers,
        top_k=heatmap_cfg.topk_per_class,
        save_path=topk_path,
        row_norm=heatmap_cfg.row_norm,
    )
    log(f"Saved aggregate band top-k CSV: {topk_path}")

    log("Note: Embedding plots are skipped in aggregate mode (different parent models).")
    log_file.close()


def run_single_analysis(
    exp_dir,
    config,
    full_dataset,
    analysis_level,
    head_index,
    task,
    train_idx_all,
    test_idx_all,
    base_train_dataset,
    base_test_dataset,
    heatmap_cfg=None,
):
    """单模型分析：Grad-CAM / IG / embedding / 波段热图。"""
    heatmap_cfg = _ensure_heatmap_cfg(heatmap_cfg)
    parent_idx = task["parent_idx"]
    num_classes = task["num_classes"]
    class_names = task["class_names"]
    model_path = task["model_path"]
    tag = task["tag"]

    # ---------------- 输出目录 ----------------
    analysis_dir = os.path.join(exp_dir, f"{tag}_analysis")
    fig_dir = os.path.join(analysis_dir, "figures")
    log_dir = os.path.join(analysis_dir, "logs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "analysis_log.txt")
    log_file = open(log_path, "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + os.linesep)

    if parent_idx is None:
        log(f"Analysis target: {analysis_level}")
    else:
        log(f"Analysis target: {analysis_level} (parent={parent_idx})")

    train_loader, test_loader, _, _ = _build_task_loaders(
        task,
        config,
        full_dataset,
        analysis_level,
        head_index,
        train_idx_all,
        test_idx_all,
        base_train_dataset,
        base_test_dataset,
    )

    # ---------------- 模型 ----------------
    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    log(
        f"Using device: {device} (config.use_gpu={config.use_gpu}, "
        f"cuda_available={torch.cuda.is_available()})"
    )

    model = RamanClassifier1D(
        num_classes=num_classes,
        config=config
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ------------------------------------------------------------
    # 执行一次前向传播：
    # - 初始化模型内部状态
    # - 确保后续 Grad-CAM / hook 正常注册与触发
    # ------------------------------------------------------------
    sample_x, _, _ = next(iter(train_loader))
    sample_x = sample_x.to(device)
    _ = model(sample_x)

    # 使用 Integrated Gradients 评估不同输入通道的相对贡献
    log("")
    log("=== Computing input channel importance ===")
    channel_names = [f"{config.norm_method}"]
    if config.smooth_use:
        channel_names.append("smooth")
    if getattr(config, "raw_use", False):
        channel_names.append("raw")
    if config.d1_use:
        channel_names.append("d1")

    log(f"Using channel names: {channel_names}")

    importance = compute_input_channel_importance_IG(
        model,
        train_loader,
        device,
        channel_names=channel_names,
        save_dir=fig_dir,
        steps=64,
        num_batches=10,
        baseline_mode="mean",
        target_mode="true",
        head_index=head_index
    )

    log(f"Input channel importance: {importance}")

    # 多层 Grad-CAM 级联 Layer-wise Importance 分析
    log("")
    log("=== Running Multi-layer Grad-CAM Analysis ===")
    analyzable, groups = collect_analyzable_layers(model)
    analyzer = LayerGradCAMAnalyzer(model, device)

    for name, layer in analyzable.items():
        analyzer.register_layer(name, layer)

    layer_scores = analyzer.run(
        train_loader,
        save_dir=fig_dir,
        num_batches=3,
        target_mode="true",
        head_index=head_index
    )

    merged_scores = merge_scores_by_group(layer_scores, groups)

    log("")
    log("=== Layer Importance (merged by stage) ===")
    for k, v in merged_scores.items():
        log(f"{k:30s}: {v:.4f}")

    # SE 模块分析
    if config.se_use:
        log("")
        log("===== SE Module Summary (Compact) =====")
        for name, module in model.named_modules():
            if isinstance(module, SEBlock1D):
                if module.latest_scale is None:
                    log(f"{name}: scale not computed")
                    continue

                s = module.latest_scale.mean(dim=0).detach().cpu().numpy()
                log(
                    f"{name}: "
                    f"mean={s.mean():.4f}, "
                    f"std={s.std():.4f}, "
                    f"min={s.min():.4f}, "
                    f"max={s.max():.4f}"
                )

    # Embedding 可视化
    embed_method = str(config.embedding_method).lower()
    embed_tag = embed_method.replace("-", "").replace("_", "")

    embed_levels = [analysis_level]
    if full_dataset.head_names:
        top_level = full_dataset.head_names[0]
        if top_level not in embed_levels:
            embed_levels.append(top_level)

    inherit_missing = getattr(config, "inherit_missing_levels", False)
    if inherit_missing:
        feats, hier_labels, label_names = collect_embeddings(
            model,
            test_loader,
            device,
            dataset=full_dataset,
            level_names=embed_levels,
            return_label_names=True,
        )
    else:
        feats, hier_labels = collect_embeddings(
            model,
            test_loader,
            device,
            dataset=full_dataset,
            level_names=embed_levels,
        )
        label_names = None
    plot_embedding_hierarchical(
        feats,
        hier_labels=hier_labels,
        save_path=os.path.join(fig_dir, f"{embed_tag}_hier.png"),
        method=embed_method,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        tsne_perplexity=config.tsne_perplexity,
        tsne_iter=config.tsne_iter,
        label_names=label_names,
    )

    if inherit_missing:
        feats, hier_labels, label_names = collect_embeddings_train_test(
            model,
            train_loader,
            test_loader,
            device,
            dataset=full_dataset,
            level_names=embed_levels,
            return_label_names=True,
        )
    else:
        feats, hier_labels = collect_embeddings_train_test(
            model,
            train_loader,
            test_loader,
            device,
            dataset=full_dataset,
            level_names=embed_levels,
        )
        label_names = None

    plot_embedding_hierarchical(
        feats,
        hier_labels=hier_labels,
        save_path=os.path.join(fig_dir, f"{embed_tag}_hier_train_test.png"),
        method=embed_method,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        tsne_perplexity=config.tsne_perplexity,
        tsne_iter=config.tsne_iter,
        label_names=label_names,
    )

    # 波段重要性热图（按类别）
    log("")
    log("=== Computing band importance heatmap ===")
    heatmap_loader = train_loader if heatmap_cfg.use_train_loader else test_loader

    if inherit_missing:
        missing_tag = getattr(full_dataset, "MISSING_TAG", "__missing__")
        heatmap_class_names = _effective_label_names(full_dataset, analysis_level, missing_tag)
        heatmap_name_to_idx = {n: i for i, n in enumerate(heatmap_class_names)}
        heatmap_num_classes = len(heatmap_class_names)
    else:
        missing_tag = getattr(full_dataset, "MISSING_TAG", "__missing__")
        heatmap_class_names = class_names
        heatmap_name_to_idx = None
        heatmap_num_classes = num_classes

    band_importance, counts = compute_class_band_importance_ig(
        model,
        heatmap_loader,
        device,
        num_classes=heatmap_num_classes,
        head_index=head_index,
        steps=heatmap_cfg.steps,
        num_batches=heatmap_cfg.num_batches,
        baseline_mode="mean",
        target_mode=heatmap_cfg.target_mode,
        max_per_class=heatmap_cfg.max_per_class,
        level_name=analysis_level if inherit_missing else None,
        label_name_to_idx=heatmap_name_to_idx,
        missing_tag=missing_tag,
    )

    mean_spectra, _ = compute_class_mean_spectrum(
        heatmap_loader,
        device,
        num_classes=heatmap_num_classes,
        head_index=head_index,
        max_per_class=heatmap_cfg.max_per_class,
        level_name=analysis_level if inherit_missing else None,
        label_name_to_idx=heatmap_name_to_idx,
        missing_tag=missing_tag,
    )

    wavenumbers = build_wavenumber_axis(band_importance.shape[1], config)
    bad_bands = _get_bad_bands(config)
    heatmap_path = os.path.join(fig_dir, "band_importance_heatmap.png")
    plot_band_importance_heatmap(
        band_importance,
        counts,
        heatmap_class_names,
        wavenumbers,
        heatmap_path,
        row_norm=heatmap_cfg.row_norm,
        mean_spectra=mean_spectra,
        bad_bands=bad_bands,
    )
    log(f"Saved band importance heatmap: {heatmap_path}")

    topk_path = os.path.join(
        fig_dir, f"band_top{heatmap_cfg.topk_per_class}_per_class.csv"
    )
    save_topk_bands_csv(
        band_importance,
        heatmap_class_names,
        wavenumbers,
        top_k=heatmap_cfg.topk_per_class,
        save_path=topk_path,
        row_norm=heatmap_cfg.row_norm,
    )
    log(f"Saved band top-k CSV: {topk_path}")

    log_file.close()


def build_analysis_context(
    exp_dir,
    analysis_level,
    parent_idx,
    use_train_aug,
    inherit_missing_levels=False,
):
    """构建通用上下文（数据集 / 划分 / 任务列表）。"""
    exp_dir = _resolve_project_path(exp_dir)
    config = load_experiment(exp_dir)

    dataset_root = os.fspath(
        resolve_dataset_stage(
            _resolve_project_path(config.dataset_root),
            stage="train",
            project_root=os.fspath(PROJECT_ROOT),
            must_exist=True,
        )
    )
    config.dataset_root = dataset_root

    config.inherit_missing_levels = bool(inherit_missing_levels)

    full_dataset = RamanDataset(
        dataset_root,
        augment=False,
        config=config
    )

    analysis_level = resolve_analysis_level(full_dataset, analysis_level, config)
    head_index = full_dataset.head_name_to_idx[analysis_level]

    meta = _load_hierarchy_meta(exp_dir) or {}
    level_models = meta.get("level_models", {})
    parent_models = meta.get("parent_models", {})

    tasks, auto_all = _build_analysis_tasks(
        exp_dir,
        analysis_level,
        head_index,
        full_dataset,
        level_models,
        parent_models,
        parent_idx,
    )

    split = load_split_files(full_dataset, exp_dir)
    if split is not None:
        train_idx, test_idx = split
    else:
        split_level = getattr(config, "split_level", None) or "leaf"
        train_idx, test_idx = split_by_lowest_level_ratio(
            full_dataset,
            lowest_level=split_level,
            train_ratio=config.train_split,
            seed=config.seed
        )
    train_idx_all = np.array(sorted(train_idx))
    test_idx_all = np.array(sorted(test_idx))

    base_train_dataset = RamanDataset(
        dataset_root,
        augment=use_train_aug,
        config=config
    )
    base_test_dataset = RamanDataset(
        dataset_root,
        augment=False,
        config=config
    )

    return {
        "exp_dir": exp_dir,
        "config": config,
        "full_dataset": full_dataset,
        "analysis_level": analysis_level,
        "head_index": head_index,
        "tasks": tasks,
        "train_idx_all": train_idx_all,
        "test_idx_all": test_idx_all,
        "base_train_dataset": base_train_dataset,
        "base_test_dataset": base_test_dataset,
        "auto_all": auto_all,
    }


def _run_single_tasks(ctx, heatmap_cfg=None):
    """执行单模型分析任务列表。"""
    if ctx["auto_all"]:
        print(
            f"No global model for {ctx['analysis_level']}; "
            f"running per-parent analysis for {len(ctx['tasks'])} parents."
        )
    elif len(ctx["tasks"]) > 1:
        print(f"Running per-parent analysis for {len(ctx['tasks'])} parents.")

    for task in ctx["tasks"]:
        run_single_analysis(
            ctx["exp_dir"],
            ctx["config"],
            ctx["full_dataset"],
            ctx["analysis_level"],
            ctx["head_index"],
            task,
            ctx["train_idx_all"],
            ctx["test_idx_all"],
            ctx["base_train_dataset"],
            ctx["base_test_dataset"],
            heatmap_cfg=heatmap_cfg,
        )


def run_analysis_pipeline(overrides=None, heatmap_cfg=None):
    """统一分析入口：按 mode 切换单模型或聚合分析。"""
    overrides = overrides or AnalysisOverrides()
    if not overrides.exp_dir:
        raise ValueError("analyze 需要显式传入 exp_dir。")

    mode = str(overrides.mode).lower()
    if mode not in ("single", "aggregate"):
        raise ValueError(f"未知分析模式：{overrides.mode}，可选值为：single / aggregate")

    ctx = build_analysis_context(
        overrides.exp_dir,
        overrides.analysis_level,
        overrides.parent_idx,
        overrides.use_train_aug,
        inherit_missing_levels=overrides.inherit_missing_levels,
    )

    if mode == "single":
        _run_single_tasks(ctx, heatmap_cfg=heatmap_cfg)
        return

    parent_tasks = [t for t in ctx["tasks"] if t["parent_idx"] is not None]
    if not parent_tasks:
        if overrides.fallback_to_single:
            print("Aggregate fallback: no parent models found; running single-model analysis.")
            _run_single_tasks(ctx, heatmap_cfg=heatmap_cfg)
        else:
            print("Aggregate analysis skipped: no parent models found.")
        return

    run_aggregate_analysis(
        ctx["exp_dir"],
        ctx["config"],
        ctx["full_dataset"],
        ctx["analysis_level"],
        ctx["head_index"],
        parent_tasks,
        ctx["train_idx_all"],
        ctx["test_idx_all"],
        ctx["base_train_dataset"],
        ctx["base_test_dataset"],
        heatmap_cfg=heatmap_cfg,
    )
