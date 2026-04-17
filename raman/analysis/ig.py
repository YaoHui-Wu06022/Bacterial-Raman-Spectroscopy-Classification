import os
import csv

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def _ensure_dir(d):
    """保存结果前确保目录存在。"""
    if d is None:
        return
    os.makedirs(d, exist_ok=True)

def _to_numpy(x):
    """将张量或数组统一转成 numpy，便于绘图和保存。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _needs_cudnn_rnn_guard(model):
    """评估态下若模型含 RNN 系列模块，则临时关闭 cuDNN 以避免反传限制。"""
    if model.training:
        return False
    if not torch.backends.cudnn.enabled:
        return False
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            return True
    return False

def _select_logits(pred, head_name=None):
    """兼容多种前向返回格式，只抽取当前分析所需的 logits。"""
    if isinstance(pred, dict):
        if head_name is None:
            head_name = list(pred.keys())[-1]
        return pred[head_name]
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred

def _compute_baseline_mean_spectrum(loader, device, num_batches=10):
    """
    用前 num_batches 个批次的平均光谱作为积分起点，更接近真实数据分布。
    返回形状为 [1, C, L] 的张量，与输入 x 保持相同通道数和长度。
    """
    means = []
    it = iter(loader)
    for _ in range(num_batches):
        try:
            x, _ , _= next(it)
        except StopIteration:
            break
        x = x.to(device)
        means.append(x.mean(dim=0, keepdim=True))  # 单个批次的平均光谱，形状 [1, C, L]
    if len(means) == 0:
        raise RuntimeError("Loader is empty; cannot compute baseline.")
    return torch.mean(torch.stack(means, dim=0), dim=0)

def compute_input_channel_importance_IG(
    model,
    loader,
    device,
    channel_names,
    save_dir=None,
    steps=64,
    num_batches=5,
    head_name=None,
    head_index=None,
):
    """
    输出“输入通道重要性”，即对每个输入通道的归因值做平均。
    - num_batches:
        取前 num_batches 个批次做平均，比只看单个批次稳定得多
    """
    _ensure_dir(save_dir)
    ig_batches = compute_ig_batches(
        model,
        loader,
        device,
        steps=steps,
        num_batches=num_batches,
        head_name=head_name,
        head_index=head_index,
    )
    channel_importance = compute_channel_importance_from_ig(ig_batches)

    if save_dir is not None:
        plt.figure(figsize=(7, 4))
        xnames = channel_names[: len(channel_importance)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(xnames)))
        plt.bar(xnames, _to_numpy(channel_importance), color=colors)
        plt.title("Input Channel Contribution (Integrated Gradients)")
        plt.ylabel("Relative Importance")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "channel_importance_IG.png"), dpi=300)
        plt.show()
        plt.close()

    return channel_importance

def _effective_label_names(dataset, level_name, missing_tag="__missing__"):
    """按数据集中实际出现顺序收集某一层的有效类别名。"""
    if hasattr(dataset, "_resolve_level_name"):
        level_name = dataset._resolve_level_name(level_name)
    names = []
    seen = set()
    for hier in dataset.hier_names:
        name = hier.get(level_name)
        if name is None:
            name = missing_tag
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names

def _batch_effective_label_ids(hier, level_name, name_to_idx, missing_tag="__missing__"):
    """把当前批次的层级名映射成连续类别 id，缺失项映射为 -1。"""
    if hier is None:
        return None
    level_vals = hier.get(level_name, [])
    ids = []
    for v in level_vals:
        if v is None:
            v = missing_tag
        ids.append(name_to_idx.get(v, -1))
    return np.asarray(ids, dtype=np.int64)

def compute_ig_batches(
    model,
    loader,
    device,
    steps=32,
    num_batches=6,
    head_name=None,
    head_index=None,
    num_classes=None,
    max_per_class=None,
    level_name=None,
    label_name_to_idx=None,
    missing_tag="__missing__",
):
    """
    统一计算一轮 IG，返回每个批次的原始归因结果。
    后续可以基于这些结果分别汇总通道重要性和类别波段重要性。
    """
    model.eval()
    disable_cudnn = _needs_cudnn_rnn_guard(model)
    prev_cudnn = torch.backends.cudnn.enabled
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    baseline = _compute_baseline_mean_spectrum(loader, device, num_batches=num_batches)

    ig_batches = []

    it = iter(loader)
    for _ in range(num_batches):
        try:
            x, y, hier = next(it)
        except StopIteration:
            break

        x = x.to(device)
        y = y.to(device)

        if y.ndim == 2:
            current_head_index = y.size(1) - 1 if head_index is None else head_index
            y_head = y[:, current_head_index]
        else:
            y_head = y

        target_valid_mask = y_head >= 0

        if num_classes is not None and label_name_to_idx is not None and level_name is not None:
            group_ids = _batch_effective_label_ids(hier, level_name, label_name_to_idx, missing_tag)
            if group_ids is None:
                continue
            group_ids_t = torch.from_numpy(group_ids).to(y_head.device)
            stat_mask = group_ids_t >= 0
        else:
            group_ids_t = y_head
            stat_mask = target_valid_mask

        ig_mask = stat_mask if num_classes is not None else target_valid_mask
        if not ig_mask.any():
            continue

        b = baseline.expand(x.size(0), -1, -1)

        with torch.no_grad():
            logits0 = _select_logits(model(x), head_name=head_name)
            pred = logits0.argmax(dim=1)
            target = torch.where(target_valid_mask, y_head, pred)

        safe_target = target.clone()
        safe_target[~ig_mask] = 0

        total_grad = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, steps, device=device)

        for alpha in alphas:
            x_step = (b + alpha * (x - b)).detach().requires_grad_(True)
            logits = _select_logits(model(x_step), head_name=head_name)
            score_each = logits.gather(1, safe_target.view(-1, 1)).squeeze(1)
            score = (score_each * ig_mask.float()).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            total_grad += x_step.grad.detach()

        avg_grad = total_grad / float(steps)
        ig = (x - b) * avg_grad

        ig_batches.append(
            {
                "ig": ig.detach().cpu().numpy().astype(np.float32, copy=False),
                "group_ids": group_ids_t.detach().cpu().numpy().astype(np.int64, copy=False),
                "valid_mask": ig_mask.detach().cpu().numpy().astype(bool, copy=False),
            }
        )

    if not ig_batches:
        if disable_cudnn:
            torch.backends.cudnn.enabled = prev_cudnn
        raise RuntimeError("No batch available to compute IG importance.")

    if disable_cudnn:
        torch.backends.cudnn.enabled = prev_cudnn
    return ig_batches

def compute_channel_importance_from_ig(ig_batches):
    """根据原始 IG 批次结果汇总输入通道重要性。"""
    total_channel = None
    total_weight = 0

    for batch in ig_batches:
        valid_mask = batch["valid_mask"]
        if valid_mask is None or not np.any(valid_mask):
            continue
        valid_ig = batch["ig"][valid_mask]
        if valid_ig.size == 0:
            continue
        channel_importance = np.abs(valid_ig).mean(axis=(0, 2))
        if total_channel is None:
            total_channel = channel_importance
        else:
            total_channel += channel_importance
        total_weight += 1

    if total_weight == 0 or total_channel is None:
        raise RuntimeError("No batch available to compute IG importance.")

    channel_importance = total_channel / float(total_weight)
    channel_importance = channel_importance / (channel_importance.sum() + 1e-8)
    return channel_importance.astype(np.float32, copy=False)

def compute_band_importance_from_ig(ig_batches, num_classes, max_per_class=None):
    """根据原始 IG 批次结果汇总每个类别的波段重要性。"""
    band_total = None
    band_counts = np.zeros(num_classes, dtype=np.int64)

    for batch in ig_batches:
        ig = batch["ig"]
        group_ids = batch["group_ids"]
        valid_mask = batch["valid_mask"]
        if valid_mask is None or not np.any(valid_mask):
            continue

        ig_band = np.abs(ig).mean(axis=1)
        if band_total is None:
            band_total = np.zeros((num_classes, ig_band.shape[1]), dtype=np.float32)

        for i in range(ig_band.shape[0]):
            if not valid_mask[i]:
                continue
            cls = int(group_ids[i])
            if cls < 0 or cls >= num_classes:
                continue
            if max_per_class is not None and band_counts[cls] >= max_per_class:
                continue
            band_total[cls] += ig_band[i]
            band_counts[cls] += 1

    if band_total is None:
        raise RuntimeError("No valid samples for band importance heatmap.")

    band_avg = band_total / np.maximum(band_counts[:, None], 1)
    return band_avg, band_counts

def compute_class_band_importance_ig(
    model,
    loader,
    device,
    num_classes,
    head_index,
    steps=32,
    num_batches=6,
    max_per_class=None,
    level_name=None,
    label_name_to_idx=None,
    missing_tag="__missing__",
):
    """计算每个类别在波数轴上的平均 IG 重要性。"""
    ig_batches = compute_ig_batches(
        model,
        loader,
        device,
        steps=steps,
        num_batches=num_batches,
        head_index=head_index,
        num_classes=num_classes,
        max_per_class=max_per_class,
        level_name=level_name,
        label_name_to_idx=label_name_to_idx,
        missing_tag=missing_tag,
    )
    avg, counts = compute_band_importance_from_ig(
        ig_batches,
        num_classes=num_classes,
        max_per_class=max_per_class,
    )
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
    """统计每个类别的平均光谱，默认只使用第 0 个输入通道。"""
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

        # 兼容单通道和多通道输入；多通道时只画主光谱通道
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
                # 先按预处理参数生成完整波数轴，再删掉坏段，尽量和实际输入长度严格对齐
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
    """按类别绘制热图：曲线表示平均光谱，曲线下填充颜色表示波段重要性。"""
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
        # 在波数坐标上用灰色阴影标记被剔除的坏段
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
                # 缺口处不做填充，避免曲线跨越坏段被错误连接
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
    """把每个类别最重要的前 k 个波段导出到 CSV。"""
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
    """绘制聚合模式下的输入通道重要性柱状图。"""
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
