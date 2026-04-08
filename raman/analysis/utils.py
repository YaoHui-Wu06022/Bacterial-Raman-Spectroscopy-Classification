import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import umap.umap_ as umap
from collections import OrderedDict
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import inspect
from sklearn.manifold import TSNE

def make_tsne_compatible(**kwargs):
    """
    根据 sklearn 当前版本自动选择 TSNE 支持的参数
    - 老版本：n_iter
    - 新版本：max_iter
    """
    sig = inspect.signature(TSNE.__init__)
    params = sig.parameters

    # 统一处理迭代次数参数
    if "max_iter" in params and "n_iter" in kwargs:
        kwargs["max_iter"] = kwargs.pop("n_iter")
    elif "n_iter" in params and "max_iter" in kwargs:
        kwargs["n_iter"] = kwargs.pop("max_iter")

    return TSNE(**kwargs)
# Utils
def _ensure_dir(d):
    if d is None:
        return
    os.makedirs(d, exist_ok=True)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _needs_cudnn_rnn_guard(model):
    if model.training:
        return False
    if not torch.backends.cudnn.enabled:
        return False
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            return True
    return False

def _reduce_importance(A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(A * G))

def _select_logits(pred, head_name=None):
    if isinstance(pred, dict):
        if head_name is None:
            head_name = list(pred.keys())[-1]
        return pred[head_name]
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred


# Integrated Gradients：输入通道重要性
@torch.no_grad()
def _compute_baseline_mean_spectrum(loader, device, num_batches=10):
    """
    用前 num_batches 的 batch 平均光谱作为 baseline（更接近“数据流形”）。
    返回: baseline tensor, shape [1, C, L]（与 x 同通道同长度）
    """
    means = []
    it = iter(loader)
    for _ in range(num_batches):
        try:
            x, _ , _= next(it)
        except StopIteration:
            break
        x = x.to(device)
        means.append(x.mean(dim=0, keepdim=True))  # [1, C, L]
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
    baseline_mode="mean",  # "zero" or "mean"
    target_mode="true",    # "true" or "pred"
    head_name=None,
    head_index=None,
):
    """
    输出“输入通道重要性”（对每个通道的 attribution 做平均）。
    - baseline_mode:
        "zero": baseline = 0（数学可行但物理弱）
        "mean": baseline = 数据均值谱（推荐，稳定且物理合理）
    - target_mode:
        "true": 用真实标签作为目标（常用于解释训练分布）
        "pred": 用模型预测类别作为目标（常用于解释推理行为）
    - num_batches:
        取前 num_batches 个 batch 做平均（比单 batch 稳定很多）
    """

    model.eval()
    _ensure_dir(save_dir)
    disable_cudnn = _needs_cudnn_rnn_guard(model)
    prev_cudnn = torch.backends.cudnn.enabled
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    # -------- baseline --------
    if baseline_mode == "zero":
        baseline = None  # 按 batch 生成 zeros_like
    elif baseline_mode == "mean":
        baseline = _compute_baseline_mean_spectrum(loader, device, num_batches=num_batches)
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    # -------- accumulate IG over multiple batches --------
    total_channel = None
    total_weight = 0

    it = iter(loader)
    used_batches = 0

    for _ in range(num_batches):
        try:
            x, y, _= next(it)
        except StopIteration:
            break

        used_batches += 1
        x = x.to(device)
        y = y.to(device)

        if y.ndim == 2:
            if head_index is None:
                head_index = y.size(1) - 1
            y = y[:, head_index]

        if baseline_mode == "zero":
            b = torch.zeros_like(x)
        else:
            # baseline: [1,C,L] -> [B,C,L]
            b = baseline.expand(x.size(0), -1, -1)

        # target class
        with torch.no_grad():
            logits0 = _select_logits(model(x), head_name=head_name)
            if target_mode == "pred":
                target = logits0.argmax(dim=1)
            else:
                target = y

        # IG integral
        total_grad = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, steps, device=device)

        for alpha in alphas:
            x_step = (b + alpha * (x - b)).detach().requires_grad_(True)

            logits = _select_logits(model(x_step), head_name=head_name)

            # 取目标类别 logit 的和（更标准的 attribution 目标）
            score = logits.gather(1, target.view(-1, 1)).sum()

            model.zero_grad(set_to_none=True)
            score.backward()
            total_grad += x_step.grad.detach()

        avg_grad = total_grad / float(steps)
        ig = (x - b) * avg_grad  # [B,C,L]

        # 通道重要性：对 batch 和序列维平均绝对值
        channel_importance = ig.abs().mean(dim=(0, 2))  # [C]

        # accumulate
        if total_channel is None:
            total_channel = channel_importance.detach()
        else:
            total_channel += channel_importance.detach()
        total_weight += 1

    if total_weight == 0:
        if disable_cudnn:
            torch.backends.cudnn.enabled = prev_cudnn
        raise RuntimeError("No batch available to compute IG importance.")

    channel_importance = total_channel / float(total_weight)
    channel_importance = channel_importance / (channel_importance.sum() + 1e-8)

    # -------- plot --------
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

    if disable_cudnn:
        torch.backends.cudnn.enabled = prev_cudnn
    return channel_importance


# Grad-CAM Layer-wise: 多层重要性
def collect_analyzable_layers(model):
    """
    自动收集：
    - conv1 / input_proj
    - 所有 ResidualBottleneck1D
    - 所有 TransformerEncoderLayer
    - LSTM
    返回:
        analyzable: { "layer1.0": module, ... }
        groups:     { "layer1.0": "layer1", "layer1.1": "layer1", ... }
    """
    from raman.model import ResidualBottleneck1D

    analyzable = {}
    groups = {}

    def recursive_find(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # --------------------------------------------------
            # ResidualBottleneck1D 归到它所在的 layer
            # --------------------------------------------------
            if isinstance(child, ResidualBottleneck1D):
                analyzable[full_name] = child
                groups[full_name] = full_name.split(".")[0]  # layer1.0 -> layer1
                continue

            # --------------------------------------------------
            # TransformerEncoderLayer 归为 "transformer"
            # --------------------------------------------------
            if isinstance(child, nn.TransformerEncoderLayer):
                analyzable[full_name] = child
                groups[full_name] = "transformer"
                continue

            # --------------------------------------------------
            # LSTM 归为 "lstm"
            # --------------------------------------------------
            if isinstance(child, nn.LSTM):
                analyzable[full_name] = child
                groups[full_name] = "lstm"
                continue

            # --------------------------------------------------
            # conv1 / input_proj 特例
            # --------------------------------------------------
            if full_name in ("conv1", "input_proj"):
                analyzable[full_name] = child
                groups[full_name] = full_name
                continue

            # 默认继续遍历
            recursive_find(child, full_name)

    recursive_find(model)
    return analyzable, groups

class LayerGradCAMAnalyzer:
    """
    - hook forward: 保存 activation
    - hook backward: 保存 gradient（用 register_full_backward_hook，避免 deprecated/backward_hook 问题）
    - importance: mean(|A*G|) over all dims
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.activations = {}
        self.gradients = {}
        self.hooks = []

    # 注册 forward/backward hook
    def register_layer(self, name, layer):
        # forward hook: save activation
        def f_hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.activations[name] = out.detach()

        # backward hook: save grad_out
        def b_hook(module, grad_in, grad_out):
            g = grad_out[0]
            self.gradients[name] = g.detach()

        self.hooks.append(layer.register_forward_hook(f_hook))
        self.hooks.append(layer.register_full_backward_hook(b_hook))

    # -------- release all hooks --------
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_importance(self):
        importance = OrderedDict()

        for name, A in self.activations.items():
            if name not in self.gradients:
                continue
            G = self.gradients[name]

            # A and G should be same shape
            score = _reduce_importance(A, G)
            importance[name] = float(score.item())

        # normalize
        total = sum(importance.values())
        for k in list(importance.keys()):
            importance[k] = importance[k] / (total + 1e-8)

        return importance

    def run(self, loader, save_dir=None, num_batches=3, target_mode="true", head_name=None, head_index=None):
        """
        从前 num_batches 个 batch 计算平均 layer importance（更稳定）。
        """
        _ensure_dir(save_dir)

        self.model.eval()
        disable_cudnn = _needs_cudnn_rnn_guard(self.model)
        prev_cudnn = torch.backends.cudnn.enabled
        if disable_cudnn:
            torch.backends.cudnn.enabled = False

        merged = None
        used = 0

        it = iter(loader)
        for _ in range(num_batches):
            try:
                x, y, _ = next(it)
            except StopIteration:
                break
            used += 1
            x, y = x.to(self.device), y.to(self.device)

            if y.ndim == 2:
                if head_index is None:
                    head_index = y.size(1) - 1
                y = y[:, head_index]

            logits = _select_logits(self.model(x), head_name=head_name)

            if target_mode == "pred":
                target = logits.argmax(dim=1)
            else:
                target = y

            # 用目标类别 logit 作为标量目标
            score = logits.gather(1, target.view(-1, 1)).sum()

            self.model.zero_grad(set_to_none=True)
            score.backward()

            scores = self.compute_importance()

            if merged is None:
                merged = OrderedDict(scores)
            else:
                for k, v in scores.items():
                    merged[k] = merged.get(k, 0.0) + v

            # 清空本轮缓存（避免跨 batch 混杂）
            self.activations.clear()
            self.gradients.clear()

        if used == 0:
            self.clear_hooks()
            if disable_cudnn:
                torch.backends.cudnn.enabled = prev_cudnn
            raise RuntimeError("No batch available to run LayerGradCAMAnalyzer.")

        # average and renormalize
        for k in list(merged.keys()):
            merged[k] /= float(used)
        total = sum(merged.values())
        for k in list(merged.keys()):
            merged[k] /= (total + 1e-8)

        if save_dir is not None:
            self.plot(merged, save_dir)

        self.clear_hooks()
        if disable_cudnn:
            torch.backends.cudnn.enabled = prev_cudnn
        return merged

    def plot(self, scores, save_dir):
        names = list(scores.keys())
        vals = list(scores.values())
        plt.figure(figsize=(10, 5))
        plt.bar(names, vals)
        plt.xticks(rotation=60)
        plt.ylabel("Layer Importance (Normalized |A × G|)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "layer_importance.png"), dpi=300)
        plt.close()

def merge_scores_by_group(layer_scores, groups):
    """
    将 layer_scores 合并到 stage 粒度（conv1 / layer1 / ... / transformer）
    """
    merged = {}
    for name, score in layer_scores.items():
        g = groups.get(name, name)
        merged[g] = merged.get(g, 0.0) + float(score)

    total = sum(merged.values())
    for k in list(merged.keys()):
        merged[k] /= (total + 1e-8)
    return merged


def _fill_missing_with_leaf(hier, head_names, missing_tag):
    leaf = hier.get("leaf", [])
    out = {}
    for name in head_names:
        vals = list(hier.get(name, []))
        if name == "leaf":
            out[name] = vals
            continue
        filled = []
        for i, v in enumerate(vals):
            if v is None or v == missing_tag:
                fill = leaf[i] if i < len(leaf) else v
                filled.append(fill)
            else:
                filled.append(v)
        out[name] = filled
    return out

def collect_embeddings(
    model,
    loader,
    device,
    dataset,
    level_names=None,
    return_label_names=False,
):
    model.eval()

    feats = []
    if level_names is None:
        level_names = [n for n in dataset.head_names if n != "leaf"]
        if not level_names:
            level_names = list(dataset.head_names)

    labels_by_level = {k: [] for k in level_names}
    label_names = {k: [] for k in level_names}
    name_to_id = {k: {} for k in level_names}
    inherit_missing = getattr(dataset.config, "inherit_missing_levels", False)
    missing_tag = getattr(dataset, "MISSING_TAG", "__missing__")

    with torch.no_grad():
        for x, _, hier in loader:
            x = x.to(device)

            if inherit_missing:
                hier_filled = _fill_missing_with_leaf(hier, dataset.head_names, missing_tag)
            else:
                hier_filled = None
                hier_labels = dataset.encode_hierarchy(hier, device="cpu")

            _, feat = model(x, return_feat=True)

            feats.append(feat.cpu().numpy())
            for k in level_names:
                if inherit_missing:
                    ids = []
                    for name in hier_filled.get(k, []):
                        if name is None or name == missing_tag:
                            name = missing_tag
                        mapping = name_to_id[k]
                        if name not in mapping:
                            mapping[name] = len(label_names[k])
                            label_names[k].append(name)
                        ids.append(mapping[name])
                    labels_by_level[k].append(np.asarray(ids, dtype=np.int64))
                else:
                    labels_by_level[k].append(hier_labels[k].numpy())

    feats = np.concatenate(feats, axis=0)
    labels_out = {k: np.concatenate(v, axis=0) for k, v in labels_by_level.items()}

    if return_label_names:
        return feats, labels_out, label_names
    return feats, labels_out

def collect_embeddings_train_test(
    model,
    train_loader,
    test_loader,
    device,
    dataset,
    level_names=None,
    return_label_names=False,
):
    model.eval()

    feats_all = []
    split_all = []

    if level_names is None:
        level_names = [n for n in dataset.head_names if n != "leaf"]
        if not level_names:
            level_names = list(dataset.head_names)

    labels_all = {k: [] for k in level_names}
    label_names = {k: [] for k in level_names}
    name_to_id = {k: {} for k in level_names}
    inherit_missing = getattr(dataset.config, "inherit_missing_levels", False)
    missing_tag = getattr(dataset, "MISSING_TAG", "__missing__")

    with torch.no_grad():
        # -------- train --------
        for x, _, hier in train_loader:
            x = x.to(device)
            if inherit_missing:
                hier_filled = _fill_missing_with_leaf(hier, dataset.head_names, missing_tag)
            else:
                hier_filled = None
                hier_labels = dataset.encode_hierarchy(hier, device="cpu")
            _, feat = model(x, return_feat=True)

            feats_all.append(feat.cpu().numpy())
            split_all.append(np.zeros(len(feat), dtype=np.int32))
            for k in level_names:
                if inherit_missing:
                    ids = []
                    for name in hier_filled.get(k, []):
                        if name is None or name == missing_tag:
                            name = missing_tag
                        mapping = name_to_id[k]
                        if name not in mapping:
                            mapping[name] = len(label_names[k])
                            label_names[k].append(name)
                        ids.append(mapping[name])
                    labels_all[k].append(np.asarray(ids, dtype=np.int64))
                else:
                    labels_all[k].append(hier_labels[k].numpy())

        # -------- test --------
        for x, _, hier in test_loader:
            x = x.to(device)
            if inherit_missing:
                hier_filled = _fill_missing_with_leaf(hier, dataset.head_names, missing_tag)
            else:
                hier_filled = None
                hier_labels = dataset.encode_hierarchy(hier, device="cpu")
            _, feat = model(x, return_feat=True)

            feats_all.append(feat.cpu().numpy())
            split_all.append(np.ones(len(feat), dtype=np.int32))
            for k in level_names:
                if inherit_missing:
                    ids = []
                    for name in hier_filled.get(k, []):
                        if name is None or name == missing_tag:
                            name = missing_tag
                        mapping = name_to_id[k]
                        if name not in mapping:
                            mapping[name] = len(label_names[k])
                            label_names[k].append(name)
                        ids.append(mapping[name])
                    labels_all[k].append(np.asarray(ids, dtype=np.int64))
                else:
                    labels_all[k].append(hier_labels[k].numpy())

    feats_all = np.concatenate(feats_all, axis=0)

    hier_labels_all = {
        k: np.concatenate(v, axis=0)
        for k, v in labels_all.items()
    }
    hier_labels_all["split"] = np.concatenate(split_all, axis=0)

    if return_label_names:
        return feats_all, hier_labels_all, label_names
    return feats_all, hier_labels_all

def plot_embedding_hierarchical(
    feats,
    hier_labels: dict,
    save_path,
    method="umap",
    n_neighbors=15,
    min_dist=0.1,
    tsne_perplexity=30,
    tsne_iter=1000,
    random_state=42,
    label_names=None,
):
    """
        自适应层级 embedding 可视化
        hier_labels:
            OrderedDict / dict
            e.g. {"level_1": level1_ids, "level_2": level2_ids}
            默认 dict 顺序 = 层级顺序（由粗到细）
        """
    # ===== 1. 准备 embedding =====
    n_samples = feats.shape[0]
    method = method.lower()

    if method == "umap":
        actual_neighbors = min(n_neighbors, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_neighbors=actual_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        emb_2d = reducer.fit_transform(feats)
        method_name = "UMAP"
    elif method == "tsne":
        if n_samples < 3:
            raise ValueError("t-SNE requires at least 3 samples.")
        perplexity = min(tsne_perplexity, max(2, (n_samples - 1) // 3))
        perplexity = min(perplexity, n_samples - 1)

        reducer = make_tsne_compatible(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            n_iter=tsne_iter,
            random_state=random_state,
        )
        emb_2d = reducer.fit_transform(feats)
        method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    # ===== 2. 解析层级 =====
    levels = [k for k in hier_labels.keys() if k != "split"]

    parent_level = levels[0]
    child_level = levels[1] if len(levels) > 1 else None

    parent_labels = np.asarray(hier_labels[parent_level])
    child_labels = (
        np.asarray(hier_labels[child_level]) if child_level is not None else None
    )

    split = np.asarray(hier_labels["split"]) if "split" in hier_labels else None

    unique_parents = np.unique(parent_labels)
    unique_children = (
        np.unique(child_labels) if child_labels is not None else np.array([0])
    )

    # ===== 3. 颜色映射 =====
    # 目的：
    # - 保证父层级在所有 scatter 调用中颜色一致
    # - 使用离散 colormap，而不是连续归一化

    if label_names and parent_level in label_names:
        num_parents = len(label_names[parent_level])
    else:
        num_parents = len(unique_parents)

    # tab10 / tab20 自动选择，避免颜色不够
    if num_parents <= 10:
        cmap = plt.get_cmap("tab10")
    elif num_parents <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        # parent 类别非常多时，退化为连续色图（仅定性使用）
        cmap = plt.get_cmap("hsv")

    # 离散边界：[-0.5, 0.5, 1.5, ..., K-0.5]
    boundaries = np.arange(num_parents + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 兼容 matplotlib 的要求

    # ===== 4. test-only 离群点检测（统计定义） =====
    outlier_mask = None

    if split is not None:
        from sklearn.neighbors import NearestNeighbors

        k = min(15, len(emb_2d) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(emb_2d)
        neigh_idx = nbrs.kneighbors(return_distance=False)

        outlier_mask = np.zeros(len(emb_2d), dtype=bool)

        for i in range(len(emb_2d)):
            if split[i] == 1:  # test only
                train_ratio = np.mean(split[neigh_idx[i]] == 0)
                if train_ratio < 0.2:
                    outlier_mask[i] = True

    # ===== 5. marker 池 =====
    markers = ["o", "s", "^", "D", "P", "X", "*", "<", ">"]

    # ===== 6. 绘图 =====
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, c in enumerate(unique_children):
        if child_labels is not None:
            idx = (child_labels == c)
        else:
            idx = np.ones(len(parent_labels), dtype=bool)

        alpha = (
            np.where(split[idx] == 0, 0.85, 0.6) if split is not None else 0.85
        )

        ax.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            c=parent_labels[idx],
            cmap=cmap,
            norm=norm,
            marker=markers[i % len(markers)],
            s=18,
            alpha=alpha,
            edgecolors="none",
        )

    # ===== 7. 标题 / 坐标 =====
    if child_level is None:
        title = f"{method_name}: {parent_level}"
    else:
        title = f"{method_name}: {parent_level} (color) / {child_level} (marker)"

    ax.set_title(title)
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    plt.tight_layout()

    # ===== 8. Colorbar =====
    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(parent_level, rotation=90)
    if label_names and parent_level in label_names:
        names = label_names[parent_level]
        ticks = np.arange(len(names))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(names)
    else:
        cbar.set_ticks(unique_parents)
        cbar.set_ticklabels(unique_parents)

    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
