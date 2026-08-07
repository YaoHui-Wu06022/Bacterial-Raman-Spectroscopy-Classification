import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import torch

def _fill_missing_labels(hier, level_names, missing_tag):
    """把缺失层级标签统一填成可视化用的占位名称"""
    out = {}
    for name in level_names:
        vals = list(hier.get(name, []))
        filled = []
        for v in vals:
            if v is None or v == missing_tag:
                filled.append(missing_tag)
            else:
                filled.append(v)
        out[name] = filled
    return out

def collect_embeddings_train_val(
    model,
    train_loader,
    val_loader,
    device,
    dataset,
    level_names=None,
    return_label_names=False,
):
    """收集 train/val 的模型 embedding，并保留业务层级标签用于联合可视化"""
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
        # 训练集样本标记为 split=0
        for x, _, hier in train_loader:
            x = x.to(device)
            if inherit_missing:
                hier_filled = _fill_missing_labels(hier, level_names, missing_tag)
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

        # 验证集样本标记为 split=1
        for x, _, hier in val_loader:
            x = x.to(device)
            if inherit_missing:
                hier_filled = _fill_missing_labels(hier, level_names, missing_tag)
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
    n_neighbors=15,
    min_dist=0.1,
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
    # ===== 1. 准备 UMAP embedding =====
    import umap.umap_ as umap

    n_samples = feats.shape[0]
    actual_neighbors = min(n_neighbors, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    emb_2d = reducer.fit_transform(feats)
    method_name = "UMAP"

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
        parent_names = list(label_names[parent_level])
    else:
        num_parents = len(unique_parents)
        parent_names = None

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
    sm.set_array([])

    def _parent_label(value):
        """返回父层类别显示名"""
        idx = int(value)
        if parent_names and 0 <= idx < len(parent_names):
            return str(parent_names[idx])
        return str(value)

    def _legend_layout(count):
        """按类别数调整右侧图例"""
        if count <= 12:
            return 1, 8, 0.84
        if count <= 30:
            return 2, 7, 0.78
        return 3, 6, 0.70
    # ===== 4. marker 池 =====
    markers = ["o", "s", "^", "D", "P", "X", "*", "<", ">"]

    def _scatter_subset(ax, sample_mask, alpha=0.85):
        """按当前子图的样本 mask 绘制，同一类别映射保持全局一致"""
        has_points = False
        for i, c in enumerate(unique_children):
            if child_labels is not None:
                idx = sample_mask & (child_labels == c)
            else:
                idx = sample_mask

            if not np.any(idx):
                continue

            has_points = True
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

        if not has_points:
            ax.text(
                0.5,
                0.5,
                "No samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.5",
            )

    # ===== 5. 坐标范围 =====
    x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
    y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
    x_pad = max((x_max - x_min) * 0.05, 1e-3)
    y_pad = max((y_max - y_min) * 0.05, 1e-3)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # ===== 6. 绘图 =====
    legend_cols, legend_font, right_margin = _legend_layout(num_parents)
    legend_rows = int(np.ceil(max(num_parents, 1) / legend_cols))
    fig_height = max(6.0, min(14.0, 0.32 * legend_rows + 1.8))

    if split is not None:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, fig_height),
            sharex=True,
            sharey=True,
        )
        plot_specs = [
            ("Train", split == 0, 0.85),
            ("Val", split == 1, 0.85),
        ]
        for ax, (split_name, sample_mask, alpha) in zip(axes, plot_specs):
            _scatter_subset(ax, sample_mask, alpha=alpha)
            ax.set_title(split_name)
            ax.set_xlabel(f"{method_name}-1")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        axes[0].set_ylabel(f"{method_name}-2")
        legend_anchor = (right_margin + 0.01, 0.5)
    else:
        fig, ax = plt.subplots(figsize=(9.5, fig_height))
        _scatter_subset(ax, np.ones(len(parent_labels), dtype=bool), alpha=0.85)
        ax.set_title(method_name)
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        legend_anchor = (right_margin + 0.01, 0.5)

    # ===== 7. 类别图例 =====
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=sm.to_rgba(int(parent_id)),
            markeredgecolor="none",
            markersize=6,
            label=_parent_label(parent_id),
        )
        for parent_id in unique_parents
    ]
    fig.subplots_adjust(left=0.055, right=right_margin, bottom=0.10, top=0.90, wspace=0.08)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=legend_anchor,
        ncol=legend_cols,
        fontsize=legend_font,
        frameon=False,
    )

    plt.savefig(save_path, dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close()
