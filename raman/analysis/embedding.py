import inspect

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE
import torch

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

def _fill_missing_labels(hier, level_names, missing_tag):
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

        # -------- test --------
        for x, _, hier in test_loader:
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
