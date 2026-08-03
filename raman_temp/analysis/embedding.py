"""Train/Val embedding 收集与 UMAP 输出。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_train_val_umap(
    model: torch.nn.Module,
    train_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    output_path: Path,
    neighbors: int,
    min_dist: float,
    seed: int,
) -> None:
    """提取 Train/Val embedding 并保存联合 UMAP 散点图。"""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP 分析需要安装 umap-learn") from exc
    with torch.no_grad():
        _train_logits, train_embeddings = model(train_inputs, return_embedding_enable=True)
        _validation_logits, validation_embeddings = model(
            validation_inputs,
            return_embedding_enable=True,
        )
    arrays = [train_embeddings.cpu().numpy(), validation_embeddings.cpu().numpy()]
    groups = np.asarray(["Train"] * len(arrays[0]) + ["Val"] * len(arrays[1]))
    points = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=seed,
    ).fit_transform(np.vstack(arrays))
    train_mask = groups == "Train"
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(points[train_mask, 0], points[train_mask, 1], s=8, label="Train")
    axis.scatter(points[~train_mask, 0], points[~train_mask, 1], s=12, label="Val")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
