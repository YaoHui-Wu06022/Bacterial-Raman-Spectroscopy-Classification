from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from raman.data import InputPreprocessor, RamanDataset
from raman.eval.experiment import (
    load_experiment_with_dataset,
    load_hierarchy_meta,
    resolve_head_level_name,
)
from raman.eval.runtime import build_experiment_runtime
from raman.training import load_split_files


# 手动设置实验目录与分析层级
EXP_DIR = "output/细菌/20260417_073717_6分类"
COMPARE_LEVEL = "level_1"
TOP_K = 3


def _normalize_suffix(folder_name: str) -> str:
    suffix = "".join(ch for ch in folder_name if not ch.isdigit())
    if suffix.startswith("CS"):
        suffix = suffix[2:]
    return suffix


def _build_compare_lookup(dataset: RamanDataset, compare_level: str) -> list[tuple[str, str]]:
    """建立 leaf 名称到目标业务层标签的映射，用于推断测试文件夹的理论正确类别"""
    seen = set()
    entries = []
    for idx, hier in enumerate(dataset.hier_names):
        leaf_label = dataset.get_leaf_key(idx)
        compare_label = hier.get(compare_level)
        if not leaf_label or not compare_label or leaf_label in seen:
            continue
        seen.add(leaf_label)
        leaf_name = leaf_label.split("/")[-1]
        entries.append((leaf_name, compare_label))
    entries.sort(key=lambda item: len(item[0]), reverse=True)
    return entries


def _infer_expected_label(folder_name: str, compare_lookup: list[tuple[str, str]]) -> str | None:
    """根据测试文件夹名后缀，反推该文件夹在 compare_level 上的理论正确类别"""
    suffix = _normalize_suffix(folder_name)
    for leaf_name, compare_label in compare_lookup:
        if suffix.endswith(leaf_name):
            return compare_label
    return None


def _iter_test_folders(test_root: Path) -> dict[str, list[Path]]:
    folders = {}
    for folder in sorted(path for path in test_root.iterdir() if path.is_dir()):
        paths = sorted(folder.rglob("*.arc_data"))
        if paths:
            folders[folder.name] = paths
    return folders


def _l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def _topk_counter(counter: Counter, inv_label_map: dict[int, str], total: int, top_k: int) -> list[dict]:
    items = []
    for label_id, count in counter.most_common(top_k):
        items.append(
            {
                "label": inv_label_map[int(label_id)],
                "count": int(count),
                "ratio": float(count / total) if total else 0.0,
            }
        )
    return items


def _get_wavenumber_axis(config, length: int) -> np.ndarray:
    if hasattr(config, "cut_min") and hasattr(config, "cut_max"):
        return np.linspace(float(config.cut_min), float(config.cut_max), length, dtype=np.float32)
    return np.arange(length, dtype=np.float32)


def _plot_spectrum_comparison(
    save_path: Path,
    folder_name: str,
    expected_label: str | None,
    test_signals: np.ndarray,
    wavenumbers: np.ndarray,
    expected_mean_signal: np.ndarray | None,
):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for signal in test_signals:
        ax.plot(wavenumbers, signal, color="#9ECAE1", alpha=0.45, linewidth=1.0)

    test_mean = test_signals.mean(axis=0)
    ax.plot(wavenumbers, test_mean, color="#1F77B4", linewidth=2.0, label="Test Mean")

    if expected_mean_signal is not None:
        ax.plot(
            wavenumbers,
            expected_mean_signal,
            color="#E45756",
            linewidth=2.4,
            label=f"Train Mean ({expected_label})",
        )

    ax.set_title(f"Spectrum Shape Compare | {folder_name}")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Normalized Intensity")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_vote_distribution(
    save_path: Path,
    folder_name: str,
    expected_label: str | None,
    items: list[dict],
    vote_type: str,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [item["label"] for item in items]
    values = [item["count"] for item in items]
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_title(f"{folder_name} | expected={expected_label or 'None'} | {vote_type}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, item in enumerate(items):
        ax.text(i, values[i], f'{item["count"]}\n{item["ratio"]:.2f}', ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_centroid_similarity(
    save_path: Path,
    folder_name: str,
    expected_label: str | None,
    items: list[dict],
):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [item["label"] for item in items]
    values = [item["score"] for item in items]
    ax.bar(range(len(labels)), values, color="#F28E2B")
    ax.set_title(f"{folder_name} | expected={expected_label or 'None'} | Centroid Similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, item in enumerate(items):
        ax.text(i, values[i], f'{item["score"]:.3f}', ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _build_train_embedding_bank(
    dataset: RamanDataset,
    model: torch.nn.Module,
    compare_level: str,
    train_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
):
    """收集训练集 embedding 及对应标签，作为最近邻分析的训练特征库"""
    level_idx = dataset.head_name_to_idx[compare_level]
    feats_list = []
    labels_list = []

    with torch.no_grad():
        for start in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[start : start + batch_size]
            valid_indices = [int(i) for i in batch_indices if int(dataset.level_labels[int(i), level_idx]) >= 0]
            if not valid_indices:
                continue

            xs = torch.stack([dataset[int(i)][0] for i in valid_indices], dim=0).to(device)
            _, feat = model(xs, return_feat=True)
            feat = _l2_normalize_rows(feat).cpu()

            feats_list.append(feat)
            labels_list.append(dataset.level_labels[valid_indices, level_idx].astype(np.int64))

    if not feats_list:
        raise RuntimeError("No training embeddings were collected.")

    train_feats = torch.cat(feats_list, dim=0)
    train_labels = np.concatenate(labels_list, axis=0)
    return train_feats, train_labels


def _build_train_mean_signal_bank(
    dataset: RamanDataset,
    compare_level: str,
    train_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """按类别统计训练集平均谱形，供测试谱形对照图使用"""
    level_idx = dataset.head_name_to_idx[compare_level]
    signal_bank: dict[int, list[np.ndarray]] = {}

    for idx in train_indices:
        idx = int(idx)
        class_id = int(dataset.level_labels[idx, level_idx])
        if class_id < 0:
            continue
        signal = dataset[idx][0][0].detach().cpu().numpy().astype(np.float32, copy=False)
        signal_bank.setdefault(class_id, []).append(signal)

    mean_bank = {}
    for class_id, signals in signal_bank.items():
        mean_bank[class_id] = np.mean(np.stack(signals, axis=0), axis=0)
    return mean_bank


def _build_class_centroids(train_feats: torch.Tensor, train_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """把训练 embedding 按类别求中心，并做 L2 归一化，供 centroid 相似度分析"""
    feat_dim = train_feats.size(1)
    centroids = torch.zeros((num_classes, feat_dim), dtype=torch.float32)
    valid_mask = torch.zeros(num_classes, dtype=torch.bool)

    for class_id in range(num_classes):
        mask = train_labels == class_id
        if not np.any(mask):
            continue
        center = train_feats[mask].mean(dim=0, keepdim=True)
        centroids[class_id] = _l2_normalize_rows(center)[0]
        valid_mask[class_id] = True

    if not valid_mask.any():
        raise RuntimeError("No valid class centroids were built.")
    return centroids


def _collect_folder_embeddings(
    model: torch.nn.Module,
    preprocessor: InputPreprocessor,
    paths: list[Path],
    device: torch.device,
):
    """提取单个测试文件夹下所有光谱的 embedding、预测类别和主通道谱形"""
    feats = []
    preds = []
    signals = []

    with torch.no_grad():
        for path in paths:
            x = preprocessor(str(path))
            logits, feat = model(x.to(device), return_feat=True)
            probs = torch.softmax(logits, dim=1)[0]
            preds.append(int(torch.argmax(probs).item()))
            feats.append(_l2_normalize_rows(feat)[0].cpu())
            signals.append(x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False))

    folder_feats = torch.stack(feats, dim=0)
    folder_signals = np.stack(signals, axis=0)
    return folder_feats, preds, folder_signals


def main():
    # ---------------- 读取实验、数据集与类别元数据 ----------------
    exp_dir_str, config = load_experiment_with_dataset(EXP_DIR)
    exp_dir = Path(exp_dir_str)

    # 训练集根目录来自实验配置；独立测试集默认约定放在同级的 dataset_test 下
    dataset_train_root = Path(config.dataset_root)
    dataset_test_root = dataset_train_root.parent / "dataset_test"

    # 构建训练数据集对象
    # expected_label、投票统计和 centroid 对比都统一落到 compare_level 上
    device = torch.device("cuda" if getattr(config, "use_gpu", False) and torch.cuda.is_available() else "cpu")
    full_dataset = RamanDataset(dataset_train_root, augment=False, config=config)
    compare_level = resolve_head_level_name(full_dataset, COMPARE_LEVEL)
    level_idx = full_dataset.head_name_to_idx[compare_level]

    # - inv_label_map：把类别 id 还原成类别名，方便出图和汇总
    # - label_map：把 expected_label 反查成类别 id
    # - num_classes：构建模型和类别中心时需要总类别数
    inv_label_map = full_dataset.inv_label_maps_by_level[level_idx]
    label_map = full_dataset.label_maps_by_level[level_idx]
    num_classes = full_dataset.num_classes_by_level[compare_level]

    # 读取训练时保存的层级元数据，确认当前 dataset_train 的类别顺序
    meta = load_hierarchy_meta(exp_dir)
    if meta is None:
        raise FileNotFoundError(f"找不到 hierarchy_meta.json：{exp_dir}")
    train_class_names = meta.get("class_names_by_level", {}).get(compare_level)
    # 匹配数据集目录
    if train_class_names:
        current_class_names = full_dataset.class_names_by_level[level_idx]
        if list(train_class_names) != list(current_class_names):
            raise ValueError(
                f"{compare_level} 的实验类别顺序与当前 dataset_train 不一致，"
                "请确认当前数据目录与实验目录来自同一版数据"
            )

    runtime = build_experiment_runtime(str(exp_dir), device, config=config, meta=meta)
    model = runtime.load_single_level_model(compare_level, num_classes=num_classes)

    # 如果实验目录里已经保存了 train/test 切分，就优先复用训练划分
    # 这样最近邻库、类别中心和平均谱形都和当时训练时看到的训练集保持一致
    split = load_split_files(full_dataset, str(exp_dir))
    if split is None:
        train_indices = np.arange(len(full_dataset), dtype=np.int64)
    else:
        train_indices, _ = split

    # 这三个训练侧“对照库”分别服务于三种诊断视角：
    # 1. train_feats / train_labels：最近邻投票，看测试 embedding 更贴近哪些训练样本
    # 2. train_mean_signal_bank：谱形对照图，看测试谱形是否偏离理论正确类均值
    # 3. class_centroids：类别中心相似度，看整个测试文件夹整体更像哪一类
    train_feats, train_labels = _build_train_embedding_bank(
        full_dataset,
        model,
        compare_level,
        train_indices,
        device,
    )
    train_mean_signal_bank = _build_train_mean_signal_bank(
        full_dataset,
        compare_level,
        train_indices,
    )
    class_centroids = _build_class_centroids(train_feats, train_labels, num_classes)
    train_feats_t = train_feats.t().contiguous()

    preprocessor = InputPreprocessor(config, device)
    test_folders = _iter_test_folders(dataset_test_root)
    compare_lookup = _build_compare_lookup(full_dataset, compare_level)
    signal_length = next(iter(train_mean_signal_bank.values())).shape[0]
    wavenumbers = _get_wavenumber_axis(config, signal_length)

    out_dir = exp_dir / "embedding_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"

    # ---------------- 逐个测试文件夹做三类诊断 ----------------
    # 1. 谱形对照：测试谱形 vs 理论正确类别训练均值
    # 2. 邻域判断：模型投票 / embedding 最近邻投票
    # 3. 中心判断：测试文件夹中心与各类 centroid 的相似度
    rows = []
    for folder_name, paths in test_folders.items():
        folder_out_dir = out_dir / folder_name
        folder_out_dir.mkdir(parents=True, exist_ok=True)

        # 根据测试文件夹名反推“理论正确类”
        expected_label = _infer_expected_label(folder_name, compare_lookup)
        expected_id = label_map.get(expected_label) if expected_label is not None else None

        # 提取该测试文件夹下每条光谱的 embedding、模型 top1 预测以及主通道谱形
        folder_feats, model_preds, folder_signals = _collect_folder_embeddings(model, preprocessor, paths, device)

        # 最近邻分析：直接看每条测试谱在训练 embedding 库里最靠近哪条训练谱
        # 如果这里稳定指向错误类，通常说明问题更偏 embedding 空间本身，而不是分类头
        similarity = torch.matmul(folder_feats, train_feats_t)
        nearest_indices = torch.argmax(similarity, dim=1).cpu().numpy()
        neighbor_preds = train_labels[nearest_indices].astype(np.int64)

        # 模型投票与最近邻投票都按“整个测试文件夹”汇总
        model_counter = Counter(int(pred) for pred in model_preds)
        neighbor_counter = Counter(int(pred) for pred in neighbor_preds.tolist())
        n_test = len(paths)

        model_items = _topk_counter(model_counter, inv_label_map, n_test, TOP_K)
        neighbor_items = _topk_counter(neighbor_counter, inv_label_map, n_test, TOP_K)

        # centroid 分析：把整个测试文件夹的 embedding 先求一个中心
        # 再和各类别训练中心做余弦相似度，适合看“这批样本整体最像谁”
        folder_centroid = _l2_normalize_rows(folder_feats.mean(dim=0, keepdim=True))[0]
        centroid_scores = torch.matmul(class_centroids, folder_centroid)
        centroid_topk_idx = torch.argsort(centroid_scores, descending=True)[:TOP_K].cpu().numpy().tolist()
        centroid_items = [
            {
                "label": inv_label_map[int(class_id)],
                "score": float(centroid_scores[int(class_id)].item()),
            }
            for class_id in centroid_topk_idx
        ]
        model_top1_label = model_items[0]["label"] if model_items else ""
        model_top1_ratio = model_items[0]["ratio"] if model_items else 0.0
        neighbor_top1_label = neighbor_items[0]["label"] if neighbor_items else ""
        neighbor_top1_ratio = neighbor_items[0]["ratio"] if neighbor_items else 0.0
        centroid_top1_label = centroid_items[0]["label"] if centroid_items else ""
        centroid_top1_score = centroid_items[0]["score"] if centroid_items else 0.0
        expected_mean_signal = None if expected_id is None else train_mean_signal_bank.get(int(expected_id))

        # 输出四张图：
        # 1. spectra.png：测试谱形与理论正确类训练均值的对照
        # 2. model_vote.png：模型 top1 投票分布
        # 3. neighbor_vote.png：embedding 最近邻投票分布
        # 4. centroid_similarity.png：测试文件夹中心与各类别中心的相似度
        _plot_spectrum_comparison(
            folder_out_dir / "spectra.png",
            folder_name,
            expected_label,
            folder_signals,
            wavenumbers,
            expected_mean_signal,
        )
        _plot_vote_distribution(
            folder_out_dir / "model_vote.png",
            folder_name,
            expected_label,
            model_items,
            "Model Vote",
        )
        _plot_vote_distribution(
            folder_out_dir / "neighbor_vote.png",
            folder_name,
            expected_label,
            neighbor_items,
            "Embedding Neighbor Vote",
        )
        _plot_centroid_similarity(
            folder_out_dir / "centroid_similarity.png",
            folder_name,
            expected_label,
            centroid_items,
        )

        # 汇总为一行，便于后续在表格里快速定位是哪一种分析视角出了问题
        # 常见的解读方式是：
        # - model 错、neighbor 对：更像分类头决策边界有问题
        # - model 和 neighbor 都错到同一类：更像 embedding 已经偏到错误类
        # - centroid 与 expected 的 margin 很小：说明这批样本整体上和错误类距离也很近
        rows.append(
            {
                "folder": folder_name,
                "expected_label": expected_label or "",
                "model_top1_label": model_top1_label,
                "model_top1_ratio": f"{model_top1_ratio:.6f}",
                "neighbor_top1_label": neighbor_top1_label,
                "neighbor_top1_ratio": f"{neighbor_top1_ratio:.6f}",
                "centroid_top1_label": centroid_top1_label,
                "centroid_top1_score": f"{centroid_top1_score:.6f}",
            }
        )

    # ---------------- 输出总表，方便统一浏览所有测试文件夹 ----------------
    # summary.csv 只保留每个测试文件夹最关键的 expected / model / neighbor / centroid 结果
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "folder",
                "expected_label",
                "model_top1_label",
                "model_top1_ratio",
                "neighbor_top1_label",
                "neighbor_top1_ratio",
                "centroid_top1_label",
                "centroid_top1_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Embedding compare saved to:", out_dir)


if __name__ == "__main__":
    main()
