from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from raman.data import InputPreprocessor, RamanDataset
from raman.eval.experiment import load_experiment_with_train_dataset, resolve_head_level_name
from raman.model import RamanClassifier1D
from raman.training import load_split_files


# 手动设置实验目录
EXP_DIR = "output/细菌/20260417_073717_6分类"
COMPARE_LEVEL = "level_1"  # 必须显式设置为业务层
TOP_K = 3

def _load_hierarchy_meta(exp_dir: Path) -> dict:
    meta_path = exp_dir / "hierarchy_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_model_path(exp_dir: Path, compare_level: str) -> Path:
    meta = _load_hierarchy_meta(exp_dir)
    level_models = meta.get("level_models", {})
    model_name = level_models.get(compare_level, f"{compare_level}_model.pt")
    model_path = exp_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {compare_level}: {model_path}")
    return model_path


def _normalize_suffix(folder_name: str) -> str:
    suffix = "".join(ch for ch in folder_name if not ch.isdigit())
    if suffix.startswith("CS"):
        suffix = suffix[2:]
    return suffix


def _build_compare_lookup(dataset: RamanDataset, compare_level: str) -> list[tuple[str, str]]:
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


def _counter_ratio(counter: Counter, label_id: int | None, total: int) -> float:
    if label_id is None or total == 0:
        return 0.0
    return float(counter.get(int(label_id), 0) / total)


def _nearest_wrong_neighbor(counter: Counter, expected_id: int | None, inv_label_map: dict[int, str]) -> tuple[str, float]:
    total = sum(counter.values())
    for label_id, count in counter.most_common():
        if expected_id is None or int(label_id) != int(expected_id):
            return inv_label_map[int(label_id)], float(count / total) if total else 0.0
    return "", 0.0


def _format_topk(items: list[dict]) -> str:
    return json.dumps(items, ensure_ascii=False)


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
    model: RamanClassifier1D,
    compare_level: str,
    train_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
):
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
    model: RamanClassifier1D,
    preprocessor: InputPreprocessor,
    paths: list[Path],
    device: torch.device,
):
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
    exp_dir_str, config = load_experiment_with_train_dataset(EXP_DIR)
    exp_dir = Path(exp_dir_str)
    dataset_train_root = Path(config.dataset_root)
    dataset_test_root = dataset_train_root.parent / "dataset_test"

    device = torch.device("cuda" if getattr(config, "use_gpu", False) and torch.cuda.is_available() else "cpu")
    full_dataset = RamanDataset(dataset_train_root, augment=False, config=config)
    compare_level = resolve_head_level_name(full_dataset, COMPARE_LEVEL)
    level_idx = full_dataset.head_name_to_idx[compare_level]
    inv_label_map = full_dataset.inv_label_maps_by_level[level_idx]
    label_map = full_dataset.label_maps_by_level[level_idx]
    num_classes = full_dataset.num_classes_by_level[compare_level]
    meta = _load_hierarchy_meta(exp_dir)
    train_class_names = meta.get("class_names_by_level", {}).get(compare_level)
    if train_class_names:
        current_class_names = full_dataset.class_names_by_level[level_idx]
        if list(train_class_names) != list(current_class_names):
            raise ValueError(
                f"{compare_level} 的实验类别顺序与当前 dataset_train 不一致，"
                "请确认当前数据目录与实验目录来自同一版数据。"
            )

    model_path = _resolve_model_path(exp_dir, compare_level)
    model = RamanClassifier1D(num_classes=num_classes, config=config).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    split = load_split_files(full_dataset, str(exp_dir))
    if split is None:
        train_indices = np.arange(len(full_dataset), dtype=np.int64)
    else:
        train_indices, _ = split

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

    rows = []
    for folder_name, paths in test_folders.items():
        folder_out_dir = out_dir / folder_name
        folder_out_dir.mkdir(parents=True, exist_ok=True)
        expected_label = _infer_expected_label(folder_name, compare_lookup)
        expected_id = label_map.get(expected_label) if expected_label is not None else None

        folder_feats, model_preds, folder_signals = _collect_folder_embeddings(model, preprocessor, paths, device)
        similarity = torch.matmul(folder_feats, train_feats_t)
        nearest_indices = torch.argmax(similarity, dim=1).cpu().numpy()
        neighbor_preds = train_labels[nearest_indices].astype(np.int64)

        model_counter = Counter(int(pred) for pred in model_preds)
        neighbor_counter = Counter(int(pred) for pred in neighbor_preds.tolist())
        n_test = len(paths)

        model_items = _topk_counter(model_counter, inv_label_map, n_test, TOP_K)
        neighbor_items = _topk_counter(neighbor_counter, inv_label_map, n_test, TOP_K)

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
        centroid_top1_id = int(centroid_topk_idx[0])
        centroid_top1_label = inv_label_map[centroid_top1_id]
        centroid_top1_cos = float(centroid_scores[centroid_top1_id].item())
        expected_centroid_cos = float(centroid_scores[int(expected_id)].item()) if expected_id is not None else None

        nearest_wrong_centroid_label = ""
        nearest_wrong_centroid_cos = None
        if expected_id is None:
            nearest_wrong_centroid_label = centroid_top1_label
            nearest_wrong_centroid_cos = centroid_top1_cos
        else:
            for class_id in centroid_topk_idx:
                if int(class_id) != int(expected_id):
                    nearest_wrong_centroid_label = inv_label_map[int(class_id)]
                    nearest_wrong_centroid_cos = float(centroid_scores[int(class_id)].item())
                    break

        nearest_wrong_neighbor_label, nearest_wrong_neighbor_ratio = _nearest_wrong_neighbor(
            neighbor_counter,
            expected_id,
            inv_label_map,
        )

        model_top1_label = model_items[0]["label"] if model_items else ""
        model_top1_count = model_items[0]["count"] if model_items else 0
        model_top1_ratio = model_items[0]["ratio"] if model_items else 0.0
        neighbor_top1_label = neighbor_items[0]["label"] if neighbor_items else ""
        neighbor_top1_count = neighbor_items[0]["count"] if neighbor_items else 0
        neighbor_top1_ratio = neighbor_items[0]["ratio"] if neighbor_items else 0.0
        expected_mean_signal = None if expected_id is None else train_mean_signal_bank.get(int(expected_id))

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

        rows.append(
            {
                "folder": folder_name,
                "expected_label": expected_label or "",
                "model_top1_label": model_top1_label,
                "model_top1_count": model_top1_count,
                "model_top1_ratio": f"{model_top1_ratio:.6f}",
                "neighbor_top1_label": neighbor_top1_label,
                "neighbor_top1_count": neighbor_top1_count,
                "neighbor_top1_ratio": f"{neighbor_top1_ratio:.6f}",
                "expected_model_ratio": f"{_counter_ratio(model_counter, expected_id, n_test):.6f}",
                "expected_neighbor_ratio": f"{_counter_ratio(neighbor_counter, expected_id, n_test):.6f}",
                "nearest_wrong_neighbor_label": nearest_wrong_neighbor_label,
                "nearest_wrong_neighbor_ratio": f"{nearest_wrong_neighbor_ratio:.6f}",
                "centroid_top1_label": centroid_top1_label,
                "centroid_top1_cos": f"{centroid_top1_cos:.6f}",
                "expected_centroid_cos": "" if expected_centroid_cos is None else f"{expected_centroid_cos:.6f}",
                "nearest_wrong_centroid_label": nearest_wrong_centroid_label,
                "nearest_wrong_centroid_cos": "" if nearest_wrong_centroid_cos is None else f"{nearest_wrong_centroid_cos:.6f}",
                "centroid_margin_expected_minus_wrong": (
                    ""
                    if expected_centroid_cos is None or nearest_wrong_centroid_cos is None
                    else f"{(expected_centroid_cos - nearest_wrong_centroid_cos):.6f}"
                ),
                "spectrum_plot": f"{folder_name}/spectra.png",
                "model_vote_plot": f"{folder_name}/model_vote.png",
                "neighbor_vote_plot": f"{folder_name}/neighbor_vote.png",
                "centroid_similarity_plot": f"{folder_name}/centroid_similarity.png",
                "model_topk": _format_topk(model_items),
                "neighbor_topk": _format_topk(neighbor_items),
                "centroid_topk": json.dumps(centroid_items, ensure_ascii=False),
                "n_test_spectra": n_test,
            }
        )

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "folder",
                "expected_label",
                "model_top1_label",
                "model_top1_count",
                "model_top1_ratio",
                "neighbor_top1_label",
                "neighbor_top1_count",
                "neighbor_top1_ratio",
                "expected_model_ratio",
                "expected_neighbor_ratio",
                "nearest_wrong_neighbor_label",
                "nearest_wrong_neighbor_ratio",
                "centroid_top1_label",
                "centroid_top1_cos",
                "expected_centroid_cos",
                "nearest_wrong_centroid_label",
                "nearest_wrong_centroid_cos",
                "centroid_margin_expected_minus_wrong",
                "spectrum_plot",
                "model_vote_plot",
                "neighbor_vote_plot",
                "centroid_similarity_plot",
                "model_topk",
                "neighbor_topk",
                "centroid_topk",
                "n_test_spectra",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Embedding compare saved to:", out_dir)


if __name__ == "__main__":
    main()
