# -*- coding: utf-8 -*-
# ============================================================
#  PCA + SVM using train_files.json / test_files.json
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from raman.config_io import load_experiment
from raman.dataset import RamanDataset
from raman.train_utils import load_split_files


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path):
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


# =========================
# User settings
# =========================
EXP_DIR = resolve_path("output_细菌/20260130_085003")  # set your output folder
LEVEL = "level_1"  # None -> use config.train_level

OUTPUT_DIR = resolve_path(os.path.join("PCA+SVM", "output", os.path.basename(EXP_DIR)))

USE_ALL_CHANNELS = False  # False: use base channel only; True: flatten all channels

PCA_N_COMPONENTS = 0.95  # float for variance ratio, or int for fixed components
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
RANDOM_STATE = 42


def extract_features(dataset, indices, level_idx):
    X_list = []
    y_list = []
    skipped = 0

    for idx in indices:
        x, labels, _ = dataset[idx]
        y = labels[level_idx] if labels.ndim > 0 else labels
        y = int(y)
        if y < 0:
            skipped += 1
            continue

        if USE_ALL_CHANNELS:
            feat = x.reshape(-1).numpy()
        else:
            # Dataset 返回形状为 [C, L]，取第 1 个通道
            feat = x[0].numpy()

        X_list.append(feat)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No valid samples after filtering.")

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64), skipped


def plot_pca_scatter(X_train_pca, y_train, class_names, out_path):
    if X_train_pca.shape[1] < 2:
        print("[Warn] PCA components < 2, skip scatter plot.")
        return

    plt.figure(figsize=(8, 6))
    for cls in range(len(class_names)):
        train_mask = y_train == cls
        if train_mask.any():
            plt.scatter(
                X_train_pca[train_mask, 0],
                X_train_pca[train_mask, 1],
                s=12,
                alpha=0.6,
                label=f"{class_names[cls]}",
            )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA scatter (train only)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(7, 6))
    cm_sum = cm.sum(axis=1, keepdims=True).astype(np.float32)
    cm_sum[cm_sum == 0] = 1.0
    cm_norm = cm.astype(np.float32) / cm_sum

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                annot[i, j] = "0\n(0)"
            else:
                annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})"

    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=annot,
        fmt="",
        annot_kws={"size": 10},
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    exp_dir = resolve_path(EXP_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_experiment(exp_dir)
    dataset_root = resolve_path(config.dataset_root)
    config.dataset_root = dataset_root

    dataset = RamanDataset(dataset_root, augment=False, config=config)

    level = LEVEL or getattr(config, "train_level", None) or "leaf"
    if hasattr(dataset, "_resolve_level_name"):
        level = dataset._resolve_level_name(level)
    if level not in dataset.head_name_to_idx:
        raise ValueError(f"Unknown level: {level}. Available: {dataset.head_names}")
    level_idx = dataset.head_name_to_idx[level]
    class_names = dataset.class_names_by_level[level_idx]

    split = load_split_files(dataset, exp_dir)
    if split is None:
        raise FileNotFoundError("train_files.json/test_files.json not found in EXP_DIR.")
    train_idx, test_idx = split

    X_train, y_train, skipped_train = extract_features(dataset, train_idx, level_idx)
    X_test, y_test, skipped_test = extract_features(dataset, test_idx, level_idx)

    print(f"[Info] Train samples: {len(y_train)} (skipped {skipped_train})")
    print(f"[Info] Test  samples: {len(y_test)} (skipped {skipped_test})")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    svm = SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA)
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    # Save outputs
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"PCA components: {X_train_pca.shape[1]}\n")
        f.write("Explained variance ratio:\n")
        f.write(np.array2string(pca.explained_variance_ratio_, precision=4))
        f.write("\n\n")
        f.write(report)

    np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_pca_scatter(
        X_train_pca,
        y_train,
        class_names,
        os.path.join(OUTPUT_DIR, "pca_scatter.png"),
    )

    print(f"[Saved] outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
