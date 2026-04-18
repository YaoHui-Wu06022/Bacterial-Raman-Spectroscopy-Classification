import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import compute_classification_metrics
from .experiment import (
    load_experiment_with_dataset,
    resolve_head_level_name,
)
from .report import (
    format_classification_report_text,
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    write_text,
)
from raman.data import RamanDataset
from raman.training import load_split_files


@dataclass
class BaselineContext:
    """收拢一次 PCA+SVM 基线评估所需的运行上下文"""

    exp_dir: str
    config: object
    dataset_root: str
    level: str | None
    use_all_channels: bool
    pca_n_components: float | int
    svm_c: float
    svm_kernel: str
    svm_gamma: str
    random_state: int


@dataclass
class BaselineOverrides:
    """统一收拢基线评估的覆盖项"""

    exp_dir: str | None = None
    level: str | None = None
    use_all_channels: bool = False
    pca_n_components: float | int = 0.95
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    random_state: int = 42


def configure_baseline(overrides=None):
    """按覆盖项构建 PCA+SVM 基线评估上下文"""
    overrides = overrides or BaselineOverrides()
    if not overrides.exp_dir:
        raise ValueError("pca_svm_baseline 需要显式传入 exp_dir")

    exp_dir, config = load_experiment_with_dataset(overrides.exp_dir)
    return BaselineContext(
        exp_dir=exp_dir,
        config=config,
        dataset_root=config.dataset_root,
        level=overrides.level,
        use_all_channels=bool(overrides.use_all_channels),
        pca_n_components=overrides.pca_n_components,
        svm_c=float(overrides.svm_c),
        svm_kernel=overrides.svm_kernel,
        svm_gamma=overrides.svm_gamma,
        random_state=int(overrides.random_state),
    )


def extract_features(dataset, indices, level_idx, use_all_channels):
    """按索引提取特征和标签，并跳过无效标签样本"""
    x_list = []
    y_list = []
    skipped = 0

    for idx in indices:
        x, labels, _ = dataset[idx]
        y = int(labels[level_idx] if labels.ndim > 0 else labels)
        if y < 0:
            skipped += 1
            continue

        feat = x.reshape(-1).numpy() if use_all_channels else x[0].numpy()
        x_list.append(feat)
        y_list.append(y)

    if not x_list:
        raise RuntimeError("筛选后没有有效样本")

    return np.stack(x_list, axis=0), np.array(y_list, dtype=np.int64), skipped


def evaluate_test_set(context):
    """使用训练阶段切分结果运行 PCA+SVM 基线评估"""
    config = context.config
    dataset = RamanDataset(context.dataset_root, augment=False, config=config)

    level = resolve_head_level_name(
        dataset,
        context.level,
    )
    level_idx = dataset.head_name_to_idx[level]
    class_names = dataset.class_names_by_level[level_idx]

    result_dir = os.path.join(context.exp_dir, f"{level}_baseline_test_result")
    os.makedirs(result_dir, exist_ok=True)

    out_metrics = os.path.join(result_dir, "metrics.txt")
    out_cm_png = os.path.join(result_dir, "confusion_matrix.png")
    out_cm_raw = os.path.join(result_dir, "confusion_matrix.csv")
    out_pca_png = os.path.join(result_dir, "pca_scatter.png")

    split = load_split_files(dataset, context.exp_dir)
    if split is None:
        raise FileNotFoundError("train_files.json/test_files.json not found in EXP_DIR.")
    train_idx, test_idx = split

    x_train, y_train, skipped_train = extract_features(
        dataset,
        train_idx,
        level_idx,
        context.use_all_channels,
    )
    x_test, y_test, skipped_test = extract_features(
        dataset,
        test_idx,
        level_idx,
        context.use_all_channels,
    )

    print(f"[Info] 训练样本数: {len(y_train)} (跳过 {skipped_train})")
    print(f"[Info] 测试样本数: {len(y_test)} (跳过 {skipped_test})")

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)

    pca = PCA(n_components=context.pca_n_components, random_state=context.random_state)
    x_train_pca = pca.fit_transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    svm = SVC(C=context.svm_c, kernel=context.svm_kernel, gamma=context.svm_gamma)
    svm.fit(x_train_pca, y_train)
    y_pred = svm.predict(x_test_pca)

    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        labels=range(len(class_names)),
    )
    acc = metrics["accuracy"]
    print(f"\n[Baseline] 测试集 Accuracy: {acc * 100:.4f}%")

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = format_classification_report_text(report_dict, class_names, acc)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    metrics_text = (
        f"Accuracy: {acc * 100:.4f}%\n"
        f"PCA components: {x_train_pca.shape[1]}\n"
        "Explained variance ratio:\n"
        f"{np.array2string(pca.explained_variance_ratio_, precision=4)}\n\n"
        f"{report_text}"
    )
    write_text(out_metrics, metrics_text)
    save_confusion_matrix_csv(cm, class_names, out_cm_raw)
    save_confusion_matrix_figure(cm, class_names, out_cm_png)
    if x_train_pca.shape[1] < 2:
        print("[Warn] PCA 维数小于 2，跳过散点图")
    else:
        plt.figure(figsize=(8, 6))
        for cls_idx, cls_name in enumerate(class_names):
            train_mask = y_train == cls_idx
            if not train_mask.any():
                continue
            plt.scatter(
                x_train_pca[train_mask, 0],
                x_train_pca[train_mask, 1],
                s=12,
                alpha=0.6,
                label=cls_name,
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA scatter (train only)")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_pca_png, dpi=300)
        plt.close()

    print("All BASELINE TEST results saved to:", result_dir)
    return result_dir


def run_pca_svm_baseline(overrides=None):
    """先应用覆盖项，再执行 PCA+SVM 基线评估"""
    context = configure_baseline(overrides)
    return evaluate_test_set(context)
