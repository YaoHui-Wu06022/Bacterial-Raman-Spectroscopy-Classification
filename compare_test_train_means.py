from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from raman.analysis.core import build_wavenumber_axis
from raman.config_io import load_experiment


# 手动设置实验目录
EXP_DIR = "output/细菌/20260408_115130"


def load_processed_spectrum(path):
    """读取预处理后的 .arc_data 强度列。"""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, 1].astype(np.float32)


def snv(x):
    """单条光谱做 SNV，和模型默认输入视角保持一致。"""
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return x - mean
    return (x - mean) / std


def mean_of_spectra(paths, use_snv=False):
    """对一组光谱求均值谱。"""
    spectra = []
    for path in paths:
        x = load_processed_spectrum(path)
        if use_snv:
            x = snv(x)
        spectra.append(x)
    if not spectra:
        raise RuntimeError("No spectra found.")
    return np.mean(np.stack(spectra, axis=0), axis=0)


def minmax(x):
    """用于原始均值谱展示的简单缩放。"""
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def cosine(a, b):
    """计算余弦相似度。"""
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (da * db))


def infer_expected_leaf(folder_name, leaf_labels):
    """根据测试菌文件夹名后缀推断对应训练叶子类。"""
    suffix = "".join(ch for ch in folder_name if not ch.isdigit())
    if suffix.startswith("CS"):
        suffix = suffix[2:]

    matches = []
    for leaf_label in leaf_labels:
        leaf_name = leaf_label.split("/")[-1]
        if suffix.endswith(leaf_name):
            matches.append((leaf_name, leaf_label))

    if not matches:
        return None
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches[0][1]


def build_train_leaf_stats(train_root):
    """为每个训练叶子类预先计算原始均值谱和 SNV 均值谱。"""
    leaf_stats = {}
    for path in sorted(train_root.rglob("*.arc_data")):
        rel = path.relative_to(train_root)
        parts = rel.parts[:-1]
        if len(parts) < 2:
            continue
        label = f"{parts[0]}/{parts[-1]}"
        leaf_stats.setdefault(label, []).append(path)

    out = {}
    for label, paths in sorted(leaf_stats.items()):
        out[label] = {
            "count": len(paths),
            "paths": paths,
            "mean_raw": mean_of_spectra(paths, use_snv=False),
            "mean_snv": mean_of_spectra(paths, use_snv=True),
        }
    return out


def collect_test_folder_stats(test_root):
    """收集每个测试菌文件夹下的样本。"""
    folders = {}
    for folder in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        paths = sorted(folder.rglob("*.arc_data"))
        if paths:
            folders[folder.name] = paths
    return folders


def plot_compare(
    save_path,
    folder_name,
    expected_label,
    nearest_label,
    test_mean_raw,
    test_mean_snv,
    expected_mean_raw,
    expected_mean_snv,
    nearest_mean_raw,
    nearest_mean_snv,
    wavenumbers,
    scores,
):
    """画测试菌均值谱与训练类均值谱对比。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(wavenumbers, minmax(test_mean_raw), label=f"{folder_name} test", linewidth=2.0)
    if expected_mean_raw is not None:
        axes[0].plot(wavenumbers, minmax(expected_mean_raw), label=f"{expected_label} train", linewidth=1.8)
    axes[0].plot(wavenumbers, minmax(nearest_mean_raw), label=f"{nearest_label} nearest", linewidth=1.8)
    axes[0].set_ylabel("Min-Max Mean")
    axes[0].set_title(
        f"{folder_name} | expected={expected_label or 'None'} | "
        f"nearest={nearest_label} | "
        f"expected_cos={scores.get('expected_cos')} | nearest_cos={scores.get('nearest_cos')}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].plot(wavenumbers, test_mean_snv, label=f"{folder_name} test", linewidth=2.0)
    if expected_mean_snv is not None:
        axes[1].plot(wavenumbers, expected_mean_snv, label=f"{expected_label} train", linewidth=1.8)
    axes[1].plot(wavenumbers, nearest_mean_snv, label=f"{nearest_label} nearest", linewidth=1.8)
    axes[1].set_xlabel("Wavenumber")
    axes[1].set_ylabel("SNV Mean")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    tick_count = 6
    idx = np.linspace(0, len(wavenumbers) - 1, tick_count, dtype=int)
    axes[1].set_xticks(wavenumbers[idx])
    axes[1].set_xticklabels([f"{wavenumbers[i]:.0f}" for i in idx])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    exp_dir = Path(EXP_DIR)
    config = load_experiment(str(exp_dir))
    dataset_root = Path(config.dataset_root)
    train_root = dataset_root / "dataset_train"
    test_root = dataset_root / "dataset_test"

    out_dir = exp_dir / "test_train_mean_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"

    train_leaf_stats = build_train_leaf_stats(train_root)
    test_folders = collect_test_folder_stats(test_root)
    leaf_labels = sorted(train_leaf_stats.keys())

    first_train = next(iter(train_leaf_stats.values()))
    wavenumbers = build_wavenumber_axis(len(first_train["mean_raw"]), config)

    rows = []
    for folder_name, paths in test_folders.items():
        test_mean_raw = mean_of_spectra(paths, use_snv=False)
        test_mean_snv = mean_of_spectra(paths, use_snv=True)

        expected_label = infer_expected_leaf(folder_name, leaf_labels)
        scores = []
        for label, stats in train_leaf_stats.items():
            score = cosine(test_mean_snv, stats["mean_snv"])
            scores.append((label, score))
        scores.sort(key=lambda item: item[1], reverse=True)

        if expected_label is not None:
            nearest_wrong = next(label for label, _ in scores if label != expected_label)
            expected_cos = next(score for label, score in scores if label == expected_label)
            nearest_cos = next(score for label, score in scores if label == nearest_wrong)
        else:
            nearest_wrong = scores[0][0]
            expected_cos = None
            nearest_cos = scores[0][1]

        expected_stats = train_leaf_stats.get(expected_label)
        nearest_stats = train_leaf_stats[nearest_wrong]

        plot_path = out_dir / f"{folder_name}.png"
        plot_compare(
            save_path=plot_path,
            folder_name=folder_name,
            expected_label=expected_label,
            nearest_label=nearest_wrong,
            test_mean_raw=test_mean_raw,
            test_mean_snv=test_mean_snv,
            expected_mean_raw=None if expected_stats is None else expected_stats["mean_raw"],
            expected_mean_snv=None if expected_stats is None else expected_stats["mean_snv"],
            nearest_mean_raw=nearest_stats["mean_raw"],
            nearest_mean_snv=nearest_stats["mean_snv"],
            wavenumbers=wavenumbers,
            scores={
                "expected_cos": None if expected_cos is None else f"{expected_cos:.4f}",
                "nearest_cos": f"{nearest_cos:.4f}",
            },
        )

        rows.append(
            {
                "folder": folder_name,
                "expected_label": expected_label or "",
                "nearest_wrong_label": nearest_wrong,
                "expected_cos": "" if expected_cos is None else f"{expected_cos:.6f}",
                "nearest_wrong_cos": f"{nearest_cos:.6f}",
                "margin_expected_minus_wrong": (
                    "" if expected_cos is None else f"{(expected_cos - nearest_cos):.6f}"
                ),
                "top1_label": scores[0][0],
                "top1_cos": f"{scores[0][1]:.6f}",
                "top2_label": scores[1][0],
                "top2_cos": f"{scores[1][1]:.6f}",
                "n_test_spectra": len(paths),
            }
        )

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "folder",
                "expected_label",
                "nearest_wrong_label",
                "expected_cos",
                "nearest_wrong_cos",
                "margin_expected_minus_wrong",
                "top1_label",
                "top1_cos",
                "top2_label",
                "top2_cos",
                "n_test_spectra",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved plots to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
