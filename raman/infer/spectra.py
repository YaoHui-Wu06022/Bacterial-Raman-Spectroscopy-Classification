from pathlib import Path

import numpy as np

from raman.tool.dataset import dataset_bundle_root
from raman.tool.hierarchy import label_from_parts
from raman.tool.naming import normalize_folder_prefix
from raman.tool.path import resolve_project_path
from raman.tool.plotting import add_bad_band_spans, insert_nan_gaps
from raman.tool.spectrum import build_valid_mask, expected_wavenumbers, get_config_bad_bands
from raman.training.split import TRAIN_SPLIT_NAME


def validate_input_length(signal_length, config, source):
    """确认测试谱长度和模型训练配置一致"""
    expected = expected_wavenumbers(config).shape[0]
    if int(signal_length) != int(expected):
        raise ValueError(
            f"Input length mismatch for {source}: got {signal_length}, expected {expected}. "
            f"请确认 dataset/<数据集>/test 中的独立测试谱已经按该模型 run 的输入配置预处理"
        )


def preprocess_with_config_mask(path, preprocessor, config):
    """按模型 bad_bands 对齐已预处理光谱，再构建模型输入"""
    x = preprocessor(path)
    expected = expected_wavenumbers(config).shape[0]
    if int(x.shape[-1]) == int(expected):
        return x

    data = np.loadtxt(path, dtype=np.float32)
    data = np.atleast_2d(data)
    if data.shape[1] >= 2:
        keep_mask = build_valid_mask(data[:, 0], get_config_bad_bands(config))
        if keep_mask is not None and int(keep_mask.sum()) == int(expected):
            from raman.data.input import build_model_input

            signal = data[:, 1][keep_mask].astype(np.float32, copy=False)
            aligned = build_model_input(
                signal,
                config=config,
                sg_smooth=preprocessor.sg_smooth,
                sg_d1=preprocessor.sg_d1,
                device=preprocessor.device,
                augment=False,
            )
            return aligned.unsqueeze(0)

    validate_input_length(x.shape[-1], config, path)
    return x


def _candidate_train_roots(dataset_root):
    """列出可用的训练侧光谱目录"""
    dataset_root = dataset_bundle_root(resolve_project_path(dataset_root))
    train_root = dataset_root / "train"
    if train_root.is_dir():
        return [train_root]
    return []


def _iter_labeled_train_files(train_root, level_name):
    """遍历训练侧光谱并带上业务层标签"""
    train_root = Path(train_root)
    for path in sorted(train_root.rglob("*.arc_data")):
        rel = path.relative_to(train_root)
        if len(rel.parts) < 3:
            continue
        label = label_from_parts(rel.parts[:-1], level_name)
        if label:
            yield path, label, normalize_folder_prefix(rel.parts[-2])


def _load_train_file_list(exp_dir):
    """读取实验根保存的训练文件清单"""
    import json

    path = Path(exp_dir) / TRAIN_SPLIT_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_train_mean_files(exp_dir, dataset_root, level_name):
    """优先使用模型训练清单，缺失时回退到当前 train"""
    dataset_root = dataset_bundle_root(resolve_project_path(dataset_root))
    train_files = _load_train_file_list(exp_dir)
    train_root = dataset_root / "train"
    if train_files and train_root.is_dir():
        rows = []
        for rel in train_files:
            path = train_root / rel
            if not path.is_file():
                continue
            parts = Path(rel).parts[:-1]
            label = label_from_parts(parts, level_name)
            if label:
                rows.append((path, label))
        if rows:
            return rows

    rows = []
    for root in _candidate_train_roots(dataset_root):
        for path, label, _ in _iter_labeled_train_files(root, level_name):
            rows.append((path, label))
    if not rows:
        raise FileNotFoundError(f"No train spectra found under {dataset_root}")
    return rows


def build_train_mean_bank(exp_dir, dataset_root, level_name, preprocessor, config):
    """构建训练均值谱对照库"""
    from tqdm import tqdm

    signals: dict[str, list[np.ndarray]] = {}
    for path, label in tqdm(
        _resolve_train_mean_files(exp_dir, dataset_root, level_name),
        desc="Building train mean spectra",
        unit="spectrum",
    ):
        x = preprocess_with_config_mask(path, preprocessor, config)
        signal = x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        signals.setdefault(label, []).append(signal)
    return {
        label: np.mean(np.stack(items, axis=0), axis=0)
        for label, items in signals.items()
    }


def plot_spectra(
    save_path,
    folder_name,
    test_signals,
    wavenumbers,
    expected_label=None,
    predicted_label=None,
    train_mean_bank=None,
    bad_bands=(),
):
    """绘制测试光谱，可选叠加训练均值对照"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_mean_bank = train_mean_bank or {}

    def plot_line(ax, y, **kwargs):
        wn_plot, y_plot = insert_nan_gaps(wavenumbers, y)
        ax.plot(wn_plot, y_plot, **kwargs)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    add_bad_band_spans(ax, bad_bands, alpha=0.18, label="Removed Bad Band")
    for signal in test_signals:
        plot_line(ax, signal, color="#9ECAE1", alpha=0.38, linewidth=0.9)

    test_mean = test_signals.mean(axis=0)
    plot_line(ax, test_mean, color="#1F77B4", linewidth=2.0, label="Test Mean")

    expected_mean = train_mean_bank.get(expected_label)
    if expected_mean is not None:
        plot_line(ax, expected_mean, color="#E45756", linewidth=2.2, label=f"Train Mean ({expected_label})")

    predicted_mean = train_mean_bank.get(predicted_label)
    if predicted_label != expected_label and predicted_mean is not None:
        plot_line(
            ax,
            predicted_mean,
            color="#F28E2B",
            linewidth=2.0,
            linestyle="--",
            label=f"Predicted Mean ({predicted_label})",
        )

    ax.set_title(f"Spectrum Compare | {folder_name}")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Normalized Intensity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
