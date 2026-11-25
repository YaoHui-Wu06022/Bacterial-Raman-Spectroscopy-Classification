"""在单谱上审阅训练增强：四种单谱扰动出一张 2×2 图，同标签 mixup 单独一张。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ramanv2.data.arc_data import read_arc_data  # noqa: E402
from ramanv2.common.plotting import (  # noqa: E402
    add_bad_band_spans,
    apply_canvas,
    plot_with_bad_band_breaks,
)
from ramanv2.core.config import TrainingConfig  # noqa: E402
from ramanv2.data.runtime.augmentation import (  # noqa: E402
    PIECEWISE_GAIN_STD,
    _build_random_segments,
    augment_axis_warp,
    augment_piecewise_gain,
    augment_strong_baseline,
    augment_weak_baseline,
)
from ramanv2.spectra.preprocess import preprocess_single_spectrum  # noqa: E402

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "病原菌"
DEFAULT_SAMPLE = "Acinetobacter/AB20231212_01/cell16_Area01_000.arc_data"
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "augmentation_study"

CUT_MIN = 400.0
CUT_MAX = 1800.0
TARGET_POINTS = 1024
BAD_BAND = (900.0, 940.0)
DRAW_COUNT = 12

FIGURE_DPI = 120
PERTURBATION_SIZE = (13.0, 8.5)
MIXUP_SIZE = (11.0, 6.0)
GRID_COLOR = "#EDEDED"
SPINE_WIDTH = 0.8
TICK_WIDTH = 0.8
TICK_LENGTH = 3.5
TICK_SIZE = 10
DRAW_ALPHA = 0.45
DRAW_WIDTH = 0.9
BASE_WIDTH = 1.0
PERTURBATION_COLORS = ("#2F5FD0", "#177E6B", "#793DD1", "#C15D00")
MIXUP_COLORS = ("#793DD1", "#177E6B", "#C15D00", "#2F5FD0", "#B84A62")
MIXUP_COUNT = 6


def _preprocess_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """按训练口径把一条原始谱处理到 1024 点强度域，供审图叠加使用。"""
    wavenumbers, raw = read_arc_data(path)
    axis, values = preprocess_single_spectrum(
        wavenumbers,
        raw,
        CUT_MIN,
        CUT_MAX,
        np.linspace(CUT_MIN, CUT_MAX, TARGET_POINTS),
        (),
        1e5,
        0.01,
        15,
        baseline_fit_min=200.0,
        baseline_fit_max=2000.0,
        baseline_method="airPLS",
    )
    return axis, values


def _load_corrected_spectrum(dataset_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    return _preprocess_spectrum(dataset_dir / "init" / sample_id)


def _load_class_partners(dataset_dir: Path, sample_id: str) -> list[np.ndarray]:
    """读取同目录下其余同类光谱，作为同标签 mixup 的另一半。"""
    sample_path = dataset_dir / "init" / sample_id
    partners = []
    for path in sorted(sample_path.parent.glob("*.arc_data")):
        if path.name == sample_path.name:
            continue
        _, values = _preprocess_spectrum(path)
        partners.append(values)
    if not partners:
        raise ValueError(f"同类目录没有可配对的第二条光谱：{sample_path.parent}")
    return partners


def _build_perturbations() -> list[tuple[str, object]]:
    def draw_gain(items: np.ndarray) -> np.ndarray:
        segments = _build_random_segments(items.size, 60, 180)
        return augment_piecewise_gain(items, segments, PIECEWISE_GAIN_STD)

    return [
        ("A  Piecewise gain", draw_gain),
        ("B  Axis warp", augment_axis_warp),
        ("C  Weak baseline", augment_weak_baseline),
        ("D  Strong baseline", augment_strong_baseline),
    ]


def _prepare_panel(panel) -> None:
    apply_canvas(
        panel,
        grid_color=GRID_COLOR,
        spine_color="black",
        spine_width=SPINE_WIDTH,
        tick_size=TICK_SIZE,
    )
    panel.tick_params(width=TICK_WIDTH, length=TICK_LENGTH)
    add_bad_band_spans(panel, (BAD_BAND,), zorder=4)


def _plot_curve(panel, axis: np.ndarray, values: np.ndarray, color: str, **options) -> None:
    plot_with_bad_band_breaks(panel, axis, values, (BAD_BAND,), color=color, **options)


def build_perturbation_figure(
    dataset_dir: Path, sample_id: str, out_dir: Path
) -> Path:
    """四种子谱扰动排成 2×2：每格是原始黑谱叠加多次同算子增强。"""
    axis, values = _load_corrected_spectrum(dataset_dir, sample_id)
    figure, axes = plt.subplots(2, 2, figsize=PERTURBATION_SIZE, sharex=True, sharey=True)
    for index, (title, operation) in enumerate(_build_perturbations()):
        panel = axes.ravel()[index]
        _prepare_panel(panel)
        for draw_index in range(DRAW_COUNT):
            np.random.seed(1000 * index + draw_index)
            draw = operation(values.copy())
            _plot_curve(panel, axis, draw, PERTURBATION_COLORS[index], linewidth=DRAW_WIDTH, alpha=DRAW_ALPHA)
        _plot_curve(panel, axis, values, "black", linewidth=BASE_WIDTH, zorder=3)
        panel.set_title(title, loc="left", fontsize=11)
        panel.set_xlim(float(axis.min()), float(axis.max()))
    for row_index in range(2):
        axes[row_index, 0].set_ylabel("intensity", fontsize=10)
    for panel in axes[0]:
        panel.tick_params(labelbottom=False)
    for panel in axes[1]:
        panel.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10)
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "augmentation_perturbations.png"
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def build_mixup_figure(dataset_dir: Path, sample_id: str, out_dir: Path) -> Path:
    """同标签 mixup 单独一张：多条与同类谱按 Beta 权重混合的结果叠加。"""
    axis, values = _load_corrected_spectrum(dataset_dir, sample_id)
    partners = _load_class_partners(dataset_dir, sample_id)
    alpha = float(TrainingConfig().mixup_alpha)
    figure, panel = plt.subplots(figsize=MIXUP_SIZE)
    _prepare_panel(panel)
    for draw_index in range(MIXUP_COUNT):
        np.random.seed(draw_index)
        partner = partners[np.random.randint(len(partners))]
        weight = np.random.beta(alpha, alpha)
        mixed = weight * values + (1.0 - weight) * partner
        _plot_curve(
            panel,
            axis,
            mixed,
            MIXUP_COLORS[draw_index % len(MIXUP_COLORS)],
            linewidth=1.2,
            alpha=0.75,
        )
    panel.set_title("Same-class mixup (batch)", loc="left", fontsize=12)
    panel.set_xlim(float(axis.min()), float(axis.max()))
    panel.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10)
    panel.set_ylabel("intensity", fontsize=10)
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "augmentation_mixup.png"
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="重建单谱增强效果审图")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE, help="init 下的光谱相对路径")
    parser.add_argument("--out-dir", default=None, help="输出目录；默认写入时间戳目录")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    paths = (
        build_perturbation_figure(DEFAULT_DATASET_DIR, args.sample_id, out_dir),
        build_mixup_figure(DEFAULT_DATASET_DIR, args.sample_id, out_dir),
    )
    for path in paths:
        print("saved", path)


if __name__ == "__main__":
    main()
