"""在单个真实谱上审阅当前训练增强的效果；图中不标注概率与指标数值。"""

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

from ramanv2.common.arc_data import read_arc_data  # noqa: E402
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
    augment_raw_spectrum,
    augment_strong_baseline,
    augment_weak_baseline,
    build_augmentation_spec,
)
from ramanv2.spectra.preprocess import preprocess_single_spectrum  # noqa: E402

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "病原菌"
DEFAULT_SAMPLE = "Acinetobacter/AB20231212_01/cell16_Area01_000.arc_data"
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "augmentation_study"

CUT_MIN = 400.0
CUT_MAX = 1800.0
TARGET_POINTS = 1024
DRAW_COUNT = 12
DISPLAY_MIN = 600.0
DISPLAY_MAX = 1800.0
BAD_BAND = (900.0, 940.0)

# 五排宽窄双栏版面：边距、行距和栏宽比沿用既有审图比例。
FIGURE_SIZE = (14.63, 14.04)
FIGURE_DPI = 100
GRID_SPEC = {
    "left": 54 / 1463,
    "right": 1441 / 1463,
    "bottom": 53 / 1404,
    "top": 1351 / 1404,
    "hspace": 81 / 195,
    "wspace": 79 / 654,
    "width_ratios": (985, 323),
}
ROW_COLORS = ("#2F5FD0", "#177E6B", "#793DD1", "#C15D00", "#0F1B2D")
GRID_COLOR = "#EDEDED"
DRAW_ALPHA = 0.5
DRAW_WIDTH = 0.8
BASE_WIDTH = 1.6
TICK_WIDTH = 0.8
TICK_LENGTH = 3.5


def _load_corrected_spectrum(dataset_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    wavenumbers, raw = read_arc_data(dataset_dir / "init" / sample_id)
    axis, values, _ = preprocess_single_spectrum(
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
        cosmic_ray_enable=True,
        cosmic_ray_window_points=7,
        cosmic_ray_threshold=7.0,
        cosmic_ray_max_iter=2,
    )
    return axis, values


def _build_operations() -> list[tuple[str, str, object]]:
    spec = build_augmentation_spec(TrainingConfig())

    def draw_gain(items: np.ndarray) -> np.ndarray:
        segments = _build_random_segments(items.size, 60, 180)
        return augment_piecewise_gain(items, segments, PIECEWISE_GAIN_STD)

    operations = (
        ("Piecewise gain", draw_gain),
        ("Axis warp", augment_axis_warp),
        ("Weak baseline", augment_weak_baseline),
        ("Strong baseline", augment_strong_baseline),
        ("Default combination", lambda items: augment_raw_spectrum(items, spec)),
    )
    return [(title, color, operation) for (title, operation), color in zip(operations, ROW_COLORS)]


def build_study_figure(dataset_dir: Path, sample_id: str, out_dir: Path) -> Path:
    """五排宽窄双栏审图：左栏原始强度叠加多次增强，右栏差值；坏区断开并盖灰带。"""
    axis, values = _load_corrected_spectrum(dataset_dir, sample_id)
    operations = _build_operations()
    figure, axes = plt.subplots(
        len(operations),
        2,
        figsize=FIGURE_SIZE,
        gridspec_kw=GRID_SPEC,
        sharex=True,
    )
    for row_index, (title, color, operation) in enumerate(operations):
        left, right = axes[row_index]
        for panel in (left, right):
            apply_canvas(
                panel,
                grid_color=GRID_COLOR,
                spine_color="black",
                spine_width=0.8,
                tick_size=10,
            )
            panel.tick_params(width=TICK_WIDTH, length=TICK_LENGTH)
            add_bad_band_spans(panel, (BAD_BAND,), zorder=4)
        for draw_index in range(DRAW_COUNT):
            np.random.seed(1000 * row_index + draw_index)
            draw = operation(values.copy())
            plot_with_bad_band_breaks(
                left, axis, draw, (BAD_BAND,), color=color, linewidth=DRAW_WIDTH, alpha=DRAW_ALPHA
            )
            plot_with_bad_band_breaks(
                right, axis, draw - values, (BAD_BAND,), color=color, linewidth=DRAW_WIDTH, alpha=DRAW_ALPHA
            )
        plot_with_bad_band_breaks(left, axis, values, (BAD_BAND,), color="black", linewidth=BASE_WIDTH, zorder=3)
        left.set_title(title, loc="left", fontsize=10)
        left.set_ylabel("intensity", fontsize=10)
        right.axhline(0.0, color="black", linewidth=0.8)
        right.set_ylabel("Δ", fontsize=10)
    axes[-1, 0].set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10)
    axes[-1, 1].set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10)
    for left, right in axes:
        left.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
        right.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
    figure.suptitle("Training augmentations on one real spectrum", fontsize=12, y=0.9868)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "augmentation_effects.png"
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="重建单谱增强效果审图")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE, help="init 下的光谱相对路径")
    parser.add_argument("--out-dir", default=None, help="输出目录；默认写入时间戳目录")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    output_path = build_study_figure(DEFAULT_DATASET_DIR, args.sample_id, out_dir)
    print("saved", output_path)


if __name__ == "__main__":
    main()