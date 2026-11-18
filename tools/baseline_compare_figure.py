"""在同一条光谱上对比 AsLS、airPLS 与 5 阶多项式三种基线。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ramanv2.common.arc_data import read_arc_data  # noqa: E402
from ramanv2.spectra.bad_bands import build_valid_mask  # noqa: E402
from ramanv2.spectra.preprocess import estimate_baseline  # noqa: E402

DEFAULT_SAMPLE = (
    REPO_ROOT
    / "dataset"
    / "病原菌"
    / "init"
    / "Klebsiella"
    / "KOX20240110_01"
    / "cell10_Area01_000.arc_data"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "baseline_inspection" / "baseline_methods_compare.png"

X_LIMITS = (200.0, 2000.0)
BAD_BANDS = ((900.0, 940.0),)
BASELINE_LAMBDA = 1e5
BASELINE_P = 0.01
BASELINE_ITERATIONS = 15

FIGURE_SIZE = (11.0, 6.0)
FIGURE_DPI = 140
RAW_COLOR = "#4C72B0"
RAW_WIDTH = 1.0
METHODS = (
    ("Polynomial", "poly", "#B84A62"),
    ("AsLS", "asls", "#4E9A51"),
    ("airPLS", "airpls", "#E08214"),
)
METHOD_WIDTH = 2.0
GRID_COLOR = "#EDEDED"
TICK_SIZE = 12
LABEL_SIZE = 13


def _style(panel) -> None:
    panel.grid(True, color=GRID_COLOR, linewidth=0.8)
    panel.tick_params(labelsize=TICK_SIZE)
    for spine in panel.spines.values():
        spine.set_color("#5F6368")
        spine.set_linewidth(1.2)


def build_figure(sample_path: Path, output_path: Path) -> Path:
    """绘制原始谱与三种方法的拟合基线；图例只保留方法名。"""
    wavenumbers, intensity = read_arc_data(sample_path)
    values = intensity.astype(np.float64)
    valid_mask = build_valid_mask(wavenumbers, BAD_BANDS)
    figure, panel = plt.subplots(figsize=FIGURE_SIZE)
    panel.plot(wavenumbers, values, color=RAW_COLOR, linewidth=RAW_WIDTH)
    for label, method, color in METHODS:
        baseline = estimate_baseline(
            values,
            method=method,
            lam=BASELINE_LAMBDA,
            p=BASELINE_P,
            niter=BASELINE_ITERATIONS,
            valid_mask=valid_mask,
        )
        panel.plot(wavenumbers, baseline, color=color, linewidth=METHOD_WIDTH, label=label)
    panel.set_xlim(*X_LIMITS)
    panel.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=LABEL_SIZE)
    panel.set_ylabel("intensity", fontsize=LABEL_SIZE)
    panel.legend(fontsize=TICK_SIZE, frameon=False)
    _style(panel)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="三种基线方法对比图")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE), help="单条光谱路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出图片路径")
    args = parser.parse_args()
    print("saved", build_figure(Path(args.sample), Path(args.output)))


if __name__ == "__main__":
    main()