"""生成白底粗箭头坐标轴的同属堆叠光谱图（四色示意）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ramanv2.common.arc_data import read_arc_data  # noqa: E402
from ramanv2.common.naming import build_natural_key  # noqa: E402
from ramanv2.data.config import resolve_build_config  # noqa: E402
from ramanv2.spectra.preprocess import fit_baseline_spectrum  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "病原菌" / "init" / "Klebsiella"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "spectrum_stacked"
FOLDER_PREFIX = "KOX"
DISPLAY_MIN, DISPLAY_MAX = 300.0, 2000.0
CURVE_COLORS = ("#D62728", "#E08214", "#2E9E8F", "#1C60CD")
STACK_STEP = 1.35
EXTRA_GAP = 0.7
CURVE_WIDTH = 1.6
SMOOTH_WINDOW = 11
ARROW_WIDTH = 2.6
ARROW_SCALE = 30
LABEL_SIZE = 21
LABEL_FONT = "Times New Roman"
ELLIPSIS_SIZE = 30

FIGURE_SIZE = (7.8, 8.6)
FIGURE_DPI = 220


def _load_curves(data_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """每个 KOX 批次取第一条谱，返回文件名、波数与去基线后的强度。"""
    config = resolve_build_config(None)
    curves = []
    for folder in sorted(data_dir.glob(f"{FOLDER_PREFIX}*")):
        if not folder.is_dir():
            continue
        files = sorted(
            (path for path in folder.glob("*.arc_data") if path.is_file()),
            key=lambda path: build_natural_key(path.name),
        )
        if not files:
            continue
        wavenumbers, raw = read_arc_data(files[0])
        fitted, corrected, _cleaned, _baseline, _stats = fit_baseline_spectrum(
            wavenumbers,
            raw,
            baseline_fit_min=config.baseline_fit_min,
            baseline_fit_max=config.baseline_fit_max,
            bad_bands=(),
            baseline_lam=config.baseline_lam,
            baseline_asls_p=config.baseline_asls_p,
            baseline_max_iter=config.baseline_max_iter,
            baseline_method=config.baseline_method,
        )
        mask = (fitted >= DISPLAY_MIN) & (fitted <= DISPLAY_MAX)
        values = corrected[mask].astype(np.float64)
        values = savgol_filter(values, SMOOTH_WINDOW, 3, mode="interp")
        span = float(np.percentile(values, 95) - np.percentile(values, 5))
        curves.append((files[0].stem, fitted[mask], values / max(span, 1e-9)))
        if len(curves) == len(CURVE_COLORS):
            break
    if len(curves) < len(CURVE_COLORS):
        raise FileNotFoundError(f"{data_dir} 下可用于堆叠的 {FOLDER_PREFIX} 批次不足四条")
    return curves


def build_figure(data_dir: Path, out_dir: Path) -> Path:
    """四条同属光谱按固定步长堆叠，配粗箭头坐标轴与粗体标签。"""
    plt.rcParams["font.family"] = LABEL_FONT
    plt.rcParams["mathtext.fontset"] = "stix"
    curves = _load_curves(data_dir)
    figure, panel = plt.subplots(figsize=FIGURE_SIZE)
    figure.patch.set_facecolor("white")
    panel.set_facecolor("white")
    plotted = []
    offsets = []
    for index, (_name, wavenumbers, values) in enumerate(curves):
        offset = (len(curves) - index - 1) * STACK_STEP
        if index <= len(curves) - 2:
            offset += EXTRA_GAP
        offsets.append(offset)
        stacked = values + offset
        plotted.append(stacked)
        panel.plot(
            wavenumbers,
            stacked,
            color=CURVE_COLORS[index],
            linewidth=CURVE_WIDTH,
            solid_capstyle="round",
        )
    panel.set_xlim(DISPLAY_MIN, DISPLAY_MAX)
    panel.set_ylim(min(np.min(item) for item in plotted) - 0.3, float(np.max(plotted[0])) + 0.02)
    gap_index = len(curves) - 2
    gap_x = (DISPLAY_MIN + DISPLAY_MAX) / 2.0
    gap_column = int(np.argmin(np.abs(curves[gap_index][1] - gap_x)))
    ellipsis_y = (
        float(plotted[gap_index][gap_column]) + float(plotted[gap_index + 1][gap_column])
    ) / 2.0
    panel.text(
        gap_x,
        ellipsis_y,
        r"$\vdots$",
        ha="center",
        va="center",
        fontsize=ELLIPSIS_SIZE,
        fontweight="bold",
        color="black",
    )
    panel.axis("off")
    panel.annotate(
        "",
        xy=(1.04, -0.045),
        xytext=(-0.06, -0.045),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=ARROW_WIDTH, mutation_scale=ARROW_SCALE),
    )
    panel.annotate(
        "",
        xy=(-0.06, 1.0),
        xytext=(-0.06, -0.045),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=ARROW_WIDTH, mutation_scale=ARROW_SCALE),
    )
    panel.text(
        -0.135,
        0.5,
        "Intensity",
        rotation=90,
        transform=panel.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    panel.text(
        0.5,
        -0.115,
        "Raman Shift (cm$^{-1}$)",
        transform=panel.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.12)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "stacked_spectra.png"
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="同属四色堆叠光谱示意")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="包含同属批次文件夹的目录")
    parser.add_argument("--out-dir", default=None, help="输出目录；默认写入时间戳目录")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    output_path = build_figure(Path(args.data_dir), out_dir)
    print("saved", output_path)


if __name__ == "__main__":
    main()