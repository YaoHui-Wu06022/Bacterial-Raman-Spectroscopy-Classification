"""按当前预处理口径重绘 PPT 用分阶段光谱图，沿用旧图的视觉风格。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ramanv2.common.arc_data import read_arc_data  # noqa: E402
from ramanv2.common.plotting import (  # noqa: E402
    FIGURE_DPI_DECK,
    FIGURE_SIZE_DECK,
    PLOT_BACKGROUND,
    PLOT_BASELINE_COLOR,
    PLOT_FAINT_COLOR,
    PLOT_GRID_COLOR,
    PLOT_LINE_COLOR,
    PLOT_SMOOTH_COLOR,
    PLOT_SPINE_COLOR,
    PLOT_SPINE_WIDTH,
    PLOT_TICK_SIZE,
    PLOT_ZERO_COLOR,
    apply_canvas,
    save_figure,
)
from ramanv2.spectra.normalize import normalize_spectrum  # noqa: E402
from ramanv2.spectra.preprocess import fit_baseline_spectrum, preprocess_single_spectrum  # noqa: E402

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "病原菌"
DEFAULT_SAMPLE = "Klebsiella/KP20240511_01/cell8_Area01_000.arc_data"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "ppt流程图"

CUT_MIN = 400.0
CUT_MAX = 1800.0
TARGET_POINTS = 1024
WIN_SMOOTH = 11
CHANNEL_OFFSET = 4.0

BASELINE_OPTIONS = {
    "baseline_lam": 1e5,
    "baseline_asls_p": 0.01,
    "baseline_max_iter": 15,
    "baseline_method": "airPLS",
    "baseline_fit_min": 200.0,
    "baseline_fit_max": 2000.0,
    "cosmic_ray_enable": True,
    "cosmic_ray_window_points": 7,
    "cosmic_ray_threshold": 7.0,
    "cosmic_ray_max_iter": 2,
}


def _new_figure():
    """建立统一画布：白底、浅灰细网格、灰色粗边框、大号刻度。"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DECK, dpi=FIGURE_DPI_DECK)
    fig.patch.set_facecolor(PLOT_BACKGROUND)
    apply_canvas(
        ax,
        facecolor=PLOT_BACKGROUND,
        grid_color=PLOT_GRID_COLOR,
        grid_linewidth=1.0,
        spine_color=PLOT_SPINE_COLOR,
        spine_width=PLOT_SPINE_WIDTH,
        tick_size=PLOT_TICK_SIZE,
    )
    return fig, ax


def _save(fig, path: Path) -> None:
    save_figure(fig, path, dpi=FIGURE_DPI_DECK)


def _load_bad_bands(dataset_dir: Path) -> tuple[tuple[float, float], ...]:
    spec = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    return tuple(tuple(float(value) for value in band) for band in spec.get("bad_bands", ()))


def build_stage_figures(dataset_dir: Path, sample_id: str, out_dir: Path) -> list[Path]:
    """生成 01-06 七个阶段图，返回写出路径。"""
    wavenumbers, raw = read_arc_data(dataset_dir / "init" / sample_id)
    bad_bands = _load_bad_bands(dataset_dir)
    fitted_w, corrected, _cleaned, baseline, _stats = fit_baseline_spectrum(
        wavenumbers, raw, bad_bands=bad_bands, **BASELINE_OPTIONS
    )
    # 所选谱在坏段内没有下凹，绘图时不再剔除坏段，避免出现直线过渡的“断口”。
    figure_bad_bands: tuple[tuple[float, float], ...] = ()
    target = np.linspace(CUT_MIN, CUT_MAX, TARGET_POINTS)
    axis, interpolated, _ = preprocess_single_spectrum(
        wavenumbers,
        raw,
        CUT_MIN,
        CUT_MAX,
        target,
        figure_bad_bands,
        BASELINE_OPTIONS["baseline_lam"],
        BASELINE_OPTIONS["baseline_asls_p"],
        BASELINE_OPTIONS["baseline_max_iter"],
        baseline_fit_min=BASELINE_OPTIONS["baseline_fit_min"],
        baseline_fit_max=BASELINE_OPTIONS["baseline_fit_max"],
        baseline_method=BASELINE_OPTIONS["baseline_method"],
        cosmic_ray_enable=BASELINE_OPTIONS["cosmic_ray_enable"],
        cosmic_ray_window_points=BASELINE_OPTIONS["cosmic_ray_window_points"],
        cosmic_ray_threshold=BASELINE_OPTIONS["cosmic_ray_threshold"],
        cosmic_ray_max_iter=BASELINE_OPTIONS["cosmic_ray_max_iter"],
    )
    crop = (fitted_w >= CUT_MIN) & (fitted_w <= CUT_MAX)
    cropped_w, cropped = fitted_w[crop], corrected[crop]

    written: list[Path] = []

    fig, ax = _new_figure()
    ax.plot(wavenumbers, raw, color=PLOT_LINE_COLOR, linewidth=2)
    ax.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    path = out_dir / "01_raw.png"
    _save(fig, path)
    written.append(path)

    fig, ax = _new_figure()
    ax.plot(wavenumbers, raw, color=PLOT_LINE_COLOR, linewidth=2)
    ax.plot(fitted_w, baseline, color=PLOT_BASELINE_COLOR, linewidth=2)
    ax.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    path = out_dir / "01_raw_with_baseline.png"
    _save(fig, path)
    written.append(path)

    fig, ax = _new_figure()
    ax.plot(fitted_w, corrected, color=PLOT_LINE_COLOR, linewidth=2)
    ax.axhline(0.0, color=PLOT_ZERO_COLOR, linewidth=1.5, linestyle="--")
    ax.set_xlim(float(fitted_w.min()), float(fitted_w.max()))
    path = out_dir / "02_baseline_removed.png"
    _save(fig, path)
    written.append(path)

    fig, ax = _new_figure()
    ax.plot(cropped_w, cropped, color=PLOT_LINE_COLOR, linewidth=2)
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "03_cut.png"
    _save(fig, path)
    written.append(path)

    fig, ax = _new_figure()
    ax.plot(cropped_w, cropped, color=PLOT_FAINT_COLOR, linewidth=1.5)
    ax.plot(axis, interpolated, color=PLOT_LINE_COLOR, linewidth=2)
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "04_interpolated.png"
    _save(fig, path)
    written.append(path)

    normalized = normalize_spectrum(interpolated, "snv")
    fig, ax = _new_figure()
    ax.plot(axis, normalized, color=PLOT_LINE_COLOR, linewidth=2)
    ax.axhline(0.0, color=PLOT_ZERO_COLOR, linewidth=1.5, linestyle="--")
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "05_normalized.png"
    _save(fig, path)
    written.append(path)

    smooth_snv = normalize_spectrum(
        savgol_filter(interpolated, WIN_SMOOTH, 3, mode="interp"), "snv"
    )
    fig, ax = _new_figure()
    ax.plot(axis, normalized, color=PLOT_LINE_COLOR, linewidth=2)
    ax.plot(axis, smooth_snv + CHANNEL_OFFSET, color=PLOT_SMOOTH_COLOR, linewidth=2)
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "06_model_input.png"
    _save(fig, path)
    written.append(path)

    return written


def publish_figures(out_dir: Path, sample_id: str) -> None:
    """先在临时目录生成并校验，再覆盖发布到目标目录。"""
    staging = Path(tempfile.mkdtemp(prefix="ppt_figs_"))
    try:
        written = build_stage_figures(DEFAULT_DATASET_DIR, sample_id, staging)
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in written:
            shutil.copy2(path, out_dir / path.name)
        print("published", len(written), "figures to", out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="重绘 PPT 分阶段光谱图")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE, help="init 下的光谱相对路径")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    args = parser.parse_args()
    publish_figures(Path(args.out_dir), args.sample_id)


if __name__ == "__main__":
    main()