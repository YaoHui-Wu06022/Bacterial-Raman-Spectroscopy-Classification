"""按当前预处理口径重绘 PPT 用分阶段光谱图，沿用旧图的视觉风格。"""

from __future__ import annotations

import argparse
import json
import re
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
    PLOT_BACKGROUND,
    PLOT_GRID_COLOR,
    PLOT_SPINE_COLOR,
    PLOT_SPINE_WIDTH,
    PLOT_TICK_SIZE,
    apply_canvas,
    save_figure,
)
from ramanv2.spectra.normalize import normalize_spectrum  # noqa: E402
from ramanv2.spectra.preprocess import fit_baseline_spectrum, preprocess_single_spectrum  # noqa: E402
from ramanv2.data.runtime.augmentation import (  # noqa: E402
    augment_axis_warp,
    augment_piecewise_gain,
    augment_strong_baseline,
)

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "病原菌"
DEFAULT_SAMPLE = "Klebsiella/KP20240511_01/cell8_Area01_000.arc_data"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "ppt_fig"
LEGACY_FIGURE_NAMES = ("05_normalized.png", "06_model_input.png")

# 本套 PPT 光谱图配色：主线跟模板蓝，基线与增强线用非主色对比。
FIGURE_SIZE_DECK = (12.0, 8.0)
FIGURE_DPI_DECK = 120
PLOT_LINE_COLOR = "#1C60CD"
PLOT_BASELINE_COLOR = "#E08214"
PLOT_INPUT_RAW_COLOR = "#E08214"
PLOT_INPUT_SMOOTH_COLOR = "#4E9A51"
PLOT_AUG_WARP_COLOR = "#E08214"
PLOT_AUG_BASELINE_COLOR = "#4E9A51"
PLOT_AUG_GAIN_COLOR = "#B84A62"
PLOT_ZERO_COLOR = "#91979E"
PLOT_FAINT_COLOR = "#A8B0BA"

CUT_MIN = 400.0
CUT_MAX = 1800.0
TARGET_POINTS = 1024
WIN_SMOOTH = 11
CHANNEL_OFFSET = 4.0
# 示意图为看清差异放大了增强幅度；训练仍使用模块默认参数。
SCHEMATIC_GAIN_STD = 0.30
SCHEMATIC_WARP_ALPHA = 0.05
SCHEMATIC_WARP_BETA = 3.0

BASELINE_OPTIONS = {
    "baseline_lam": 1e5,
    "baseline_asls_p": 0.01,
    "baseline_max_iter": 15,
    "baseline_method": "airPLS",
    "baseline_fit_min": 200.0,
    "baseline_fit_max": 2000.0,
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


def _load_date_affine(dataset_dir: Path, sample_id: str) -> tuple[float, float]:
    """读取该采集日期已确认的波数轴仿射参数。"""
    match = re.search(r"(20\d{6})", sample_id)
    if match is None:
        raise ValueError(f"样本路径里没有八位日期：{sample_id}")
    report = dataset_dir / "shift_data" / match.group(1) / "shift" / "affine.json"
    if not report.is_file():
        raise FileNotFoundError(f"缺少日期仿射报告：{report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    return float(payload["scale"]), float(payload["offset"])


def build_stage_figures(dataset_dir: Path, sample_id: str, out_dir: Path) -> list[Path]:
    """生成 00-07 八个阶段图，返回写出路径。"""
    wavenumbers, raw = read_arc_data(dataset_dir / "init" / sample_id)
    bad_bands = _load_bad_bands(dataset_dir)
    fitted_w, corrected, baseline = fit_baseline_spectrum(
        wavenumbers, raw, bad_bands=bad_bands, **BASELINE_OPTIONS
    )
    # 所选谱在坏段内没有下凹，绘图时不再剔除坏段，避免出现直线过渡的“断口”。
    figure_bad_bands: tuple[tuple[float, float], ...] = ()
    target = np.linspace(CUT_MIN, CUT_MAX, TARGET_POINTS)
    axis, interpolated = preprocess_single_spectrum(
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
    )
    crop = (fitted_w >= CUT_MIN) & (fitted_w <= CUT_MAX)
    cropped_w, cropped = fitted_w[crop], corrected[crop]

    written: list[Path] = []

    scale, offset = _load_date_affine(dataset_dir, sample_id)
    before_axis = (wavenumbers - offset) / scale
    calib_target = 1001.4
    fig, ax = _new_figure()
    ax.plot(before_axis, raw, color=PLOT_FAINT_COLOR, linewidth=1.6, label="before")
    ax.plot(wavenumbers, raw, color=PLOT_LINE_COLOR, linewidth=2, label="after")
    ax.set_xlim(float(before_axis.min()), float(before_axis.max()))
    ax.legend(loc="upper right", fontsize=16, frameon=False)
    ax.text(
        0.02,
        0.04,
        f"corrected = {scale:.5f}·w + {offset:.2f}",
        transform=ax.transAxes,
        fontsize=15,
        color=PLOT_ZERO_COLOR,
    )
    zoom = ax.inset_axes([0.06, 0.55, 0.34, 0.38])
    zoom.plot(before_axis, raw, color=PLOT_FAINT_COLOR, linewidth=1.4)
    zoom.plot(wavenumbers, raw, color=PLOT_LINE_COLOR, linewidth=1.8)
    zoom.axvline(calib_target, color=PLOT_ZERO_COLOR, linewidth=1.0, linestyle="--")
    zoom.set_xlim(calib_target - 55.0, calib_target + 55.0)
    zoom.tick_params(labelsize=12)
    for spine in zoom.spines.values():
        spine.set_color(PLOT_SPINE_COLOR)
        spine.set_linewidth(1.2)
    path = out_dir / "00_shift_correction.png"
    _save(fig, path)
    written.append(path)

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

    span = float(np.percentile(interpolated, 98) - np.percentile(interpolated, 2))
    offset_step = 1.15 * span
    np.random.seed(11)
    segments = []
    left = 0
    while left < interpolated.size:
        right = min(interpolated.size, left + int(np.random.randint(90, 161)))
        segments.append((left, right))
        left = right
    augmented_warp = augment_axis_warp(
        interpolated,
        alpha=SCHEMATIC_WARP_ALPHA,
        beta=SCHEMATIC_WARP_BETA,
    )
    np.random.seed(29)
    augmented_baseline = augment_strong_baseline(interpolated)
    np.random.seed(5)
    augmented_gain = augment_piecewise_gain(interpolated, segments, SCHEMATIC_GAIN_STD)

    fig, ax = _new_figure()
    tiers = (
        (augmented_warp, PLOT_AUG_WARP_COLOR),
        (augmented_baseline, PLOT_AUG_BASELINE_COLOR),
        (augmented_gain, PLOT_AUG_GAIN_COLOR),
    )
    for tier_index, (values, color) in enumerate(tiers):
        offset = tier_index * offset_step
        ax.plot(axis, interpolated + offset, color=PLOT_FAINT_COLOR, linewidth=1.5)
        ax.plot(axis, values + offset, color=color, linewidth=2)
        ax.fill_between(axis, interpolated + offset, values + offset, color=color, alpha=0.18)
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "05_augmentation.png"
    _save(fig, path)
    written.append(path)

    normalized = normalize_spectrum(interpolated, "snv")
    fig, ax = _new_figure()
    ax.plot(axis, normalized, color=PLOT_LINE_COLOR, linewidth=2)
    ax.axhline(0.0, color=PLOT_ZERO_COLOR, linewidth=1.5, linestyle="--")
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "06_normalized.png"
    _save(fig, path)
    written.append(path)

    smooth_snv = normalize_spectrum(
        savgol_filter(interpolated, WIN_SMOOTH, 3, mode="interp"), "snv"
    )
    fig, ax = _new_figure()
    ax.plot(axis, normalized, color=PLOT_INPUT_RAW_COLOR, linewidth=2)
    ax.plot(axis, smooth_snv + CHANNEL_OFFSET, color=PLOT_INPUT_SMOOTH_COLOR, linewidth=2)
    ax.set_xlim(CUT_MIN, CUT_MAX)
    path = out_dir / "07_model_input.png"
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
        for name in LEGACY_FIGURE_NAMES:
            stale = out_dir / name
            if stale.is_file():
                stale.unlink()
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