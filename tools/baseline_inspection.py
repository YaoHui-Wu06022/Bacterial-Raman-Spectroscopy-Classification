"""对照 KOX 批次原始谱与当前拟合基线；只取原生轴覆盖 200–2000 的批次。"""

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

from ramanv2.analysis.spectrum_viewer import prepare_single_spectrum  # noqa: E402
from ramanv2.common.arc_data import read_arc_data  # noqa: E402
from ramanv2.common.naming import build_natural_key  # noqa: E402
from ramanv2.core.config import InputConfig  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "病原菌" / "init" / "Klebsiella"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "baseline_inspection"
FOLDER_PREFIX = "KOX"
X_LIMITS = (200.0, 2000.0)
COVER_MIN, COVER_MAX = 250.0, 1950.0
PANEL_COUNT = 16
PROFILE_ID = "PB"

FIGURE_DPI = 130
FIGURE_SIZE = (17.0, 12.0)
COLUMNS = 4
RAW_COLOR = "#4C72B0"
BASELINE_COLOR = "#DD8452"
GRID_COLOR = "#EDEDED"


def _covers_axis(path: Path) -> bool:
    """判断光谱原生轴是否覆盖到 200–2000 的两端附近。"""
    wavenumbers, _ = read_arc_data(path)
    if not wavenumbers.size:
        return False
    return float(wavenumbers.min()) <= COVER_MIN and float(wavenumbers.max()) >= COVER_MAX


def _collect_samples(data_dir: Path, panel_count: int) -> list[Path]:
    """收集覆盖目标轴范围的批次光谱，均匀抽取用于绘图的样本。"""
    files: list[Path] = []
    for folder in sorted(data_dir.glob(f"{FOLDER_PREFIX}*")):
        if not folder.is_dir():
            continue
        folder_files = sorted(
            (path for path in folder.glob("*.arc_data") if path.is_file()),
            key=lambda path: build_natural_key(path.name),
        )
        if not folder_files or not _covers_axis(folder_files[0]):
            continue
        files.extend(folder_files)
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有轴范围覆盖 200–2000 的 {FOLDER_PREFIX} 批次")
    if len(files) <= panel_count:
        return files
    indices = np.linspace(0, len(files) - 1, panel_count).round().astype(int)
    return [files[index] for index in dict.fromkeys(indices.tolist())]


def _window_y_limits(raw_wavenumbers, raw_intensity, baseline_wavenumbers, baseline):
    """按横轴范围内的原始谱与基线取值计算纵轴。"""
    lower, upper = X_LIMITS
    raw_mask = (raw_wavenumbers >= lower) & (raw_wavenumbers <= upper)
    base_mask = (baseline_wavenumbers >= lower) & (baseline_wavenumbers <= upper)
    values = np.concatenate([raw_intensity[raw_mask], baseline[base_mask]])
    low, high = float(np.min(values)), float(np.max(values))
    padding = max((high - low) * 0.08, 1.0)
    return low - padding, high + padding


def build_figure(data_dir: Path, out_dir: Path) -> Path:
    """多条 KOX 光谱排成网格：每条叠加原始谱与当前拟合基线。"""
    samples = _collect_samples(data_dir, PANEL_COUNT)
    rows = int(np.ceil(len(samples) / COLUMNS))
    figure, axes = plt.subplots(rows, COLUMNS, figsize=FIGURE_SIZE, squeeze=False)
    lower, upper = X_LIMITS
    for index, sample_path in enumerate(samples):
        panel = axes[index // COLUMNS][index % COLUMNS]
        prepared = prepare_single_spectrum(sample_path, InputConfig(), profile_id=PROFILE_ID)
        panel.plot(
            prepared.raw_wavenumbers,
            prepared.raw_intensity,
            color=RAW_COLOR,
            linewidth=1.0,
            label="raw",
        )
        panel.plot(
            prepared.baseline_wavenumbers,
            prepared.baseline,
            color=BASELINE_COLOR,
            linewidth=1.8,
            label="fitted baseline",
        )
        panel.set_title(f"{sample_path.parent.name}\n{sample_path.stem}", fontsize=8, loc="left")
        panel.grid(True, color=GRID_COLOR, linewidth=0.8)
        panel.set_xlim(lower, upper)
        panel.set_ylim(
            *_window_y_limits(
                prepared.raw_wavenumbers,
                prepared.raw_intensity,
                prepared.baseline_wavenumbers,
                prepared.baseline,
            )
        )
        if index == 0:
            panel.legend(loc="upper right", fontsize=9)
        if index % COLUMNS == 0:
            panel.set_ylabel("intensity", fontsize=10)
        if index // COLUMNS == rows - 1:
            panel.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10)
    for empty_index in range(len(samples), rows * COLUMNS):
        axes[empty_index // COLUMNS][empty_index % COLUMNS].axis("off")
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "baseline_raw_vs_fit.png"
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="KOX 原始谱与拟合基线对照图")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="包含 KOX 批次文件夹的目录")
    parser.add_argument("--out-dir", default=None, help="输出目录；默认写入时间戳目录")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    output_path = build_figure(Path(args.data_dir), out_dir)
    print("saved", output_path)


if __name__ == "__main__":
    main()
