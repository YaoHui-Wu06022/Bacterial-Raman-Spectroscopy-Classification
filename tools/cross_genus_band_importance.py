"""跨菌属平均波段归因图。

修改 ANALYSIS_DIR 后直接运行：
    python tools/cross_genus_band_importance.py

输出写到 ANALYSIS_DIR/figures：
- cross_genus_mean_band_importance.png
- cross_genus_mean_band_importance.csv
- cross_genus_top_peaks.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ramanv2.common.plotting import configure_matplotlib_fonts  # noqa: E402


ANALYSIS_DIR = (
    REPO_ROOT
    / "output"
    / "病原菌"
    / "400-1800区间"
    / "20261117_213047_genus_42"
    / "level_1"
    / "analysis_result"
)

FIGURE_NAME = "cross_genus_mean_band_importance.png"
STATS_NAME = "cross_genus_mean_band_importance.csv"
PEAKS_NAME = "cross_genus_top_peaks.csv"

FIGURE_SIZE = (12, 4.6)
DPI = 300
MEAN_COLOR = "#4C72B0"
PEAK_COLOR = "#C44E52"
IQR_ALPHA = 0.18
BAD_BAND_COLOR = "#d9d9d9"
BAD_BAND_ALPHA = 0.35
GAP_FACTOR = 1.5
PEAK_MIN_MEAN = 0.20
PEAK_MIN_PROMINENCE = 0.10
CONSENSUS_THRESHOLD = 0.5
XTICK_STEP = 200
Y_MAX = 0.80


def load_matrix(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取逐类 CSV，返回类别矩阵、波数轴和归因矩阵。"""
    frame = pd.read_csv(csv_path)
    matrix = frame.pivot(index="class", columns="band_index", values="importance").sort_index()
    wavenumbers = (
        frame.drop_duplicates("band_index")
        .sort_values("band_index")["wavenumber"]
        .to_numpy(dtype=float)
    )
    return matrix, wavenumbers, matrix.to_numpy(dtype=float)


def find_gaps(wavenumbers: np.ndarray) -> list[tuple[int, int]]:
    """按波数间距识别坏区，返回左右相邻点的索引对。"""
    step = float(np.median(np.diff(wavenumbers)))
    indices = np.where(np.diff(wavenumbers) > step * GAP_FACTOR)[0]
    return [(int(index), int(index) + 1) for index in indices]


def build_segments(length: int, gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """把波数轴按坏区拆成连续段，避免跨坏区连线。"""
    segments = []
    start = 0
    for left, _right in gaps:
        segments.append((start, left + 1))
        start = left + 1
    segments.append((start, length))
    return segments


def build_statistics(values: np.ndarray, wavenumbers: np.ndarray) -> pd.DataFrame:
    """计算跨类均值、分位数与共识计数。"""
    return pd.DataFrame(
        {
            "wavenumber": wavenumbers,
            "mean": values.mean(axis=0),
            "median": np.median(values, axis=0),
            "q25": np.percentile(values, 25, axis=0),
            "q75": np.percentile(values, 75, axis=0),
            "std": values.std(axis=0),
            "n_ge_0_5": (values >= CONSENSUS_THRESHOLD).sum(axis=0),
        }
    )


def build_peak_table(
    matrix: pd.DataFrame,
    values: np.ndarray,
    wavenumbers: np.ndarray,
    peaks: np.ndarray,
    prominences: np.ndarray,
    edge_ids: set[int],
) -> pd.DataFrame:
    """生成排除坏区边界后的峰值表。"""
    rows = []
    for position, peak in enumerate(peaks):
        index = int(peak)
        if index in edge_ids:
            continue
        top = sorted(zip(matrix.index, values[:, index]), key=lambda item: -item[1])[:4]
        rows.append(
            {
                "wavenumber": float(wavenumbers[index]),
                "mean": float(values[:, index].mean()),
                "prominence": float(prominences[position]),
                "n_ge_0_5": int((values[:, index] >= CONSENSUS_THRESHOLD).sum()),
                "top_genera": "; ".join(f"{name}:{value:.3f}" for name, value in top),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def write_figure(
    output_path: Path,
    wavenumbers: np.ndarray,
    mean: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    gaps: list[tuple[int, int]],
    peaks: np.ndarray,
    prominences: np.ndarray,
    edge_ids: set[int],
) -> None:
    """绘制跨属均值与 IQR，断开坏区并标注有效峰。"""
    configure_matplotlib_fonts()
    segments = build_segments(len(wavenumbers), gaps)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for left, right in gaps:
        axis.axvspan(
            wavenumbers[left],
            wavenumbers[right],
            color=BAD_BAND_COLOR,
            alpha=BAD_BAND_ALPHA,
            linewidth=0,
            zorder=0,
        )
    for start, stop in segments:
        x = wavenumbers[start:stop]
        axis.fill_between(
            x,
            q25[start:stop],
            q75[start:stop],
            color=MEAN_COLOR,
            alpha=IQR_ALPHA,
            linewidth=0,
            label="IQR" if start == 0 else None,
        )
        axis.plot(
            x,
            mean[start:stop],
            color=MEAN_COLOR,
            lw=1.6,
            label="Mean" if start == 0 else None,
        )
    for position, peak in enumerate(peaks):
        index = int(peak)
        if index in edge_ids:
            continue
        if mean[index] < PEAK_MIN_MEAN or prominences[position] < PEAK_MIN_PROMINENCE:
            continue
        axis.scatter([wavenumbers[index]], [mean[index]], s=18, color=PEAK_COLOR, zorder=4)
        axis.annotate(
            f"{wavenumbers[index]:.0f}",
            (wavenumbers[index], mean[index]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color="#333333",
        )
    axis.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    axis.set_ylim(0, Y_MAX)
    axis.set_xticks(np.arange(np.ceil(wavenumbers.min() / XTICK_STEP) * XTICK_STEP, wavenumbers.max() + 1, XTICK_STEP))
    axis.set_xlabel("Wavenumber (cm$^{-1}$)")
    axis.set_ylabel("Mean normalized importance")
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    figure_dir = ANALYSIS_DIR / "figures"
    csv_path = figure_dir / "band_importance_per_class.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"缺少逐类归因 CSV：{csv_path}")

    matrix, wavenumbers, values = load_matrix(csv_path)
    if len(wavenumbers) != values.shape[1]:
        raise ValueError("波数轴与归因矩阵列数不一致")

    stats = build_statistics(values, wavenumbers)
    stats.to_csv(figure_dir / STATS_NAME, index=False, encoding="utf-8")

    gaps = find_gaps(wavenumbers)
    edge_ids = {index for gap in gaps for index in gap}
    mean = stats["mean"].to_numpy()
    peaks, properties = find_peaks(mean, prominence=0.02, distance=12)
    prominences = properties["prominences"]
    peak_table = build_peak_table(
        matrix,
        values,
        wavenumbers,
        peaks,
        prominences,
        edge_ids,
    )
    peak_table.to_csv(figure_dir / PEAKS_NAME, index=False, encoding="utf-8")

    write_figure(
        figure_dir / FIGURE_NAME,
        wavenumbers,
        mean,
        stats["q25"].to_numpy(),
        stats["q75"].to_numpy(),
        gaps,
        peaks,
        prominences,
        edge_ids,
    )
    print(f"figure={figure_dir / FIGURE_NAME}")
    print(f"stats={figure_dir / STATS_NAME}")
    print(f"peaks={figure_dir / PEAKS_NAME}")


if __name__ == "__main__":
    main()
