"""绘制 focal loss 与普通交叉熵的对比图，用于 PPT 展示。"""

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

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "ppt流程图"
FILE_NAME = "focal_loss_slide.png"

FIGURE_SIZE = (8.0, 5.6)
FIGURE_DPI = 130
GAMMAS = (0.5, 1.0, 1.2, 2.0)
COLORS = ("#7B8794", "#4E9A51", "#1C60CD", "#B84A62")
CE_COLOR = "#5F6368"
GRID_COLOR = "#EDEDED"
SPINE_COLOR = "#5F6368"
TICK_SIZE = 14
LABEL_SIZE = 15
LINE_WIDTH = 2.4

PROBABILITIES = np.linspace(0.001, 1.0, 600)


def _style(panel) -> None:
    panel.grid(True, color=GRID_COLOR, linewidth=1.0)
    for spine in panel.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.6)
    panel.tick_params(labelsize=TICK_SIZE, width=1.4, length=5)


def build_figure(out_dir: Path) -> Path:
    """绘制各 γ 的 focal loss 曲线与普通交叉熵的对比。"""
    figure, panel = plt.subplots(figsize=FIGURE_SIZE)
    cross_entropy = -np.log(PROBABILITIES)
    panel.plot(
        PROBABILITIES,
        cross_entropy,
        color=CE_COLOR,
        linewidth=LINE_WIDTH,
        linestyle="--",
        label="Cross entropy",
    )
    for gamma, color in zip(GAMMAS, COLORS):
        factor = np.power(1.0 - PROBABILITIES, gamma)
        panel.plot(
            PROBABILITIES,
            factor * cross_entropy,
            color=color,
            linewidth=LINE_WIDTH,
            label=f"Focal loss  γ={gamma:g}",
        )
    panel.set_xlabel("probability of true class  $p_t$", fontsize=LABEL_SIZE)
    panel.set_ylabel("loss", fontsize=LABEL_SIZE)
    panel.set_xlim(0.0, 1.0)
    panel.set_ylim(0.0, 4.0)
    panel.legend(fontsize=12, frameon=False)
    _style(panel)
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / FILE_NAME
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="focal loss 对比图")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    args = parser.parse_args()
    print("saved", build_figure(Path(args.out_dir)))


if __name__ == "__main__":
    main()