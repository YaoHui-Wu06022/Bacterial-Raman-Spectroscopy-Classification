"""生成正弦位置编码热图（模型维度 × 输入序列长度），用于 PPT 展示。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ramanv2.core.config import InputConfig, ModelConfig  # noqa: E402
from ramanv2.core.input_spec import build_input_spec  # noqa: E402
from ramanv2.modeling.classifier import RamanClassifier1D  # noqa: E402
from ramanv2.modeling.layers import PositionalEncoding1D  # noqa: E402
from ramanv2.modeling.model_spec import build_model_spec  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "ppt流程图"
FILE_NAME = "positional_encoding_slide.png"

FIGURE_SIZE = (7.6, 7.4)
FIGURE_DPI = 130
# 幻灯片只展示前若干位置，保证条纹和单元格可读；编码本身仍按真实维度生成。
POSITION_LIMIT = 64


def _resolve_sequence_length(input_spec) -> int:
    """让真实模型跑一次前向，得到 Transformer 实际看到的位置数。"""
    model_spec = build_model_spec(ModelConfig(), input_spec)
    model = RamanClassifier1D(2, model_spec)
    with torch.no_grad():
        features = model.forward_features(
            torch.zeros(1, input_spec.in_channels, input_spec.point_count)
        )
    return int(features.size(-1))


def build_figure(out_dir: Path) -> Path:
    """按模型实际编码维度与主干输出位置数绘制 position × dimension 热图。"""
    input_spec = build_input_spec(InputConfig())
    position_count = _resolve_sequence_length(input_spec)
    rows = min(position_count, POSITION_LIMIT)
    encoding = PositionalEncoding1D(ModelConfig().transformer_dim).pe[0].numpy()[:rows]
    figure, panel = plt.subplots(figsize=FIGURE_SIZE)
    image = panel.imshow(encoding, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="upper")
    colorbar = figure.colorbar(image, ax=panel, fraction=0.04, pad=0.02)
    colorbar.ax.tick_params(labelsize=9)
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / FILE_NAME
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="位置编码热图")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    args = parser.parse_args()
    print("saved", build_figure(Path(args.out_dir)))


if __name__ == "__main__":
    main()