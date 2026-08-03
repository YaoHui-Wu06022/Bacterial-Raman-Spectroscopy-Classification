"""模型输入张量的稳定规格。"""

from __future__ import annotations

from dataclasses import dataclass

from raman_temp.core.config import InputConfig


@dataclass(frozen=True)
class InputSpec:
    """描述确定性光谱输入处理与模型张量形状。"""

    in_channels: int
    point_count: int
    norm_method: str = "snv"
    smooth_enable: bool = False
    d1_enable: bool = False
    smooth_window: int = 15
    d1_window: int = 15
    delta: float = 1.0


def build_input_spec(input_config: InputConfig) -> InputSpec:
    """从输入配置构建模型和数据层共用的张量规格。"""
    point_count = int(input_config.target_points)
    if point_count < 2:
        raise ValueError("target_points 必须至少为 2")
    in_channels = 1 + int(input_config.smooth_use) + int(input_config.d1_use)
    return InputSpec(
        in_channels=in_channels,
        point_count=point_count,
        norm_method=str(input_config.norm_method),
        smooth_enable=bool(input_config.smooth_use),
        d1_enable=bool(input_config.d1_use),
        smooth_window=int(input_config.win_smooth),
        d1_window=int(input_config.win1),
        delta=(float(input_config.cut_max) - float(input_config.cut_min))
        / (point_count - 1),
    )
