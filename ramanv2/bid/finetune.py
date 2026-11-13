"""BID 第二阶段：从第一阶段权重初始化。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn


def build_checkpoint_initializer(
    model_path: Path | str,
) -> Callable[[nn.Module, Any], None]:
    """返回把 model.pt 权重严格载入新模型的训练回调。"""
    resolved = Path(model_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少第一阶段权重：{resolved}")

    def initialize_model(model: nn.Module, _train_task: Any) -> None:
        state = torch.load(resolved, map_location="cpu")
        model.load_state_dict(state, strict=True)

    return initialize_model
