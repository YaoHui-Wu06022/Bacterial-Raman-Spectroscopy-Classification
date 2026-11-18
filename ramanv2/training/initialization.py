"""跨实验的预训练权重部分加载。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


def build_partial_checkpoint_initializer(
    model_path: Path | str,
    source_class_names: Sequence[str] | None = None,
) -> Callable[[nn.Module, Any], None]:
    """返回按形状兼容部分加载预训练权重的训练回调。

    非分类头参数只有在形状一致时才载入；分类头先按形状整体跳过，
    再按类名交集逐行拷贝，便于跨类别空间（不同层级/不同数据集）迁移。
    """
    resolved = Path(model_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少预训练权重：{resolved}")
    state = torch.load(resolved, map_location="cpu")
    source_names = tuple(source_class_names or ())

    def initialize_model(model: nn.Module, train_task: Any) -> None:
        target = model.state_dict()
        matched = {
            key: value
            for key, value in target.items()
            if key in state and state[key].shape == value.shape
        }
        model.load_state_dict(matched, strict=False)
        skipped = sorted(set(target) - set(matched))
        head_rows = 0
        head_names = tuple(getattr(train_task, "head_names", ()) or ())
        if source_names and head_names:
            source_index = {name: index for index, name in enumerate(source_names)}
            rows = [
                (target_index, source_index[name])
                for target_index, name in enumerate(head_names)
                if name in source_index
            ]
            for suffix in ("weight", "bias"):
                key = f"head.{suffix}"
                if key not in state or key not in target or state[key].shape[1:] != target[key].shape[1:]:
                    continue
                with torch.no_grad():
                    for target_index, source_index_value in rows:
                        target[key][target_index].copy_(state[key][source_index_value])
            head_rows = len(rows)
        print(
            f"[init] {resolved.name}: loaded {len(matched)}/{len(target)} tensors, "
            f"head rows matched {head_rows}/{len(head_names) if head_names else 0}"
            + (f", skipped {skipped}" if skipped else "")
        )

    return initialize_model