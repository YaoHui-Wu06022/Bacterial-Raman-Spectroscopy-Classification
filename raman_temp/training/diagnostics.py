"""训练数值异常的有限摘要写入。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


def write_nonfinite_diagnostic(
    diagnostic_written_enable: bool,
    diagnostic_path: Path,
    stage: str,
    epoch: int,
    batch_index: int,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    sample_paths: Sequence[str],
    tensors: dict[str, torch.Tensor] | None = None,
    losses: dict[str, torch.Tensor] | None = None,
) -> bool:
    """写入首个数值异�?batch 的缩�?JSON 摘要。"""
    if diagnostic_written_enable:
        return True
    record: dict[str, Any] = {
        "stage": stage,
        "epoch": epoch,
        "batch": batch_index,
        "sample_paths": [str(path) for path in sample_paths],
        "labels": labels.detach().cpu().tolist(),
        "input": _summarize_tensor(inputs),
    }
    if tensors:
        record["tensors"] = {
            name: _summarize_tensor(values)
            for name, values in tensors.items()
        }
    if losses:
        record["losses"] = {
            name: _loss_value(values)
            for name, values in losses.items()
            if values.ndim == 0
        }
    diagnostic_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def _summarize_tensor(values: torch.Tensor) -> dict[str, Any]:
    """提取张量形状、有限元素数量和有限绝对值最大值。"""
    data = values.detach()
    finite_mask = torch.isfinite(data)
    summary: dict[str, Any] = {
        "shape": list(data.shape),
        "finite_count": int(finite_mask.sum()),
        "total_count": int(data.numel()),
    }
    if finite_mask.any():
        summary["finite_abs_max"] = float(data[finite_mask].abs().max().cpu())
    return summary


def _loss_value(values: torch.Tensor) -> float:
    """将标量损失转换为可写�?JSON 的浮点数。"""
    return float(values.detach().cpu())
