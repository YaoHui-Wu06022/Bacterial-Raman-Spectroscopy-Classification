"""评估与推理共用的模型 run 清单产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ramanv2.common.file_ops import write_json


def write_run_manifest(
    output_path: Path | str,
    mode: str,
    level_name: str,
    runs: Mapping[str, Any],
) -> None:
    """写出本次评估或推理实际使用的模型 run。"""
    payload = {"mode": mode, "target_level": level_name, "runs": dict(runs)}
    write_json(output_path, payload)


__all__ = ["write_run_manifest"]
