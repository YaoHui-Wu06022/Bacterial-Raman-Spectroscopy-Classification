"""评估与推理共用的模型 run 清单产物。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_run_manifest(
    output_path: Path | str,
    mode: str,
    level_name: str,
    runs: Mapping[str, Any],
) -> None:
    """写出本次评估或推理实际使用的模型 run。"""
    payload = {"mode": mode, "target_level": level_name, "runs": dict(runs)}
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["write_run_manifest"]
