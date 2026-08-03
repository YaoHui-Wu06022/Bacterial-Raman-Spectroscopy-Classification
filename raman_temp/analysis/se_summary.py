"""已保存 SE 通道缩放统计的只读摘要。"""

from __future__ import annotations

from pathlib import Path

import torch


def write_se_summary(tasks, output_path: Path) -> None:
    """汇总各分析 run 可用的 SE sidecar 到文本文件。"""
    lines = []
    for task in tasks:
        model_tag = task.level_name if task.parent_id is None else f"{task.level_name}_{task.parent_id}"
        stats_path = Path(task.run_dir) / f"{model_tag}_se_stats.pt"
        if not stats_path.is_file():
            continue
        stats = torch.load(stats_path, map_location="cpu")
        for name, values in stats.items():
            lines.append(
                f"{task.level_name} parent={task.parent_id} {name}: "
                f"samples={int(values['sample_count'])}, "
                f"mean={float(values['channel_mean'].mean()):.4f}, "
                f"std={float(values['channel_std'].mean()):.4f}"
            )
    if lines:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
