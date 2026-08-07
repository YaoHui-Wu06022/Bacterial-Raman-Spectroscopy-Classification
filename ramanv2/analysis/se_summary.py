"""已保存 SE 通道缩放统计的只读摘要。"""

from __future__ import annotations

from pathlib import Path

import torch


def write_se_summary(tasks, output_path: Path) -> list[str]:
    """汇总各分析 run 的 SE sidecar，并写入独立摘要文件。"""
    lines = build_se_summary_lines(tasks)
    if lines:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def build_se_summary_lines(tasks) -> list[str]:
    """读取 SE 通道缩放统计并整理为分析日志行。"""
    lines: list[str] = []
    for task in tasks:
        model_tag = task.level_name if task.parent_id is None else f"{task.level_name}_{task.parent_id}"
        stats_path = Path(task.run_dir) / f"{model_tag}_se_stats.pt"
        if not stats_path.is_file():
            continue
        stats = torch.load(stats_path, map_location="cpu")
        for name, values in stats.items():
            lines.append(
                f"{name}: mean={float(values['channel_mean'].mean()):.4f}, "
                f"std={float(values['channel_std'].mean()):.4f}, "
                f"min={float(values['channel_min'].min()):.4f}, "
                f"max={float(values['channel_max'].max()):.4f}, "
                f"samples={int(values['sample_count'])}"
            )
    return lines
