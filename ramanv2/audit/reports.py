"""共享清洗阶段的报告写入。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ramanv2.audit.records import CleanRecord


def build_stage_dir(run_dir: Path, stage: str) -> Path:
    """创建一个阶段报告目录并返回该目录。"""
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    return stage_dir


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    """以 UTF-8 BOM 写入字段稳定的清洗 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """以 UTF-8 写入单个清洗阶段或运行摘要。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_raw_rows(records: list[CleanRecord]) -> list[dict[str, object]]:
    """将共享 Stage 1 记录转换为不含标签的通用 CSV 行。"""
    return [
        {
            "state": record.state,
            "reasons": ";".join(record.reasons),
            "rel_path": record.rel_path,
            "folder": record.folder,
            "points": record.points,
            "coverage": f"{record.coverage:.6f}" if np.isfinite(record.coverage) else "",
            "malformed_lines": record.malformed_lines,
            "longest_flat_points": record.longest_flat_points,
            "saturation_points": record.saturation_points,
            "noise_ratio": f"{record.noise_ratio:.8f}" if np.isfinite(record.noise_ratio) else "",
        }
        for record in records
    ]


def build_similarity_rows(records: list[CleanRecord]) -> list[dict[str, object]]:
    """将共享近邻记录转换为不含标签的通用 CSV 行。"""
    return [
        {
            "state": record.state,
            "reasons": ";".join(record.reasons),
            "rel_path": record.rel_path,
            "folder": record.folder,
            "reference_count": record.reference_count,
            "neighbor_count": record.neighbor_count,
            "neighbor_corr": f"{record.neighbor_corr:.6f}" if np.isfinite(record.neighbor_corr) else "",
            "rmse": f"{record.rmse:.6f}" if np.isfinite(record.rmse) else "",
        }
        for record in records
    ]
