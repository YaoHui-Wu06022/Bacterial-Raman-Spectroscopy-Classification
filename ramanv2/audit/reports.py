"""审核 CSV、JSON 与运行目录的落盘工具。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


DELTA_FIELDS = ("genus", "folder", "prefix", "delta")
DELTA_LOG_FIELDS = (
    "time",
    "genus",
    "folder",
    "prefix",
    "step_delta",
    "cumulative_delta",
    "files_changed",
    "note",
)


def build_audit_run_dir(dataset_dir: Path) -> Path:
    """在数据集内创建本次审核运行目录，保存报告和临时数据池。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dataset_dir / "audit_runs" / f"{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_stage_dir(run_dir: Path, stage: str) -> Path:
    """创建一个阶段报告目录并返回其路径。"""
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    return stage_dir


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    """以 UTF-8 BOM 写入结构稳定的审核 CSV，即使没有记录也保留表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """以 UTF-8 写入单个审核运行或阶段的元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_delta(path: Path) -> dict[tuple[str, str], float]:
    """读取数据集根目录的文件夹累计平移记录。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row["genus"], row["folder"]): float(row["delta"])
            for row in csv.DictReader(file, delimiter="\t")
            if row.get("genus") and row.get("folder") and row.get("delta")
        }


def write_delta(path: Path, values: dict[tuple[str, str], tuple[str, float]]) -> None:
    """写入非零累计平移，保持属、文件夹和前缀可独立反查。"""
    rows = [
        {"genus": genus, "folder": folder, "prefix": prefix, "delta": f"{delta:+g}"}
        for (genus, folder), (prefix, delta) in sorted(values.items())
        if abs(delta) > 1e-9
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def append_delta_log(path: Path, rows: list[dict[str, object]]) -> None:
    """在 delta 日志末尾追加本次平移步骤，不重写历史记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_existing = path.is_file()
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_LOG_FIELDS, delimiter="\t")
        if not is_existing:
            writer.writeheader()
        writer.writerows(rows)
