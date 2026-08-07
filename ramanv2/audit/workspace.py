"""共享清洗运行池与统一提交。"""

from __future__ import annotations

import shutil
from pathlib import Path

from ramanv2.audit.records import CleanRecord


def build_work_dir(run_dir: Path, source_dir: Path) -> Path:
    """复制输入目录到运行池，隔离阶段候选移动。"""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"缺少清洗输入目录：{source_dir}")
    work_dir = run_dir / "work"
    if work_dir.exists():
        raise FileExistsError(f"清洗运行目录已包含 work：{work_dir}")
    shutil.copytree(source_dir, work_dir / "init")
    return work_dir


def move_stage_candidate(record: CleanRecord, work_dir: Path, stage: str) -> None:
    """将运行池候选移入阶段目录，不修改真实输入目录。"""
    target = work_dir / "delete" / stage / record.rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"清洗运行目录中已存在候选文件：{target}")
    shutil.move(str(record.path), str(target))
    record.path = target


def build_candidate_rows(work_dir: Path) -> list[dict[str, str]]:
    """从运行池阶段目录收集待统一提交的候选相对路径。"""
    rows = []
    delete_dir = work_dir / "delete"
    stage_dirs = sorted(path for path in delete_dir.iterdir() if path.is_dir()) if delete_dir.is_dir() else []
    for stage_dir in stage_dirs:
        for path in sorted(stage_dir.rglob("*.arc_data")):
            rows.append(
                {"stage": stage_dir.name, "rel_path": path.relative_to(stage_dir).as_posix()}
            )
    return rows


def commit_candidate_rows(rows: list[dict[str, str]], dataset_dir: Path, init_name: str) -> int:
    """在所有阶段成功后，将候选从真实 init 统一移入 delete。"""
    moved_count = 0
    for row in rows:
        source = dataset_dir / init_name / row["rel_path"]
        target = dataset_dir / "delete" / row["stage"] / row["rel_path"]
        if not source.is_file():
            raise FileNotFoundError(f"提交时找不到清洗 candidate：{source}")
        if target.exists():
            raise FileExistsError(f"清洗 candidate 提交目标已存在：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved_count += 1
    return moved_count
