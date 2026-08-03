"""审核运行池内文件移动与提交前路径管理。"""

from __future__ import annotations

import shutil
from pathlib import Path

from raman_temp.audit.records import RawRecord


def move_stage_candidate(record: RawRecord, work_dir: Path, stage: str) -> Path:
    """将运行池 candidate 移入对应阶段目录，不修改真实数据集。"""
    target = work_dir / "delete" / stage / Path(record.rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"审核运行目录中已存在候选文件：{target}")
    shutil.move(str(record.path), str(target))
    record.path = target
    return target
