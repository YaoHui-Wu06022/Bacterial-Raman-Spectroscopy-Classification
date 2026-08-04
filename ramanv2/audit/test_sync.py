"""测试菌运行池的复制和来源映射。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ramanv2.audit.reports import write_csv
from ramanv2.common.naming import is_test_source_folder, parse_folder_prefix, parse_test_folder_prefix


TEST_POOL_FIELDS = (
    "source_folder",
    "source_file",
    "target_genus",
    "target_folder",
    "target_file",
)
COMMIT_FIELDS = ("stage", "source_dataset", "source_rel_path", "work_rel_path")


@dataclass(frozen=True)
class TestPoolResult:
    """记录测试菌运行池路径、复制谱数量和未映射目录数量。"""

    work_dir: Path
    pool_init_dir: Path
    test_init_dir: Path
    copied_count: int
    skipped_count: int


def build_audit_folder_name(source_folder: str) -> str:
    """构造运行池内可反查测试菌来源的临时文件夹名称。"""
    prefix = parse_test_folder_prefix(source_folder)
    return f"{prefix}__{source_folder}__audit"


def parse_audit_source_folder(folder: str) -> str | None:
    """从运行池临时文件夹名称解析测试菌来源目录。"""
    parts = str(folder).split("__")
    if len(parts) != 3 or parts[-1] != "audit" or not is_test_source_folder(parts[1]):
        return None
    return parts[1]


def build_genus_by_prefix(init_dir: Path) -> dict[str, str]:
    """从常规样本目录建立类别前缀到唯一属目录的映射。"""
    candidates: dict[str, set[str]] = {}
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            if folder_dir.name.lower().endswith("t") or parse_audit_source_folder(folder_dir.name):
                continue
            prefix = parse_folder_prefix(folder_dir.name, uppercase_enable=True)
            candidates.setdefault(prefix, set()).add(genus_dir.name)
    return {
        prefix: next(iter(genera))
        for prefix, genera in candidates.items()
        if len(genera) == 1
    }


def build_work_pool(
    run_dir: Path,
    dataset_init_dir: Path,
    test_source_init_dir: Path,
) -> TestPoolResult:
    """复制常规谱和测试菌谱到运行目录，生成供 Stage 使用的联合数据池。"""
    if not dataset_init_dir.is_dir():
        raise FileNotFoundError(f"缺少主数据集 init：{dataset_init_dir}")
    if not test_source_init_dir.is_dir():
        raise FileNotFoundError(f"缺少测试菌 init：{test_source_init_dir}")

    work_dir = run_dir / "work"
    pool_init_dir = work_dir / "pool" / "init"
    test_init_dir = work_dir / "test" / "init"
    if work_dir.exists():
        raise FileExistsError(f"审核运行目录已包含 work：{work_dir}")
    shutil.copytree(dataset_init_dir, pool_init_dir)
    shutil.copytree(test_source_init_dir, test_init_dir)

    genus_by_prefix = build_genus_by_prefix(pool_init_dir)
    rows: list[dict[str, object]] = []
    copied_count = 0
    skipped_count = 0
    for source_dir in sorted(path for path in test_init_dir.iterdir() if path.is_dir() and is_test_source_folder(path.name)):
        prefix = parse_test_folder_prefix(source_dir.name)
        genus = genus_by_prefix.get(prefix)
        if genus is None:
            skipped_count += 1
            continue
        target_folder = build_audit_folder_name(source_dir.name)
        target_dir = pool_init_dir / genus / target_folder
        shutil.copytree(source_dir, target_dir)
        for source_path in sorted(source_dir.glob("*.arc_data")):
            rows.append(
                {
                    "source_folder": source_dir.name,
                    "source_file": source_path.name,
                    "target_genus": genus,
                    "target_folder": target_folder,
                    "target_file": source_path.name,
                }
            )
            copied_count += 1
    write_csv(run_dir / "test_pool_manifest.csv", rows, TEST_POOL_FIELDS)
    return TestPoolResult(work_dir, pool_init_dir, test_init_dir, copied_count, skipped_count)


def build_candidate_commit_rows(work_dir: Path) -> list[dict[str, str]]:
    """将运行池阶段候选解析为真实数据集可提交的来源路径。"""
    rows = []
    delete_dir = work_dir / "delete"
    for stage_dir in sorted(path for path in delete_dir.iterdir() if path.is_dir()) if delete_dir.is_dir() else []:
        for path in sorted(stage_dir.rglob("*.arc_data")):
            relative = path.relative_to(stage_dir)
            if len(relative.parts) < 3:
                continue
            genus, folder = relative.parts[:2]
            source_folder = parse_audit_source_folder(folder)
            if source_folder is None:
                source_dataset = "alldata"
                source_relative = relative.as_posix()
            else:
                source_dataset = "test"
                source_relative = (Path(source_folder) / path.name).as_posix()
            rows.append(
                {
                    "stage": stage_dir.name,
                    "source_dataset": source_dataset,
                    "source_rel_path": source_relative,
                    "work_rel_path": relative.as_posix(),
                }
            )
    return rows


def commit_candidate_rows(
    rows: list[dict[str, str]],
    dataset_dir: Path,
    test_dir: Path,
) -> int:
    """在三阶段成功后将 candidate 从真实 init 移入对应 delete/stage。"""
    moved_count = 0
    for row in rows:
        source_root = dataset_dir if row["source_dataset"] == "alldata" else test_dir
        source_path = source_root / "init" / row["source_rel_path"]
        target_path = source_root / "delete" / row["stage"] / row["source_rel_path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"提交时找不到 candidate 来源：{source_path}")
        if target_path.exists():
            raise FileExistsError(f"提交目标已存在：{target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        moved_count += 1
    return moved_count


def sync_work_test_init(
    work_test_init_dir: Path,
    test_init_dir: Path,
    candidate_rows: list[dict[str, str]],
) -> int:
    """回写运行池内保留的测试菌谱，不恢复已提交的 candidate。"""
    excluded = {
        row["source_rel_path"]
        for row in candidate_rows
        if row["source_dataset"] == "test"
    }
    copied_count = 0
    for path in sorted(work_test_init_dir.rglob("*.arc_data")):
        relative = path.relative_to(work_test_init_dir).as_posix()
        if relative in excluded:
            continue
        target = test_init_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied_count += 1
    return copied_count
