"""单文件夹人工追加平移。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from raman.audit.common import resolve_audit_folder
from raman.audit.test_pool import source_folder_from_audit_folder
from raman.audit.workflow import _append_delta_log, _read_delta, _shift_file, _write_delta
from raman.tool.dataset import resolve_dataset
from raman.tool.naming import prefix_of
from raman.tool.path import PROJECT_ROOT


def _delta_rows(values: dict[tuple[str, str], float]) -> dict[tuple[str, str], tuple[str, float]]:
    """转换为 delta 文件的写入结构。"""
    return {
        key: (prefix_of(folder), delta)
        for key, delta in values.items()
        for _genus, folder in [key]
    }


def _source_folder_for_t(test_dir: Path, target_folder: str) -> str:
    """从当前迁移 manifest 解析一个 *t 文件夹的 CS 来源。"""
    manifest = test_dir / "test_transfer_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"缺少测试菌迁移记录：{manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        sources = {
            row["source_folder"]
            for row in csv.DictReader(file)
            if row.get("target_folder") == target_folder
        }
    if len(sources) != 1:
        raise ValueError(f"无法唯一确定 {target_folder} 的测试菌来源：{sorted(sources)}")
    return sources.pop()


def apply_manual_shift(
    folder: str,
    delta: float,
    dataset_key: str = "cos",
    test_key: str = "test",
) -> dict[str, object]:
    """给一个最终数据文件夹追加平移，并同步累计记录。"""
    if abs(delta) < 1e-9:
        raise ValueError("人工平移量不能为 0")

    profile, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    init_dir = dataset_dir / "init"
    target_dir = resolve_audit_folder(folder, dataset_dir, profile, init_dir)
    relative = target_dir.relative_to(init_dir.resolve())
    if len(relative.parts) != 2:
        raise ValueError(f"人工平移目标必须位于 init/属/文件夹：{target_dir}")

    genus, target_folder = relative.parts
    if source_folder_from_audit_folder(target_folder) is not None:
        raise ValueError("不能对临时 audit 文件夹做人工平移，请指定最终文件夹")
    files = sorted(target_dir.glob("*.arc_data"))
    if not files:
        raise FileNotFoundError(f"目标文件夹没有 .arc_data：{target_dir}")

    source_folder = ""
    source_files: list[Path] = []
    if target_folder.lower().endswith("t"):
        source_folder = _source_folder_for_t(test_dir, target_folder)
        source_dir = test_dir / "init" / source_folder
        source_files = sorted(source_dir.glob("*.arc_data"))
        if not source_files:
            raise FileNotFoundError(f"对应测试菌来源不可用：{source_dir}")

    for path in files:
        _shift_file(path, delta)
    for path in source_files:
        _shift_file(path, delta)

    current = _read_delta(dataset_dir / "delta.txt")
    key = (genus, target_folder)
    cumulative = current.get(key, 0.0) + delta
    current[key] = cumulative
    _write_delta(dataset_dir / "delta.txt", _delta_rows(current))

    now = datetime.now().isoformat(timespec="seconds")
    _append_delta_log(
        dataset_dir / "delta_log.txt",
        [{
            "time": now,
            "genus": genus,
            "folder": target_folder,
            "prefix": prefix_of(target_folder),
            "step_delta": f"{delta:+g}",
            "cumulative_delta": f"{cumulative:+g}",
            "files_changed": len(files),
            "note": "manual_extra_shift",
        }],
    )

    if source_folder:
        test_current = _read_delta(test_dir / "delta.txt")
        test_key_value = (".", source_folder)
        test_cumulative = test_current.get(test_key_value, 0.0) + delta
        test_current[test_key_value] = test_cumulative
        _write_delta(test_dir / "delta.txt", _delta_rows(test_current))
        _append_delta_log(
            test_dir / "delta_log.txt",
            [{
                "time": now,
                "genus": ".",
                "folder": source_folder,
                "prefix": prefix_of(source_folder),
                "step_delta": f"{delta:+g}",
                "cumulative_delta": f"{test_cumulative:+g}",
                "files_changed": len(source_files),
                "note": f"manual_sync_from_50cos={genus}/{target_folder}",
            }],
        )

    return {
        "folder": f"{genus}/{target_folder}",
        "step_delta": delta,
        "cumulative_delta": cumulative,
        "files_changed": len(files),
        "test_source": source_folder,
        "test_files_changed": len(source_files),
    }
