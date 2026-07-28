"""测试菌全量临时 audit 池的复制与回写。"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

from raman.tool.dataset import resolve_dataset
from raman.tool.naming import prefix_of, test_folder_prefix
from raman.tool.path import PROJECT_ROOT


AUDIT_FOLDER_SUFFIX = "__audit"
AUDIT_FOLDER_SEPARATOR = "__"


def source_folder_from_audit_folder(folder_name: str) -> str | None:
    """从临时 audit 文件夹名还原 CS 来源文件夹名。"""
    parts = str(folder_name).split(AUDIT_FOLDER_SEPARATOR)
    if len(parts) != 3 or parts[2] != "audit":
        return None
    source_folder = parts[1]
    return source_folder if re.fullmatch(r"CS\d+.+", source_folder, re.IGNORECASE) else None


def _audit_folder_name(source_folder: str) -> str:
    """生成以种前缀开头、可无 CSV 反查来源的临时文件夹名。"""
    return f"{test_folder_prefix(source_folder)}{AUDIT_FOLDER_SEPARATOR}{source_folder}{AUDIT_FOLDER_SUFFIX}"


def _iter_audit_folders(target_init: Path):
    for genus_dir in sorted(path for path in target_init.iterdir() if path.is_dir()):
        for folder in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            source_folder = source_folder_from_audit_folder(folder.name)
            if source_folder is not None:
                yield genus_dir.name, folder, source_folder


def _clear_audit_pool(target_init: Path) -> int:
    """只删除本模块创建的临时 audit 文件夹，不触碰常规或 0.5 迁移目录。"""
    removed = 0
    target_root = target_init.resolve()
    for _genus, folder, _source_folder in list(_iter_audit_folders(target_init)):
        resolved = folder.resolve()
        try:
            resolved.relative_to(target_root)
        except ValueError as exc:
            raise ValueError(f"Audit pool folder escapes target init: {resolved}") from exc
        shutil.rmtree(folder)
        removed += 1
    return removed


def _target_genus_by_prefix(target_init: Path) -> dict[str, str]:
    """从现有 50cos 类别目录解析测试菌种前缀对应的唯一属。"""
    genera_by_prefix: dict[str, set[str]] = {}
    for genus_dir in sorted(path for path in target_init.iterdir() if path.is_dir()):
        for folder in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            if source_folder_from_audit_folder(folder.name) is not None:
                continue
            prefix = prefix_of(folder.name).upper()
            genera_by_prefix.setdefault(prefix, set()).add(genus_dir.name)

    result = {}
    for prefix, genera in genera_by_prefix.items():
        if len(genera) == 1:
            result[prefix] = next(iter(genera))
    return result


def transfer_all(dataset_key: str = "alldata", test_key: str = "test") -> tuple[int, int]:
    """全量复制 CS 测试谱到临时 audit 池；不依赖迁移比例或 CSV。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    target_init = dataset_dir / "init"
    source_root = test_dir / "init"
    if not target_init.is_dir():
        raise FileNotFoundError(f"Missing target init: {target_init}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing test init: {source_root}")

    _clear_audit_pool(target_init)
    genus_by_prefix = _target_genus_by_prefix(target_init)
    copied = 0
    skipped = 0
    for source_folder in sorted(path for path in source_root.iterdir() if path.is_dir() and re.fullmatch(r"CS\d+.+", path.name, re.IGNORECASE)):
        prefix = test_folder_prefix(source_folder.name)
        genus = genus_by_prefix.get(prefix)
        if genus is None:
            skipped += 1
            continue
        target_folder = target_init / genus / _audit_folder_name(source_folder.name)
        target_folder.mkdir(parents=True, exist_ok=False)
        for source_file in sorted(source_folder.glob("*.arc_data")):
            shutil.copy2(source_file, target_folder / source_file.name)
            copied += 1
    return copied, skipped


def sync_back(dataset_key: str = "alldata", test_key: str = "test") -> tuple[int, int]:
    """把临时 audit 池中保留的谱回写到测试集，再清理临时池。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    target_init = dataset_dir / "init"
    source_root = test_dir / "init"
    copied = 0
    missing = 0
    for _genus, folder, source_folder in list(_iter_audit_folders(target_init)):
        source_dir = source_root / source_folder
        for target_file in sorted(folder.glob("*.arc_data")):
            source_file = source_dir / target_file.name
            if not source_file.is_file():
                missing += 1
                continue
            shutil.copy2(target_file, source_file)
            copied += 1
    _clear_audit_pool(target_init)
    return copied, missing


def rebuild_training_test_copies(dataset_key: str = "alldata") -> None:
    """按测试菌迁移脚本当前配置重建 *t 训练副本与 manifest。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset("test", PROJECT_ROOT)
    script = test_dir / "transfer_cs_to_init.py"
    spec = importlib.util.spec_from_file_location("raman_test_transfer", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载测试菌迁移脚本：{script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET_INIT = dataset_dir / "init"
    result = module.main([])
    if result:
        raise RuntimeError(f"重建 *t 训练副本失败，退出码：{result}")
