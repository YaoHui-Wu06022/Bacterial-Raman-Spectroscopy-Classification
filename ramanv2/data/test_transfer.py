"""从测试菌 `init/` 构建 `alldata/init/` 的 `*t` 派生副本。"""

from __future__ import annotations

import csv
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ramanv2.common.naming import build_natural_key, is_test_source_folder, parse_folder_prefix, parse_test_folder_prefix


MANIFEST_FIELDS = (
    "source_dataset",
    "source_folder",
    "source_file",
    "target_dataset",
    "target_genus",
    "target_folder",
    "target_file",
    "samples_per_folder",
    "selection_mode",
    "source_group",
    "selected_group_count",
    "total_group_count",
    "train_group_ratio",
    "random_seed",
)
FOLDER_MAP_FIELDS = ("source_folder", "target_genus", "target_folder")


@dataclass(frozen=True)
class TestTransferConfig:
    """定义测试菌按测量组抽取到 `*t` 副本的固定参数。"""

    __test__ = False
    train_group_ratio: float = 0.5
    random_seed: int = 42
    target_suffix: str = "t"


@dataclass(frozen=True)
class TestTransferResult:
    """汇总一次 `*t` 副本构建的文件数、跳过目录和输出位置。"""

    __test__ = False
    transferred_count: int
    skipped_count: int
    manifest_path: Path
    folder_map_path: Path


def build_cs_sort_key(path: Path) -> tuple[int, str]:
    """按 CS 编号排序测试菌文件夹，保持分配顺序稳定。"""
    match = re.match(r"^CS(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 10**9, path.name


def parse_folder_index(folder: str, prefix: str, suffix: str) -> tuple[int, int] | None:
    """解析 `KP06t` 等目录中的编号及其补零宽度。"""
    match = re.match(
        rf"^{re.escape(prefix)}(\d+)(?:{re.escape(suffix)})?$",
        folder,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return int(match.group(1)), len(match.group(1))


def build_prefix_index(init_dir: Path, config: TestTransferConfig) -> dict[str, dict[str, object]]:
    """扫描常规目录，建立类别前缀到属和可用编号的唯一索引。"""
    index: dict[str, dict[str, object]] = {}
    genera_by_prefix: dict[str, set[str]] = defaultdict(set)
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            if folder_dir.name.lower().endswith(config.target_suffix.lower()):
                continue
            prefix = parse_folder_prefix(folder_dir.name, uppercase_enable=True)
            parsed = parse_folder_index(folder_dir.name, prefix, config.target_suffix)
            if parsed is None:
                continue
            number, width = parsed
            item = index.setdefault(prefix, {"genus": genus_dir.name, "max_number": 0, "width": width})
            item["max_number"] = max(int(item["max_number"]), number)
            item["width"] = max(int(item["width"]), width)
            genera_by_prefix[prefix].add(genus_dir.name)
    ambiguous = {
        prefix: sorted(genera)
        for prefix, genera in genera_by_prefix.items()
        if len(genera) > 1
    }
    if ambiguous:
        details = "; ".join(f"{prefix}: {', '.join(genera)}" for prefix, genera in sorted(ambiguous.items()))
        raise ValueError(f"类别前缀无法唯一映射到属目录：{details}")
    return index


def read_folder_map(path: Path) -> dict[str, tuple[str, str]]:
    """读取 CS 文件夹到稳定 `*t` 目录的映射。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            row["source_folder"]: (row["target_genus"], row["target_folder"])
            for row in csv.DictReader(file)
            if row.get("source_folder") and row.get("target_genus") and row.get("target_folder")
        }


def write_folder_map(path: Path, folder_map: dict[str, tuple[str, str]]) -> None:
    """写出稳定的 CS 到 `*t` 文件夹映射。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FOLDER_MAP_FIELDS)
        writer.writeheader()
        for source, (genus, folder) in sorted(folder_map.items(), key=lambda item: build_cs_sort_key(Path(item[0]))):
            writer.writerow({"source_folder": source, "target_genus": genus, "target_folder": folder})


def reserve_folder_numbers(
    prefix_index: dict[str, dict[str, object]],
    folder_map: dict[str, tuple[str, str]],
    config: TestTransferConfig,
) -> None:
    """保留历史映射使用过的编号，避免 CS 增减改变后续目录编号。"""
    for _source, (_genus, folder) in folder_map.items():
        prefix = parse_folder_prefix(folder, uppercase_enable=True)
        parsed = parse_folder_index(folder, prefix, config.target_suffix)
        if prefix not in prefix_index or parsed is None:
            continue
        number, width = parsed
        item = prefix_index[prefix]
        item["max_number"] = max(int(item["max_number"]), number)
        item["width"] = max(int(item["width"]), width)


def build_next_target_folder(prefix: str, prefix_index: dict[str, dict[str, object]], config: TestTransferConfig) -> tuple[str, str]:
    """为未映射 CS 文件夹分配下一个同类 `*t` 目录。"""
    if prefix not in prefix_index:
        raise KeyError(f"目标 init 中没有类别前缀：{prefix}")
    item = prefix_index[prefix]
    item["max_number"] = int(item["max_number"]) + 1
    width = int(item["width"])
    return str(item["genus"]), f"{prefix}{int(item['max_number']):0{width}d}{config.target_suffix}"


def list_test_source_dirs(source_init_dir: Path) -> list[Path]:
    """返回按 CS 编号稳定排序的测试菌来源目录。"""
    return sorted(
        (path for path in source_init_dir.iterdir() if path.is_dir() and is_test_source_folder(path.name)),
        key=build_cs_sort_key,
    )


def parse_measurement_group(path: Path) -> str:
    """从文件名提取独立测量组，确保同组重复测量不会拆分。"""
    return path.stem.split("_", 1)[0]


def select_training_files(source_dir: Path, config: TestTransferConfig) -> tuple[list[Path], set[str], int]:
    """按独立测量组随机选择固定比例的谱用于 `*t` 副本。"""
    files = sorted(source_dir.glob("*.arc_data"))
    if not files:
        raise ValueError(f"测试菌目录没有 .arc_data：{source_dir}")
    if not 0 < config.train_group_ratio <= 1:
        raise ValueError(f"train_group_ratio 必须位于 (0, 1]：{config.train_group_ratio}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[parse_measurement_group(path)].append(path)
    group_names = sorted(groups, key=build_natural_key)
    count = min(max(1, int(len(group_names) * config.train_group_ratio)), len(group_names))
    rng = random.Random(f"{config.random_seed}:{source_dir.name}:cell_groups")
    selected_groups = set(rng.sample(group_names, count))
    selected_files = [
        path
        for group in group_names
        if group in selected_groups
        for path in sorted(groups[group])
    ]
    return selected_files, selected_groups, len(group_names)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """写出推理和审核均可反查来源的测试菌迁移清单。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_transfer_stage_dir(target_init_dir: Path) -> Path:
    """创建与正式 `init/` 同级的 `*t` 副本构建目录。"""
    stage_dir = target_init_dir.parent / f"{target_init_dir.name}_test_transfer_{uuid4().hex[:8]}"
    stage_dir.mkdir(parents=True, exist_ok=False)
    (stage_dir / "folders").mkdir()
    return stage_dir


def publish_transfer_outputs(
    stage_dir: Path,
    target_init_dir: Path,
    manifest_path: Path,
    folder_map_path: Path,
) -> None:
    """发布已完成的 `*t` 目录、manifest 和稳定映射。"""
    backup_dir = target_init_dir.parent / f"{target_init_dir.name}_test_transfer_previous_{uuid4().hex[:8]}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for genus_dir in sorted(path for path in target_init_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir() and path.name.lower().endswith("t")):
            target = backup_dir / genus_dir.name / folder_dir.name
            target.parent.mkdir(parents=True, exist_ok=True)
            folder_dir.replace(target)
    for output_path in (manifest_path, folder_map_path):
        if output_path.is_file():
            output_path.replace(backup_dir / output_path.name)
    for genus_dir in sorted(path for path in (stage_dir / "folders").iterdir() if path.is_dir()):
        target_genus_dir = target_init_dir / genus_dir.name
        target_genus_dir.mkdir(parents=True, exist_ok=True)
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            folder_dir.replace(target_genus_dir / folder_dir.name)
    (stage_dir / "manifest.csv").replace(manifest_path)
    (stage_dir / "folder_map.csv").replace(folder_map_path)
    shutil.rmtree(stage_dir)


def build_test_transfer(
    source_init_dir: Path,
    target_init_dir: Path,
    manifest_path: Path,
    folder_map_path: Path,
    config: TestTransferConfig | None = None,
) -> TestTransferResult:
    """构建并发布全部测试菌 `*t` 派生副本及其来源清单。"""
    transfer_config = TestTransferConfig() if config is None else config
    if not source_init_dir.is_dir():
        raise FileNotFoundError(f"缺少测试菌 init：{source_init_dir}")
    if not target_init_dir.is_dir():
        raise FileNotFoundError(f"缺少目标 init：{target_init_dir}")
    prefix_index = build_prefix_index(target_init_dir, transfer_config)
    folder_map = read_folder_map(folder_map_path)
    reserve_folder_numbers(prefix_index, folder_map, transfer_config)
    stage_dir = build_transfer_stage_dir(target_init_dir)
    rows: list[dict[str, object]] = []
    skipped_count = 0
    for source_dir in list_test_source_dirs(source_init_dir):
        try:
            selected_files, selected_groups, total_group_count = select_training_files(source_dir, transfer_config)
            target_genus, target_folder = folder_map.get(source_dir.name, ("", ""))
            if not target_folder:
                target_genus, target_folder = build_next_target_folder(
                    parse_test_folder_prefix(source_dir.name), prefix_index, transfer_config
                )
                folder_map[source_dir.name] = (target_genus, target_folder)
        except (KeyError, ValueError):
            skipped_count += 1
            continue
        target_dir = stage_dir / "folders" / target_genus / target_folder
        target_dir.mkdir(parents=True, exist_ok=False)
        for source_path in selected_files:
            target_file = f"{source_dir.name}_{source_path.name}"
            shutil.copy2(source_path, target_dir / target_file)
            rows.append(
                {
                    "source_dataset": source_init_dir.name,
                    "source_folder": source_dir.name,
                    "source_file": source_path.name,
                    "target_dataset": target_init_dir.parent.name,
                    "target_genus": target_genus,
                    "target_folder": target_folder,
                    "target_file": target_file,
                    "samples_per_folder": len(selected_files),
                    "selection_mode": "cell_group",
                    "source_group": parse_measurement_group(source_path),
                    "selected_group_count": len(selected_groups),
                    "total_group_count": total_group_count,
                    "train_group_ratio": transfer_config.train_group_ratio,
                    "random_seed": transfer_config.random_seed,
                }
            )
    write_manifest(stage_dir / "manifest.csv", rows)
    write_folder_map(stage_dir / "folder_map.csv", folder_map)
    publish_transfer_outputs(stage_dir, target_init_dir, manifest_path, folder_map_path)
    return TestTransferResult(len(rows), skipped_count, manifest_path, folder_map_path)
