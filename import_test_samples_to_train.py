"""从测试菌抽样复制少量谱到初始数据 init"""

from __future__ import annotations

import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "dataset" / "测试菌"
TARGET_INIT = PROJECT_ROOT / "dataset" / "初始数据" / "init"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "初始数据" / "test_transfer_manifest.csv"
SUMMARY_PATH = PROJECT_ROOT / "dataset" / "初始数据" / "test_transfer_summary.md"

SAMPLES_PER_FOLDER = 8
RANDOM_SEED = 42
DRY_RUN = False


def _test_prefix(folder_name):
    """从 CS01KP 提取 KP"""
    match = re.match(r"^CS\d*(.+)$", folder_name, re.IGNORECASE)
    return match.group(1).upper() if match else folder_name.upper()


def _source_tag(folder_name):
    """从 CS01KP 提取 CS01 作为复制文件名前缀"""
    match = re.match(r"^(CS\d+)", folder_name, re.IGNORECASE)
    return match.group(1).upper() if match else folder_name.upper()


def _train_prefix(folder_name):
    """从训练小文件夹名提取种前缀"""
    match = re.match(r"^([A-Za-z]+)", folder_name)
    return match.group(1).upper() if match else folder_name.upper()


def _folder_number(folder_name, prefix):
    """提取 prefix 后面的数字尾号和位数，兼容 KP06t"""
    match = re.match(rf"^{re.escape(prefix)}(\d+)(?:t)?$", folder_name, re.IGNORECASE)
    return (int(match.group(1)), len(match.group(1))) if match else None


def _build_prefix_map(target_init):
    """扫描初始数据 init，建立种前缀到属和目标 t 文件夹的映射"""
    prefix_map = {}
    by_prefix = defaultdict(list)
    for genus_dir in sorted(path for path in target_init.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            prefix = _train_prefix(folder_dir.name)
            number_info = _folder_number(folder_dir.name, prefix)
            if number_info is not None:
                number, width = number_info
                by_prefix[prefix].append((genus_dir.name, folder_dir.name, number, width))

    for prefix, items in by_prefix.items():
        genus_names = {item[0] for item in items}
        if len(genus_names) != 1:
            continue
        genus = items[0][0]
        t_items = [item for item in items if item[1].lower().endswith("t")]
        if t_items:
            target_number = max(item[2] for item in t_items)
        else:
            target_number = max(item[2] for item in items) + 1
        width = max(item[3] for item in items)
        target_folder = f"{prefix}{target_number:0{width}d}t"
        prefix_map[prefix] = (genus, target_folder)
    return prefix_map


def _select_files(folder):
    """按文件夹独立随机种子抽样，避免新增其它文件夹时影响已抽样结果"""
    files = sorted(folder.glob("*.arc_data"))
    if len(files) <= SAMPLES_PER_FOLDER:
        return files
    rng = random.Random(f"{RANDOM_SEED}:{folder.name}")
    return sorted(rng.sample(files, SAMPLES_PER_FOLDER))


def _load_existing_manifest(path):
    """读取已有 manifest，避免重复追加同一源谱"""
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row.get("source_folder", ""), row.get("source_file", ""))
            for row in csv.DictReader(file)
        }


def _load_manifest_rows(path):
    """读取 manifest 全量行，用于生成累计摘要"""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_manifest(rows):
    """追加写入迁移 manifest"""
    fieldnames = [
        "source_dataset",
        "source_folder",
        "source_file",
        "target_genus",
        "target_folder",
        "target_file",
        "samples_per_folder",
        "random_seed",
    ]
    existing = MANIFEST_PATH.is_file()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows, skipped):
    """写出迁移摘要"""
    all_rows = _load_manifest_rows(MANIFEST_PATH) or rows
    by_target = defaultdict(int)
    by_source = defaultdict(int)
    for row in all_rows:
        by_target[f"{row['target_genus']}/{row['target_folder']}"] += 1
        by_source[row["source_folder"]] += 1

    lines = [
        "# 测试菌迁移训练样本摘要",
        "",
        f"- 源目录：`{SOURCE_ROOT}`",
        f"- 目标目录：`{TARGET_INIT}`",
        f"- manifest：`{MANIFEST_PATH}`",
        f"- 每个 CS 文件夹抽样数：{SAMPLES_PER_FOLDER}",
        f"- 随机种子：{RANDOM_SEED}",
        f"- 本次新增：{len(rows)} 条",
        f"- 累计迁移：{len(all_rows)} 条",
        f"- 跳过文件夹：{len(skipped)} 个",
        "",
        "## 目标文件夹",
        "",
    ]
    if by_target:
        for target, count in sorted(by_target.items()):
            lines.append(f"- `{target}`：{count}")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 跳过文件夹", ""])
    if skipped:
        for folder, reason in skipped:
            lines.append(f"- `{folder}`：{reason}")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 源文件夹抽样", ""])
    for folder, count in sorted(by_source.items()):
        lines.append(f"- `{folder}`：{count}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """执行测试菌到初始数据 init 的抽样复制"""
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source root: {SOURCE_ROOT}")
    if not TARGET_INIT.is_dir():
        raise FileNotFoundError(f"Missing target init: {TARGET_INIT}")

    prefix_map = _build_prefix_map(TARGET_INIT)
    existing_sources = _load_existing_manifest(MANIFEST_PATH)
    rows = []
    skipped = []

    for folder in sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir() and path.name != "audit_full_scan"):
        prefix = _test_prefix(folder.name)
        if prefix not in prefix_map:
            skipped.append((folder.name, f"未找到训练前缀 {prefix}"))
            continue
        selected = _select_files(folder)
        if not selected:
            skipped.append((folder.name, "无 .arc_data 文件"))
            continue

        genus, target_folder = prefix_map[prefix]
        target_dir = TARGET_INIT / genus / target_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        tag = _source_tag(folder.name)

        for source_file in selected:
            if (folder.name, source_file.name) in existing_sources:
                continue
            target_file = f"{tag}_{source_file.name}"
            target_path = target_dir / target_file
            if target_path.exists():
                raise FileExistsError(f"Target already exists: {target_path}")
            if not DRY_RUN:
                shutil.copy2(source_file, target_path)
            rows.append(
                {
                    "source_dataset": SOURCE_ROOT.name,
                    "source_folder": folder.name,
                    "source_file": source_file.name,
                    "target_genus": genus,
                    "target_folder": target_folder,
                    "target_file": target_file,
                    "samples_per_folder": SAMPLES_PER_FOLDER,
                    "random_seed": RANDOM_SEED,
                }
            )

    if rows and not DRY_RUN:
        _write_manifest(rows)
    _write_summary(rows, skipped)

    print(f"copied={len(rows)}")
    print(f"skipped_folders={len(skipped)}")
    print(f"manifest={MANIFEST_PATH}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
