"""将部分 CS 文件夹复制到 alldata/init，并记录推理排除名单。"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd


CS_DATA_DIR = Path(__file__).resolve().parent
DATASET_DIR = CS_DATA_DIR.parent
CS_INIT_DIR = CS_DATA_DIR / "init"
ALDATA_INIT_DIR = DATASET_DIR / "alldata" / "init"
CLASSIFIER_PATH = DATASET_DIR / "病原菌分类与规范简称.xlsx"
MANIFEST_PATH = CS_DATA_DIR / "alldata_transfer_manifest.csv"
TRANSFER_FOLDER_PATTERN = re.compile(r"^[A-Za-z]+CS\d+$")


@dataclass(frozen=True)
class TransferItem:
    """描述一个保留 CS 来源名称并复制到 alldata 的文件夹。"""

    species_prefix: str
    retained_folder: str
    source_dir: Path
    target_dir: Path
    spectrum_count: int


def parse_cs_folder_name(name: str) -> tuple[str, str]:
    """解析 CS 文件夹，返回原编号文本和规范种简称。"""
    matched = re.fullmatch(r"CS(\d+)([A-Za-z]+)", name, re.IGNORECASE)
    if matched is None:
        raise ValueError(f"无法解析 CS 文件夹名称：{name}")
    return matched.group(1), matched.group(2).upper()


def load_species_map() -> dict[str, tuple[str, str]]:
    """读取来源简称到规范种简称、英文属名的映射。"""
    table = pd.read_excel(CLASSIFIER_PATH)
    species_map: dict[str, tuple[str, str]] = {}
    for _, row in table.iterrows():
        species_prefix = str(row["规范种简称"]).strip().upper()
        genus_name = str(row["属英文"]).strip()
        if species_prefix == "NAN" or genus_name.lower() == "nan":
            continue
        species_map[species_prefix] = (species_prefix, genus_name)

    # 实验室采集简称可能不同于规范简称；规范简称本身优先，避免同名别称覆盖。
    for _, row in table.iterrows():
        alias = str(row["实验室自定义缩写"]).strip().upper()
        species_prefix = str(row["规范种简称"]).strip().upper()
        genus_name = str(row["属英文"]).strip()
        if (
            alias == "NAN"
            or species_prefix == "NAN"
            or genus_name.lower() == "nan"
        ):
            continue
        species_map.setdefault(alias, (species_prefix, genus_name))
    return species_map


def build_transfer_items() -> list[TransferItem]:
    """按物种选择高编号来源；单一来源也复制到 alldata。"""
    species_map = load_species_map()
    folders_by_prefix: dict[str, list[tuple[int, str, Path]]] = {}
    for source_dir in sorted(path for path in CS_INIT_DIR.iterdir() if path.is_dir()):
        number_text, source_prefix = parse_cs_folder_name(source_dir.name)
        if source_prefix not in species_map:
            raise KeyError(f"分类表未映射 CS 来源简称：{source_prefix}")
        species_prefix, _ = species_map[source_prefix]
        folders_by_prefix.setdefault(species_prefix, []).append(
            (int(number_text), number_text, source_dir)
        )

    items = []
    for species_prefix, folders in sorted(folders_by_prefix.items()):
        _, genus_name = species_map[species_prefix]
        folders.sort(key=lambda item: (item[0], item[2].name.casefold()))
        transfer_count = 1 if len(folders) == 1 else (len(folders) + 1) // 2
        retained_folder = folders[0][2].name
        for _, number_text, source_dir in folders[-transfer_count:] if transfer_count else []:
            target_name = f"{species_prefix}CS{number_text}"
            target_dir = ALDATA_INIT_DIR / genus_name / target_name
            spectrum_count = sum(1 for _ in source_dir.rglob("*.arc_data"))
            if not spectrum_count:
                raise ValueError(f"CS 文件夹没有 .arc_data：{source_dir}")
            items.append(
                TransferItem(
                    species_prefix,
                    retained_folder,
                    source_dir,
                    target_dir,
                    spectrum_count,
                )
            )
    return items


def copy_transfer_items(items: list[TransferItem], temporary_dir: Path) -> None:
    """先将所有待复制目录写入临时位置并核对谱数。"""
    for item in items:
        output_dir = temporary_dir / item.target_dir.relative_to(ALDATA_INIT_DIR)
        shutil.copytree(item.source_dir, output_dir)
        copied_count = sum(1 for _ in output_dir.rglob("*.arc_data"))
        if copied_count != item.spectrum_count:
            raise ValueError(f"复制后的谱数不一致：{item.source_dir}")


def collect_previous_transfer_dirs() -> list[Path]:
    """收集 alldata/init 中由 CS 迁入的种简称加 CS 编号目录。"""
    if not ALDATA_INIT_DIR.is_dir():
        return []
    return sorted(
        path
        for genus_dir in ALDATA_INIT_DIR.iterdir()
        if genus_dir.is_dir()
        for path in genus_dir.iterdir()
        if path.is_dir() and TRANSFER_FOLDER_PATTERN.fullmatch(path.name)
    )


def restore_transfer_items(
    items: list[TransferItem],
    temporary_dir: Path,
    previous_dirs: list[Path],
) -> None:
    """将临时发布结果和备份目录恢复到发布前状态。"""
    for item in reversed(items):
        source_dir = temporary_dir / item.target_dir.relative_to(ALDATA_INIT_DIR)
        if item.target_dir.exists():
            item.target_dir.replace(source_dir)
    previous_root = temporary_dir / "previous"
    for previous_dir in previous_dirs:
        backup_dir = previous_root / previous_dir.relative_to(ALDATA_INIT_DIR)
        if backup_dir.exists():
            backup_dir.replace(previous_dir)


def publish_transfer_items(items: list[TransferItem], temporary_dir: Path) -> list[Path]:
    """用已核对的目录替换上次 CS 迁入结果，发布失败时恢复原目录。"""
    previous_dirs = collect_previous_transfer_dirs()
    previous_set = set(previous_dirs)
    target_dirs = [item.target_dir for item in items]
    if len(target_dirs) != len(set(target_dirs)):
        raise ValueError("CS 迁入目标目录重复")
    for target_dir in target_dirs:
        if target_dir.exists() and target_dir not in previous_set:
            raise FileExistsError(f"alldata 目标文件夹已存在：{target_dir}")

    moved_previous_dirs: list[Path] = []
    published_items: list[TransferItem] = []
    try:
        previous_root = temporary_dir / "previous"
        for previous_dir in previous_dirs:
            backup_dir = previous_root / previous_dir.relative_to(ALDATA_INIT_DIR)
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
            previous_dir.replace(backup_dir)
            moved_previous_dirs.append(previous_dir)
        for item in items:
            source_dir = temporary_dir / item.target_dir.relative_to(ALDATA_INIT_DIR)
            item.target_dir.parent.mkdir(parents=True, exist_ok=True)
            source_dir.replace(item.target_dir)
            published_items.append(item)
    except Exception:
        restore_transfer_items(published_items, temporary_dir, moved_previous_dirs)
        raise
    return previous_dirs
    for item in items:
        if item.target_dir.exists():
            raise FileExistsError(f"alldata 目标文件夹已存在：{item.target_dir}")
        source_dir = temporary_dir / item.target_dir.relative_to(ALDATA_INIT_DIR)
        item.target_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir.replace(item.target_dir)


def write_transfer_manifest(items: list[TransferItem], output_path: Path) -> None:
    """写出已复制来源名单，供默认独立推理跳过。"""
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "species_prefix",
                "retained_folder",
                "source_folder",
                "target_genus",
                "target_folder",
                "spectrum_count",
            ),
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "species_prefix": item.species_prefix,
                    "retained_folder": item.retained_folder,
                    "source_folder": item.source_dir.name,
                    "target_genus": item.target_dir.parent.name,
                    "target_folder": item.target_dir.name,
                    "spectrum_count": item.spectrum_count,
                }
            )


def run_copy() -> None:
    """刷新 CS 迁入目录，并在成功后更新推理排除名单。"""
    items = build_transfer_items()
    temporary_dir = ALDATA_INIT_DIR.parent / f".cs_transfer_{uuid4().hex}"
    temporary_dir.mkdir()
    completed_enable = False
    published_enable = False
    previous_dirs: list[Path] = []
    try:
        copy_transfer_items(items, temporary_dir)
        manifest_path = temporary_dir / MANIFEST_PATH.name
        write_transfer_manifest(items, manifest_path)
        previous_dirs = publish_transfer_items(items, temporary_dir)
        published_enable = True
        manifest_path.replace(MANIFEST_PATH)
        completed_enable = True
        print(f"copied_folders={len(items)}")
        print(f"copied_spectra={sum(item.spectrum_count for item in items)}")
    except Exception:
        if published_enable:
            restore_transfer_items(items, temporary_dir, previous_dirs)
        raise
    finally:
        if completed_enable and temporary_dir.exists():
            shutil.rmtree(temporary_dir)


if __name__ == "__main__":
    run_copy()
