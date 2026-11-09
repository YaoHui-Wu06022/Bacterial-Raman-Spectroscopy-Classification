"""文件夹前缀解析与数据目录保留名。"""

from __future__ import annotations

import re
from pathlib import Path


PIPELINE_DIR_NAME = ".raman_pipeline"
# 流水线自身维护、不是光谱来源的目录。
PIPELINE_RESERVED_FOLDER_NAMES = frozenset({"shift", "delete", PIPELINE_DIR_NAME})
# 校准球目录名；参与清洗与归档，但不进入训练集。
BEAD_FOLDER_NAME = "小球"
NON_TRAINING_FOLDER_NAMES = frozenset({BEAD_FOLDER_NAME})
SPECIAL_FOLDER_NAMES = PIPELINE_RESERVED_FOLDER_NAMES | NON_TRAINING_FOLDER_NAMES

MANUAL_DELETE_RELATIVE = Path("delete") / "manual"
FOLDER_DELETE_RELATIVE = Path("delete") / "folder_delete"
ANALYSIS_DELETE_RELATIVE = Path("analysis_delete")

CANONICAL_FOLDER_PATTERN = re.compile(r"^([A-Za-z]+)(\d{8})_(\d+)$")
DATE_PATTERN = re.compile(r"^\d{8}$")


def is_date_name(name: str) -> bool:
    """判断目录名是否为八位日期。"""
    return bool(DATE_PATTERN.fullmatch(str(name)))


def parse_folder_prefix(name: str, uppercase_enable: bool = False) -> str:
    """解析目录开头的字母前缀；无字母前缀时返回完整名称。"""
    text = str(name)
    match = re.match(r"[A-Za-z]+", text)
    prefix = match.group(0) if match else text
    return prefix.upper() if uppercase_enable else prefix


def build_natural_key(text: str) -> list[int | str]:
    """构造按数字自然排序的键，使 cell2 排在 cell10 前。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]
