"""文件夹前缀解析与数据目录保留名。"""

from __future__ import annotations

import re


# 流水线自身维护、不是光谱来源的目录。
PIPELINE_RESERVED_FOLDER_NAMES = frozenset({"shift", "delete", ".raman_pipeline"})
# 参与清洗与归档、但不进入训练集的校准球目录。
NON_TRAINING_FOLDER_NAMES = frozenset({"小球"})


def parse_folder_prefix(name: str, uppercase_enable: bool = False) -> str:
    """解析目录开头的字母前缀；无字母前缀时返回完整名称。"""
    text = str(name)
    match = re.match(r"[A-Za-z]+", text)
    prefix = match.group(0) if match else text
    return prefix.upper() if uppercase_enable else prefix


def build_natural_key(text: str) -> list[int | str]:
    """构造按数字自然排序的键，使 cell2 排在 cell10 前。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]
