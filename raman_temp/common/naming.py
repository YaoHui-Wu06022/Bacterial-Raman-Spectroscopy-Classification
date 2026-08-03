"""文件夹前缀与测试菌目录名称解析。"""

from __future__ import annotations

import re
from pathlib import Path


def parse_folder_prefix(name: str, uppercase_enable: bool = False) -> str:
    """解析目录开头的字母前缀；无字母前缀时返回完整名称。"""
    text = str(name)
    match = re.match(r"[A-Za-z]+", text)
    prefix = match.group(0) if match else text
    return prefix.upper() if uppercase_enable else prefix


def parse_test_folder_prefix(name: str) -> str:
    """�?`CS01KP` 等测试菌文件夹解析对应的类别前缀。"""
    text = str(name).strip()
    match = re.match(r"^CS\d*(.+)$", text, re.IGNORECASE)
    return match.group(1).upper() if match else parse_folder_prefix(text, uppercase_enable=True)


def is_test_source_folder(name: str) -> bool:
    """判断目录名是否为可参与测试菌迁移�?`CS` 编号目录。"""
    return re.match(r"^CS\d+", str(name), re.IGNORECASE) is not None


def build_natural_key(text: str) -> list[int | str]:
    """构造按数字自然排序的键，使 cell2 排在 cell10 前。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_source_prefix(path: Path | str) -> str:
    """从转换谱文件名提取来源前缀，如 ``IgA01_xxx`` 中的 ``IgA01``。"""
    stem = Path(path).stem
    return stem.split("_", 1)[0]
