"""类别前缀、测试文件夹前缀和文件名前缀工具"""

import re
from pathlib import Path


def extract_letters_prefix(name, keep_sign=False, uppercase=False, fallback=None):
    """提取开头连续字母前缀，可选保留紧随其后的 +/-"""
    text = str(name)
    matched = re.match(r"([A-Za-z]+)([+-])?", text)
    if not matched:
        return fallback
    prefix = matched.group(1)
    if keep_sign and matched.group(2):
        prefix = f"{prefix}{matched.group(2)}"
    return prefix.upper() if uppercase else prefix


def prefix_of(name):
    """提取小文件夹名前缀，例如 KAE01 -> KAE"""
    return extract_letters_prefix(name, fallback=str(name))


def normalize_folder_prefix(name):
    """把 KP02、CITF03 这类文件夹名统一成大写字母前缀"""
    text = str(name)
    return extract_letters_prefix(text, uppercase=True, fallback=text.upper())


def test_folder_prefix(name):
    """把 CS01KP、CS5EC 这类测试文件夹名统一成模型类别前缀"""
    text = str(name).strip()
    match = re.match(r"^CS\d*(.+)$", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return normalize_folder_prefix(text)


def source_prefix_from_filename(path):
    """从转换后文件名提取原子文件夹前缀，如 IgA01_xxx -> IgA01"""
    filename = Path(path).name
    stem = Path(filename).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def ensure_name_prefix(prefix, filename):
    """合并小文件夹时给文件名补前缀，避免不同来源文件名冲突"""
    prefix = f"{prefix}_"
    filename = str(filename)
    return filename if filename.startswith(prefix) else f"{prefix}{filename}"
