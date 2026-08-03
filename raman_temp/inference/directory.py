"""独立推理输入目录与光谱文件的确定性枚举。"""

from __future__ import annotations

from pathlib import Path

from raman_temp.common.naming import build_natural_key


def resolve_input_dirs(
    input_dir: Path | str,
    one_dir: Path | str | None = None,
) -> list[Path]:
    """列出待预测目录，或解析指定的单个目录。"""
    root_dir = Path(input_dir).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"推理输入目录不存在：{root_dir}")
    if one_dir is None:
        return sorted(path for path in root_dir.iterdir() if path.is_dir())
    target_dir = Path(one_dir)
    if not target_dir.is_absolute():
        target_dir = root_dir / target_dir
    if not target_dir.is_dir():
        raise FileNotFoundError(f"指定推理目录不存在：{target_dir}")
    return [target_dir]


def list_spectrum_paths(input_dir: Path | str) -> list[Path]:
    """按文件名自然顺序列出一个目录中的 `.arc_data` 文件。"""
    return sorted(
        (
            path
            for path in Path(input_dir).iterdir()
            if path.is_file() and path.suffix.lower() == ".arc_data"
        ),
        key=lambda path: build_natural_key(path.name),
    )
