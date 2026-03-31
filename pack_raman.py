"""打包 raman 库到 zip 文件。"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent
RAMAN_DIR = PROJECT_ROOT / "raman"
DEFAULT_OUTPUT = PROJECT_ROOT / "raman.zip"


def _should_skip(path: Path) -> bool:
    """过滤缓存文件，避免把无用产物打进压缩包。"""
    if "__pycache__" in path.parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def pack_raman(output_path: Path) -> tuple[Path, int]:
    """将 raman 目录打包到指定 zip 文件。"""
    if not RAMAN_DIR.exists():
        raise FileNotFoundError(f"Missing raman directory: {RAMAN_DIR}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(RAMAN_DIR.rglob("*")):
            if path.is_dir() or _should_skip(path):
                continue
            zf.write(path, arcname=path.relative_to(PROJECT_ROOT))
            file_count += 1

    return output_path, file_count


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 raman 库到 zip 文件")
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help="输出 zip 路径，默认写到项目根目录下的 raman.zip",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    zip_path, file_count = pack_raman(output_path)
    print(f"Packed {file_count} files -> {zip_path}")


if __name__ == "__main__":
    main()
