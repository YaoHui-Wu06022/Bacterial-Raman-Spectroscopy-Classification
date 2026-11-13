"""把 bacteria-ID 的 npy 光谱拆成 train_ref / train_finetune / test 三个阶段目录。

一次性数据准备脚本，不属于 ramanv2 运行时：

- 源文件读自 ``dataset/bacteriaID/_sources``；
- ``X_reference.npy`` -> ``train_ref``，``X_finetune.npy`` -> ``train_finetune``，
  ``X_test.npy`` -> ``test``；
- 谱轴从降序反转为升序，数值保持原样（数据已做过 AsLS 与单位最大值归一化）；
- 类别目录使用论文 Fig.2 的显示名，每条谱写成 ``<阶段>/<类别>/sXXXXX.arc_data``。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "bacteriaID"
SOURCE_DIR = DATASET_DIR / "_sources"

STAGE_SOURCES = {
    "train_ref": "X_reference.npy",
    "train_finetune": "X_finetune.npy",
    "test": "X_test.npy",
}

STRAIN_NAMES = {
    0: "C. albicans",
    1: "C. glabrata",
    2: "K. aerogenes",
    3: "E. coli 1",
    4: "E. coli 2",
    5: "E. faecium",
    6: "E. faecalis 1",
    7: "E. faecalis 2",
    8: "E. cloacae",
    9: "K. pneumoniae 1",
    10: "K. pneumoniae 2",
    11: "P. mirabilis",
    12: "P. aeruginosa 1",
    13: "P. aeruginosa 2",
    14: "MSSA 1",
    15: "MSSA 3",
    16: "MRSA 1",
    17: "MRSA 2",
    18: "MSSA 2",
    19: "S. enterica",
    20: "S. epidermidis",
    21: "S. lugdunensis",
    22: "S. marcescens",
    23: "S. pneumoniae 2",
    24: "S. pneumoniae 1",
    25: "S. sanguinis",
    26: "Group A Strep",
    27: "Group B Strep",
    28: "Group C Strep",
    29: "Group G Strep",
}


def load_pair(npy_name: str) -> tuple[np.ndarray, np.ndarray]:
    """读取一个 npy 的 X 与同名 y；y 统一转成整数标签码。"""
    x = np.load(SOURCE_DIR / npy_name)
    y = np.load(SOURCE_DIR / ("y" + npy_name[1:]))
    if x.ndim != 2 or x.shape[1] != 1000:
        raise ValueError(f"{npy_name} 形状异常：{x.shape}")
    if len(y) != len(x):
        raise ValueError(f"{npy_name} 与标签数不一致：{len(x)} != {len(y)}")
    return x, np.rint(y).astype(np.int64)


def axis_column(wavenumbers: np.ndarray) -> np.ndarray:
    """把固定波数轴预格式化成文本列，避免每条谱重复格式化。"""
    return np.char.add(np.char.mod("%.6f", np.asarray(wavenumbers, dtype=np.float64)), " ")


def write_spectrum(path: Path, axis_col: np.ndarray, intensity: np.ndarray) -> None:
    """写一条两列文本谱；轴已由调用方反转为升序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = np.char.add(axis_col, np.char.mod("%.6f", np.asarray(intensity, dtype=np.float64)))
    path.write_text("\n".join(body.tolist()) + "\n", encoding="utf-8")


def write_stage(npy_name: str, stage_root: Path, axis_col: np.ndarray) -> dict[str, int]:
    """把一个 npy 按 ``<类别>/sXXXXX.arc_data`` 写到阶段目录，返回每类数量。"""
    x, y = load_pair(npy_name)
    unknown = sorted(set(y.tolist()) - set(STRAIN_NAMES))
    if unknown:
        raise ValueError(f"{npy_name} 出现未知标签：{unknown}")
    counts: dict[str, int] = {}
    counters: dict[int, int] = {}
    for index in range(len(x)):
        label = int(y[index])
        offset = counters.get(label, 0)
        counters[label] = offset + 1
        class_name = STRAIN_NAMES[label]
        write_spectrum(
            stage_root / class_name / f"s{offset:05d}.arc_data",
            axis_col,
            x[index],
        )
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def publish_directory(temp_dir: Path, target_dir: Path, overwrite_enable: bool) -> None:
    """发布阶段目录；已存在时默认拒绝，显式覆盖则先备份再替换。"""
    if target_dir.exists():
        if not overwrite_enable:
            raise FileExistsError(
                f"目标已存在：{target_dir}；确认重建请加 --overwrite"
            )
        backup = target_dir.with_name(
            target_dir.name + ".bak-" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.replace(target_dir, backup)
        print(f"已备份旧目录：{backup}")
    for attempt in range(20):
        try:
            os.replace(temp_dir, target_dir)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 bacteria-ID 光谱拆成三个阶段目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的阶段目录")
    args = parser.parse_args(argv)

    if not (SOURCE_DIR / "wavenumbers.npy").is_file():
        raise FileNotFoundError(f"缺少 wavenumbers.npy：{SOURCE_DIR}")
    axis = np.asarray(np.load(SOURCE_DIR / "wavenumbers.npy")[::-1], dtype=np.float64)
    axis_col = axis_column(axis)

    for stage_name, npy_name in STAGE_SOURCES.items():
        target_dir = DATASET_DIR / stage_name
        temp_dir = target_dir.with_name(target_dir.name + f".tmp-{os.getpid()}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        try:
            counts = write_stage(npy_name, temp_dir, axis_col)
            publish_directory(temp_dir, target_dir, args.overwrite)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        print(f"{target_dir}  classes={len(counts)}  spectra={sum(counts.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
