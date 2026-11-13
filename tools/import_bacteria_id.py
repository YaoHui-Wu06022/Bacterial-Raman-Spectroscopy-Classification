"""把 bacteria-ID 数据集的 npy 数组导入为 ramanv2 常规数据集目录。

这是一次性数据准备脚本，不属于 ramanv2 运行时：

- 只写 ``dataset/<dir>/dataset.json``、``init`` 与 ``init_test``；
- 谱轴从降序反转为升序（ramanv2 的插值要求升序波数）；
- 保留数据原本的预处理结果，不做任何基线/归一化改动；
- 只生成 63k 主实验：init = reference + finetune，init_test = test。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_SOURCE = DATASET_ROOT / "_sources" / "bacteria_id"

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

NPY_NAMES = {
    "reference": "X_reference.npy",
    "finetune": "X_finetune.npy",
    "test": "X_test.npy",
}

BATCH_SUFFIX = {
    "reference": "reference",
    "finetune": "finetune",
}

# 唯一的实验：reference + finetune 作训练，test 作独立测试。
EXPERIMENTS = {
    "full": {
        "directory": "bacteriaID",
        "dataset_id": "BID",
        "display_name": "bacteria-ID 30 isolates (reference+finetune)",
        "init_sources": ("reference", "finetune"),
        "test_source": "test",
    },
}


def load_pair(source_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    """读取一个来源的 X 与 y；y 统一转成整数标签码。"""
    x_name = NPY_NAMES[source]
    y_name = "y" + x_name[1:]
    x = np.load(source_dir / x_name)
    y = np.load(source_dir / y_name)
    if x.ndim != 2 or x.shape[1] != 1000:
        raise ValueError(f"{x_name} 形状异常：{x.shape}")
    if len(y) != len(x):
        raise ValueError(f"{x_name} 与 {y_name} 样本数不一致：{len(x)} != {len(y)}")
    return x, np.rint(y).astype(np.int64)


def axis_column(wavenumbers: np.ndarray) -> np.ndarray:
    """把固定波数轴预格式化成文本列，避免每条谱重复格式化。"""
    return np.char.add(np.char.mod("%.6f", np.asarray(wavenumbers, dtype=np.float64)), " ")


def write_spectrum(path: Path, axis_col: np.ndarray, intensity: np.ndarray) -> None:
    """写一条两列文本谱；轴已由调用方反转为升序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = np.char.add(axis_col, np.char.mod("%.6f", np.asarray(intensity, dtype=np.float64)))
    path.write_text("\n".join(body.tolist()) + "\n", encoding="utf-8")


def batch_name(class_name: str, source: str) -> str:
    """init 批次目录名；必须以类别开头，前缀映射才能解析出类别。"""
    return f"{class_name}_{BATCH_SUFFIX[source]}"


def _iter_labeled_rows(source_dir: Path, source: str):
    """按原始行序产出 (类别名, 类内序号, 强度行)。"""
    x, y = load_pair(source_dir, source)
    unknown = sorted(set(y.tolist()) - set(STRAIN_NAMES))
    if unknown:
        raise ValueError(f"{source} 出现未知标签：{unknown}")
    position_in_class: dict[int, int] = {}
    for index in range(len(x)):
        label = int(y[index])
        offset = position_in_class.get(label, 0)
        position_in_class[label] = offset + 1
        yield STRAIN_NAMES[label], offset, x[index]


def write_init(source_dir: Path, source: str, init_root: Path, wavenumbers: np.ndarray) -> dict[str, int]:
    """把来源按 ``init/<类别>/<类别>_<来源>/<文件>`` 写出。"""
    counts: dict[str, int] = {}
    axis_col = axis_column(wavenumbers)
    for name, offset, intensity in _iter_labeled_rows(source_dir, source):
        write_spectrum(
            init_root / name / batch_name(name, source) / f"s{offset:05d}.arc_data",
            axis_col,
            intensity,
        )
        counts[name] = counts.get(name, 0) + 1
    return counts


def write_test(source_dir: Path, source: str, test_root: Path, wavenumbers: np.ndarray) -> dict[str, int]:
    """把独立测试集写成 ``init_test/<显示名>/<文件>``。"""
    counts: dict[str, int] = {}
    axis_col = axis_column(wavenumbers)
    for name, offset, intensity in _iter_labeled_rows(source_dir, source):
        write_spectrum(
            test_root / name / f"s{offset:05d}.arc_data",
            axis_col,
            intensity,
        )
        counts[name] = counts.get(name, 0) + 1
    return counts


def write_prebuilt(
    source_dir: Path,
    source: str,
    stage_root: Path,
    native_axis: np.ndarray,
    target_axis: np.ndarray,
) -> dict[str, int]:
    """直接写出模型网格上的 train/test，跳过 airPLS 二次基线。

    init/init_test 保留原始 1000 点轴；这里只做裁剪与重采样，数值不再改动。
    """
    counts: dict[str, int] = {}
    axis_col = axis_column(target_axis)
    for name, offset, intensity in _iter_labeled_rows(source_dir, source):
        values = np.interp(target_axis, native_axis, intensity)
        write_spectrum(
            stage_root / name / f"{source}_s{offset:05d}.arc_data",
            axis_col,
            values,
        )
        counts[name] = counts.get(name, 0) + 1
    return counts


def dataset_config(experiment: dict[str, str]) -> dict[str, object]:
    """扁平数据集的 dataset.json：单层 level_1，标签来自文件夹前缀。"""
    dataset_id = experiment["dataset_id"]
    return {
        "version": 1,
        "dataset_id": dataset_id,
        "display_name": experiment["display_name"],
        "bad_bands": [],
        "default": False,
        "profiles": [],
        "label_source": {
            "kind": "folder_prefix",
            "root": "cosmic_data",
            "levels": {"class": "prefix"},
            "prefix_level": "class",
            "grouping_key": "class",
            "default_profile": dataset_id,
            "profiles": {dataset_id: {"where": {}}},
        },
    }


def publish_directory(temp_dir: Path, target_dir: Path, overwrite_enable: bool) -> None:
    """发布到最终目录；已存在时默认拒绝，显式覆盖则先备份。"""
    if target_dir.exists():
        if not overwrite_enable:
            raise FileExistsError(
                f"目标已存在：{target_dir}；确认重建请加 --overwrite（旧目录会改名为 .bak-时间戳）"
            )
        backup = target_dir.with_name(
            target_dir.name + ".bak-" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.replace(target_dir, backup)
        print(f"已备份旧目录：{backup}")
    # Windows 上杀毒/索引服务偶发短暂占用目录，重试几次再放弃。
    for attempt in range(6):
        try:
            os.replace(temp_dir, target_dir)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5)


def import_experiment(
    name: str,
    experiment: dict[str, object],
    source_dir: Path,
    overwrite_enable: bool,
    prebuilt_enable: bool,
    grid: tuple[float, float, int],
) -> None:
    """生成一个实验数据集：dataset.json + init + init_test（可选预构建 train/test）。"""
    axis = np.load(source_dir / "wavenumbers.npy")
    if axis.shape != (1000,):
        raise ValueError(f"wavenumbers 形状异常：{axis.shape}")
    ascending_axis = np.asarray(axis[::-1], dtype=np.float64)

    target_dir = DATASET_ROOT / str(experiment["directory"])
    temp_dir = target_dir.with_name(target_dir.name + f".tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    init_counts: dict[str, int] = {}
    for source in experiment["init_sources"]:
        counts = write_init(source_dir, source, temp_dir / "init", ascending_axis)
        for key, value in counts.items():
            init_counts[key] = init_counts.get(key, 0) + value

    test_source = str(experiment["test_source"])
    test_counts = write_test(
        source_dir,
        test_source,
        temp_dir / "init_test",
        ascending_axis,
    )

    prebuilt_note = ""
    if prebuilt_enable:
        cut_min, cut_max, points = grid
        target_axis = np.linspace(float(cut_min), float(cut_max), int(points))
        for source in experiment["init_sources"]:
            write_prebuilt(
                source_dir,
                source,
                temp_dir / "train",
                ascending_axis,
                target_axis,
            )
        write_prebuilt(
            source_dir,
            test_source,
            temp_dir / "test",
            ascending_axis,
            target_axis,
        )
        prebuilt_note = f"  prebuilt={int(points)}pt[{cut_min},{cut_max}]"

    config_path = temp_dir / "dataset.json"
    config_path.write_text(
        json.dumps(dataset_config(experiment), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    publish_directory(temp_dir, target_dir, overwrite_enable)
    print(
        f"[{name}] {target_dir}  init_classes={len(init_counts)}"
        f"  init_spectra={sum(init_counts.values())}"
        f"  test_folders={len(test_counts)}"
        f"  test_spectra={sum(test_counts.values())}{prebuilt_note}"
    )


def verify_experiment(name: str, experiment: dict[str, object], source_dir: Path) -> None:
    """回读若干文件，核对数值与原始 npy 完全一致。"""
    target_dir = DATASET_ROOT / str(experiment["directory"])
    axis = np.asarray(np.load(source_dir / "wavenumbers.npy")[::-1], dtype=np.float64)
    for source in experiment["init_sources"]:
        x, y = load_pair(source_dir, source)
        class_name = STRAIN_NAMES[int(y[0])]
        first_file = (
            target_dir
            / "init"
            / class_name
            / batch_name(class_name, str(source))
            / "s00000.arc_data"
        )
        data = np.loadtxt(first_file)
        if data.shape != (1000, 2):
            raise ValueError(f"{first_file} 形状异常：{data.shape}")
        if not np.allclose(data[:, 0], axis, atol=1e-6):
            raise ValueError(f"{first_file} 波数轴回读不一致")
        if not np.allclose(data[:, 1], x[0], atol=1e-6):
            raise ValueError(f"{first_file} 强度回读不一致")
    print(f"[{name}] 回读校验通过")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入 bacteria-ID npy 为 ramanv2 数据集")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="npy 所在目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的数据集目录")
    parser.add_argument(
        "--prebuilt",
        action="store_true",
        help="同时写出模型网格上的 train/test（跳过 airPLS 二次基线）",
    )
    parser.add_argument("--cut-min", type=float, default=600.0)
    parser.add_argument("--cut-max", type=float, default=1800.0)
    parser.add_argument("--points", type=int, default=896)
    args = parser.parse_args(argv)

    source_dir = Path(args.source)
    if not (source_dir / "wavenumbers.npy").is_file():
        raise FileNotFoundError(f"缺少 wavenumbers.npy：{source_dir}")

    import_experiment(
        "full",
        EXPERIMENTS["full"],
        source_dir,
        args.overwrite,
        args.prebuilt,
        (args.cut_min, args.cut_max, args.points),
    )
    verify_experiment("full", EXPERIMENTS["full"], source_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
