"""把 dataset/bacteriaID 打包成 notebook 可直接恢复的 data.npz。

仓库自带的打包入口只收 init/init_test；本数据集额外预构建了 train/test，
所以这里按同一归档格式把四个阶段一起打包，云端恢复后即可直接训练。
路径命名直接复用导入脚本，避免两边各写一套。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ramanv2.packaging.platform_bundle import ARCHIVE_VERSION, DATA_ARCHIVE_NAME


def _load_importer():
    """按文件路径加载导入脚本，复用其类别表与目录命名。"""
    spec = importlib.util.spec_from_file_location(
        "bacteria_import",
        Path(__file__).with_name("import_bacteria_id.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(source_dir: Path, source: str):
    """按原始行序产出 (类别名, 类内序号, 强度行)。"""
    x, y = importer.load_pair(source_dir, source)
    counters: dict[int, int] = {}
    for index in range(len(x)):
        label = int(y[index])
        offset = counters.get(label, 0)
        counters[label] = offset + 1
        yield importer.STRAIN_NAMES[label], offset, x[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打包 bacteriaID 四阶段 data.npz")
    parser.add_argument("--output", default=str(PROJECT_ROOT / DATA_ARCHIVE_NAME))
    args = parser.parse_args(argv)

    source_dir = importer.DEFAULT_SOURCE
    experiment = importer.EXPERIMENTS["full"]
    dataset_dir = importer.DATASET_ROOT / str(experiment["directory"])
    init_sources = tuple(experiment["init_sources"])
    test_source = str(experiment["test_source"])
    native_axis = np.asarray(np.load(source_dir / "wavenumbers.npy")[::-1], dtype=np.float32)
    target_axis = np.linspace(600.0, 1800.0, 896, dtype=np.float32)

    stage_names = ["init", "init_test", "train", "test"]
    stage_roots = [f"dataset/{dataset_dir.name}/{name}" for name in stage_names]
    sample_stages: list[int] = []
    paths: list[str] = []
    offsets: list[int] = [0]
    wavenumber_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []

    def append_sample(stage_index: int, relative: Path, axis: np.ndarray, values: np.ndarray) -> None:
        paths.append(relative.as_posix())
        sample_stages.append(stage_index)
        wavenumber_parts.append(axis)
        intensity_parts.append(np.asarray(values, dtype=np.float32))
        offsets.append(offsets[-1] + axis.size)

    # init：原生 1000 点
    init_samples = [
        (source, name, offset, values)
        for source in init_sources
        for name, offset, values in _rows(source_dir, source)
    ]
    for source, name, offset, values in init_samples:
        append_sample(
            0,
            Path(name) / importer.batch_name(name, source) / f"s{offset:05d}.arc_data",
            native_axis,
            values,
        )

    # init_test：原生 1000 点
    for name, offset, values in _rows(source_dir, test_source):
        append_sample(
            1,
            Path(importer.test_folder_name(name)) / f"s{offset:05d}.arc_data",
            native_axis,
            values,
        )

    # train / test：模型网格 600–1800、896 点
    for source, name, offset, values in init_samples:
        append_sample(
            2,
            Path(name) / f"{source}_s{offset:05d}.arc_data",
            target_axis,
            np.interp(target_axis, native_axis, values),
        )
    for name, offset, values in _rows(source_dir, test_source):
        append_sample(
            3,
            Path(importer.test_folder_name(name)) / f"test_s{offset:05d}.arc_data",
            target_axis,
            np.interp(target_axis, native_axis, values),
        )

    config_files = [
        (f"dataset/{dataset_dir.name}/dataset.json", dataset_dir / "dataset.json"),
        (
            f"dataset/{dataset_dir.name}/.raman_pipeline/dataset_mapping.json",
            dataset_dir / ".raman_pipeline" / "dataset_mapping.json",
        ),
    ]

    output = Path(args.output).resolve()
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                version=np.asarray(ARCHIVE_VERSION),
                created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
                stage_names=np.asarray(stage_names, dtype=np.str_),
                stage_roots=np.asarray(stage_roots, dtype=np.str_),
                sample_stages=np.asarray(sample_stages, dtype=np.int16),
                offsets=np.asarray(offsets, dtype=np.int64),
                paths=np.asarray(paths, dtype=np.str_),
                wavenumbers=np.concatenate(wavenumber_parts),
                intensities=np.concatenate(intensity_parts),
                config_paths=np.asarray([item[0] for item in config_files], dtype=np.str_),
                config_contents=np.asarray(
                    [item[1].read_text(encoding="utf-8") for item in config_files],
                    dtype=np.str_,
                ),
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    counts = {
        name: int(np.count_nonzero(np.asarray(sample_stages) == index))
        for index, name in enumerate(stage_names)
    }
    print(f"{output}  {output.stat().st_size / 1048576:.1f} MB")
    for name in stage_names:
        print(f"  {name}: {counts[name]}")
    return 0


importer = _load_importer()


if __name__ == "__main__":
    sys.exit(main())
