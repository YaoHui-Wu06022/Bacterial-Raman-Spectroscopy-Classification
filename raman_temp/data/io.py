"""`.arc_data`、`init.npz` 与数据集叶子目录的读写工具。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from raman_temp.core.paths import resolve_path


PACK_EXT = ".npz"


def iter_arc_dirs(root_dir: Path | str):
    """递归返回包含 `.arc_data` 文件的目录及其排序后的文件名。"""
    for root, directories, filenames in os.walk(os.fspath(root_dir)):
        directories.sort()
        arc_files = sorted(name for name in filenames if name.lower().endswith(".arc_data"))
        if arc_files:
            yield Path(root), arc_files


def read_arc_data(path: Path | str):
    """读取两列文本光谱，忽略格式错误行。"""
    wavenumbers, intensities = [], []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) != 2:
                continue
            try:
                wavenumbers.append(float(fields[0]))
                intensities.append(float(fields[1]))
            except ValueError:
                continue
    return np.asarray(wavenumbers), np.asarray(intensities)


def load_arc_intensity(path: Path | str, dtype=np.float32):
    """读取单个 `.arc_data` 文件的强度列。"""
    data = np.atleast_2d(np.loadtxt(path, dtype=dtype))
    return data[:, 1].astype(dtype, copy=False)


def write_arc_data(path: Path | str, wavenumbers, intensities, fmt: str = "%.8f"):
    """以两列文本格式写出一条光谱，并自动创建父目录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, np.column_stack([wavenumbers, intensities]), fmt=[fmt, fmt])


def is_packed_path(path: Path | str) -> bool:
    """判断路径是否为存在的 `.npz` 数据归档。"""
    return Path(path).is_file() and str(path).lower().endswith(PACK_EXT)


class PackedArcDataset:
    """�?`init.npz` 按原始样本边界恢复光谱数组。"""

    def __init__(self, npz_path: Path | str):
        if not is_packed_path(npz_path):
            raise FileNotFoundError(f"Missing packed file: {npz_path}")
        data = np.load(npz_path)
        self.root_name = str(data["root_name"][0]) if "root_name" in data else "init"
        self.paths = data["paths"].tolist()
        self.offsets = data["offsets"]
        self.lengths = data["lengths"]
        self.wn_all = data["wn_all"]
        self.sp_all = data["sp_all"]

    def __len__(self):
        return len(self.paths)

    def get(self, index: int):
        """返回指定样本的相对路径、波数轴和强度数组。"""
        start = int(self.offsets[index])
        end = start + int(self.lengths[index])
        return self.paths[index], self.wn_all[start:end], self.sp_all[start:end]

    def iter_samples(self):
        """按归档顺序迭代全部样本。"""
        for index in range(len(self)):
            yield self.get(index)


def pack_init(input_dir: Path | str, output_path: Path | str, is_verbose: bool = True):
    """�?init 目录打包为可迁移�?`.npz` 归档。"""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")
    paths, offsets, wavenumber_chunks, intensity_chunks = [], [0], [], []
    for directory, filenames in iter_arc_dirs(input_dir):
        for filename in filenames:
            full_path = directory / filename
            wavenumbers, intensities = read_arc_data(full_path)
            if not wavenumbers.size or not intensities.size:
                continue
            paths.append(full_path.relative_to(input_dir).as_posix())
            wavenumber_chunks.append(wavenumbers.astype(np.float32))
            intensity_chunks.append(intensities.astype(np.float32))
            offsets.append(offsets[-1] + wavenumbers.size)
    if not paths:
        raise RuntimeError(f"No .arc_data files found under {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.asarray(offsets, dtype=np.int64)
    np.savez_compressed(
        output_path,
        root_name=np.asarray([input_dir.resolve().name]),
        paths=np.asarray(paths),
        offsets=offsets[:-1],
        lengths=np.diff(offsets),
        wn_all=np.concatenate(wavenumber_chunks),
        sp_all=np.concatenate(intensity_chunks),
    )
    if is_verbose:
        print(f"[Pack] samples={len(paths)}, saved={output_path}")


def unpack_init(npz_path: Path | str, output_dir: Path | str, is_verbose: bool = True):
    """�?`init.npz` 恢复为目录树，不覆盖已有同名文件。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refuse to unpack into non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for relative_path, wavenumbers, intensities in PackedArcDataset(npz_path).iter_samples():
        write_arc_data(output_dir / relative_path, wavenumbers, intensities)
        restored += 1
    if is_verbose:
        print(f"[Unpack] samples={restored}, restored={output_dir}")


def resolve_init_input(base_dir: Path | str, profile):
    """优先选择 init 目录，缺失时回退�?`init.npz`。"""
    root_init = resolve_path(profile.root_init, base_dir)
    root_init_pack = resolve_path(profile.root_init_pack, base_dir)
    if root_init.is_dir() or is_packed_path(root_init):
        return root_init
    if is_packed_path(root_init_pack):
        return root_init_pack
    raise FileNotFoundError(f"Missing input dir/file: {root_init}")


def iter_init_groups(input_path: Path | str):
    """按叶子目录分组迭代目录或 `init.npz` 形式的原始样本。"""
    input_path = Path(input_path)
    if input_path.is_dir():
        for leaf_dir, filenames in iter_arc_dirs(input_path):
            samples = [
                (filename, *read_arc_data(leaf_dir / filename))
                for filename in filenames
            ]
            yield leaf_dir.relative_to(input_path), leaf_dir.name, samples
        return

    groups = {}
    packed = PackedArcDataset(input_path)
    for relative_path, wavenumbers, intensities in packed.iter_samples():
        normalized = relative_path.replace("\\", "/")
        relative_dir = Path(os.path.dirname(normalized) or ".")
        key = relative_dir.as_posix()
        group = groups.setdefault(
            key,
            {
                "rel_dir": relative_dir,
                "leaf_name": packed.root_name if relative_dir == Path(".") else relative_dir.name,
                "samples": [],
            },
        )
        group["samples"].append((os.path.basename(normalized), wavenumbers, intensities))
    for group in groups.values():
        yield group["rel_dir"], group["leaf_name"], group["samples"]
