"""数据文件读写、归档和离线构建输入源"""

import os
from pathlib import Path

import numpy as np

from raman.tool.dataset import iter_arc_dirs
from raman.tool.path import resolve_under_base

PACK_EXT = ".npz"


def read_arc_data(path):
    """读取两列文本光谱文件，返回波数和强度数组"""
    wn, sp = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                wn.append(float(parts[0]))
                sp.append(float(parts[1]))
            except Exception:
                continue
    return np.asarray(wn), np.asarray(sp)


def load_arc_intensity(path):
    """读取单个 .arc_data 文件的强度列"""
    data = np.loadtxt(path, dtype=np.float32)
    data = np.atleast_2d(data)
    return data[:, 1].astype(np.float32, copy=False)


def write_arc_data(path, wn, sp, fmt="%.8f"):
    """把一条光谱写回两列文本格式"""
    arr = np.column_stack([wn, sp])
    np.savetxt(path, arr, fmt=[fmt, fmt])


def is_packed_path(path):
    """判断一个路径是否是可读取的打包数据文件"""
    return os.path.isfile(path) and str(path).lower().endswith(PACK_EXT)

class PackedArcDataset:
    """从 init.npz 中按样本迭代恢复光谱内容"""

    def __init__(self, npz_path):
        if not is_packed_path(npz_path):
            raise FileNotFoundError(f"Missing packed file: {npz_path}")
        data = np.load(npz_path)
        self.root_name = (
            str(data["root_name"][0]) if "root_name" in data else "init"
        )
        self.paths = data["paths"].tolist()
        self.offsets = data["offsets"]
        self.lengths = data["lengths"]
        self.wn_all = data["wn_all"]
        self.sp_all = data["sp_all"]

    def __len__(self):
        return len(self.paths)

    def get(self, index):
        start = int(self.offsets[index])
        length = int(self.lengths[index])
        end = start + length
        rel_path = self.paths[index]
        wn = self.wn_all[start:end]
        sp = self.sp_all[start:end]
        return rel_path, wn, sp

    def iter_samples(self):
        for index in range(len(self.paths)):
            yield self.get(index)

def pack_init(input_dir, output_path, verbose=True):
    """把 init 下的散落光谱打包成一个 npz，便于迁移和归档"""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    root_name = input_dir.resolve().name
    paths = []
    offsets = [0]
    wn_chunks = []
    sp_chunks = []

    for root, arc_files in iter_arc_dirs(input_dir):
        for fname in arc_files:
            full_path = root / fname
            wn, sp = read_arc_data(full_path)
            if wn.size == 0 or sp.size == 0:
                continue

            rel_path = full_path.relative_to(input_dir).as_posix()
            paths.append(rel_path)

            wn = wn.astype(np.float32)
            sp = sp.astype(np.float32)
            wn_chunks.append(wn)
            sp_chunks.append(sp)
            offsets.append(offsets[-1] + wn.size)

    if not paths:
        raise RuntimeError(f"No .arc_data files found under {input_dir}")

    wn_all = np.concatenate(wn_chunks, axis=0)
    sp_all = np.concatenate(sp_chunks, axis=0)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    offsets = offsets[:-1]

    np.savez_compressed(
        output_path,
        root_name=np.asarray([root_name]),
        paths=np.asarray(paths),
        offsets=offsets,
        lengths=lengths,
        wn_all=wn_all,
        sp_all=sp_all,
    )

    if verbose:
        total = int(wn_all.size)
        print(f"[Pack] samples={len(paths)}, points={total}, saved={output_path}")

def unpack_init(npz_path, output_dir, verbose=True):
    """把 init.npz 恢复回目录树，便于重新检查和手工处理"""
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    packed = PackedArcDataset(npz_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    for rel_path, wn, sp in packed.iter_samples():
        out_path = output_dir / Path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_arc_data(out_path, wn, sp)
        restored += 1

    if verbose:
        print(f"[Unpack] samples={restored}, restored={output_dir}")


def resolve_init_input(base_dir, profile):
    """优先解析 init 目录，其次回退到打包后的 init.npz"""
    root_init = resolve_under_base(base_dir, profile.root_init)
    root_init_pack = resolve_under_base(base_dir, profile.root_init_pack)

    if root_init.is_dir():
        return root_init
    if is_packed_path(root_init):
        return root_init
    if is_packed_path(root_init_pack):
        return root_init_pack

    raise FileNotFoundError(f"Missing input dir/file: {root_init}")


def iter_init_groups(input_path):
    """按叶子目录分组迭代原始样本，兼容目录输入和 npz 打包输入"""
    input_path = Path(input_path)

    if input_path.is_dir():
        for leaf_dir, arc_files in iter_arc_dirs(input_path):
            rel_dir = leaf_dir.relative_to(input_path)
            samples = []
            for fname in arc_files:
                wn, sp = read_arc_data(leaf_dir / fname)
                samples.append((fname, wn, sp))
            yield rel_dir, leaf_dir.name, samples
        return

    packed = PackedArcDataset(input_path)
    grouped = {}

    for rel_path, wn, sp in packed.iter_samples():
        normalized_rel_path = rel_path.replace("\\", "/")
        rel_dir = Path(os.path.dirname(normalized_rel_path) or ".")
        group_key = rel_dir.as_posix()
        if group_key not in grouped:
            grouped[group_key] = {
                "rel_dir": rel_dir,
                "leaf_name": packed.root_name if rel_dir == Path(".") else rel_dir.name,
                "samples": [],
            }
        grouped[group_key]["samples"].append(
            (os.path.basename(normalized_rel_path), wn, sp)
        )

    for group in grouped.values():
        yield group["rel_dir"], group["leaf_name"], group["samples"]

