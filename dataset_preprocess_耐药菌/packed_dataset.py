import os
import numpy as np

from preprocess_common import read_arc_data

PACK_EXT = ".npz"


def is_packed_path(path):
    return os.path.isfile(path) and path.lower().endswith(PACK_EXT)


def pack_dataset_init(input_dir, output_path, verbose=True):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    root_name = os.path.basename(os.path.abspath(input_dir))
    paths = []
    offsets = [0]
    wn_chunks = []
    sp_chunks = []

    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        files.sort()
        arc_files = [f for f in files if f.lower().endswith(".arc_data")]
        if not arc_files:
            continue

        for fname in arc_files:
            full_path = os.path.join(root, fname)
            wn, sp = read_arc_data(full_path)
            if wn.size == 0 or sp.size == 0:
                continue

            rel_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
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


def write_arc_data(path, wn, sp, fmt="%.8f"):
    arr = np.column_stack([wn, sp])
    np.savetxt(path, arr, fmt=[fmt, fmt])


class PackedArcDataset:
    def __init__(self, npz_path):
        if not is_packed_path(npz_path):
            raise FileNotFoundError(f"Missing packed file: {npz_path}")
        data = np.load(npz_path)
        self.root_name = str(data["root_name"][0]) if "root_name" in data else "dataset_init"
        self.paths = data["paths"].tolist()
        self.offsets = data["offsets"]
        self.lengths = data["lengths"]
        self.wn_all = data["wn_all"]
        self.sp_all = data["sp_all"]

    def __len__(self):
        return len(self.paths)

    def get(self, idx):
        start = int(self.offsets[idx])
        length = int(self.lengths[idx])
        end = start + length
        rel_path = self.paths[idx]
        wn = self.wn_all[start:end]
        sp = self.sp_all[start:end]
        return rel_path, wn, sp

    def iter_samples(self):
        for i in range(len(self.paths)):
            yield self.get(i)
