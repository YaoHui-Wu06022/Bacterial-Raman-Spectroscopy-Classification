import os
import shutil
import re

from packed_dataset import (
    is_packed_path,
    PackedArcDataset,
    write_arc_data,
)

ROOT_INIT = r"dataset_init"
ROOT_INIT_PACK = r"dataset_init.npz"
ROOT_PROCESS_RAW = r"dataset_raw"
os.makedirs(ROOT_PROCESS_RAW, exist_ok=True)


def get_prefix(name):
    """提取前缀字母，例如 EAB01 -> EAB。"""
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) if m else None


def iter_leaf_dirs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [f for f in files if f.lower().endswith(".arc_data")]
        if arc_files:
            yield root, arc_files


def classify_dataset():
    input_path = None
    if os.path.isdir(ROOT_INIT):
        input_path = ROOT_INIT
    elif is_packed_path(ROOT_INIT):
        input_path = ROOT_INIT
    elif is_packed_path(ROOT_INIT_PACK):
        input_path = ROOT_INIT_PACK

    if input_path is None:
        raise FileNotFoundError(f"Missing input dir/file: {ROOT_INIT}")

    copied = 0
    if os.path.isdir(input_path):
        for leaf_dir, arc_files in iter_leaf_dirs(input_path):
            rel_dir = os.path.relpath(leaf_dir, input_path)
            rel_parent = os.path.dirname(rel_dir)
            leaf_name = os.path.basename(leaf_dir)

            prefix = get_prefix(leaf_name)
            target_cls = prefix if prefix else leaf_name

            if rel_parent == ".":
                target_dir = os.path.join(ROOT_PROCESS_RAW, target_cls)
            else:
                target_dir = os.path.join(ROOT_PROCESS_RAW, rel_parent, target_cls)
            os.makedirs(target_dir, exist_ok=True)

            for fname in arc_files:
                src = os.path.join(leaf_dir, fname)
                newname = f"{leaf_name}_{fname}"
                dst = os.path.join(target_dir, newname)
                shutil.copy(src, dst)
                copied += 1
    else:
        pack = PackedArcDataset(input_path)
        for rel_path, wn, sp in pack.iter_samples():
            rel_path = rel_path.replace("\\", "/")
            rel_dir = os.path.dirname(rel_path) if rel_path else "."
            if rel_dir == "":
                rel_dir = "."
            rel_parent = os.path.dirname(rel_dir)

            if rel_dir in ("", "."):
                leaf_name = pack.root_name
            else:
                leaf_name = os.path.basename(rel_dir)

            prefix = get_prefix(leaf_name)
            target_cls = prefix if prefix else leaf_name

            if rel_parent == "." or rel_parent == "":
                target_dir = os.path.join(ROOT_PROCESS_RAW, target_cls)
            else:
                target_dir = os.path.join(ROOT_PROCESS_RAW, rel_parent, target_cls)
            os.makedirs(target_dir, exist_ok=True)

            fname = os.path.basename(rel_path)
            newname = f"{leaf_name}_{fname}"
            dst = os.path.join(target_dir, newname)
            write_arc_data(dst, wn, sp)
            copied += 1

    print(f"Stage 1 complete: copied {copied} files into {ROOT_PROCESS_RAW}")


if __name__ == "__main__":
    classify_dataset()
