"""reference 专用：只做裁切、坏段剔除和插值"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset_process.common import build_valid_mask, build_wn_ref, read_arc_data
from dataset_process.pipeline import iter_arc_dirs, write_arc_data
from dataset_process.profiles import COMMON_BAD_BANDS


REFERENCE_DIR = Path("dataset/reference")
INPUT_DIR = REFERENCE_DIR / "dataset_train_raw"
OUTPUT_DIR = REFERENCE_DIR / "dataset_train"

CUT_MIN = 600.0
CUT_MAX = 1800.0
TARGET_POINTS = 896
BAD_BANDS = COMMON_BAD_BANDS
OUTPUT_FMT = "%.3f"
PROGRESS_EVERY = 100


def cut_badband_interp_single(wn, sp, wn_ref):
    """单条光谱处理：裁切范围、移除坏段、插值到统一波数轴"""
    mask_cut = (wn >= CUT_MIN) & (wn <= CUT_MAX)
    wn_cut = wn[mask_cut]
    sp_cut = sp[mask_cut]

    if wn_cut.size < 2:
        return None, None

    src_keep_mask = build_valid_mask(wn_cut, BAD_BANDS)
    if src_keep_mask is not None:
        wn_cut = wn_cut[src_keep_mask]
        sp_cut = sp_cut[src_keep_mask]

    target_keep_mask = build_valid_mask(wn_ref, BAD_BANDS)
    if target_keep_mask is not None:
        wn_ref = wn_ref[target_keep_mask]

    if wn_cut.size < 2 or wn_ref.size == 0:
        return None, None

    sp_interp = np.interp(wn_ref, wn_cut, sp_cut)
    return wn_ref, sp_interp


def process_reference():
    """把 dataset/reference/dataset_train_raw 处理后输出到 dataset/reference/dataset_train"""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Missing input dir: {INPUT_DIR}")

    wn_ref = build_wn_ref(CUT_MIN, CUT_MAX, TARGET_POINTS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = list(iter_arc_dirs(INPUT_DIR))
    total_files = sum(len(arc_files) for _, arc_files in groups)
    print(f"Reference input files: {total_files}")

    processed = 0
    skipped = 0
    errored = 0
    seen = 0

    for root, arc_files in groups:
        rel_dir = root.relative_to(INPUT_DIR)
        out_dir = OUTPUT_DIR / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] {rel_dir.as_posix()} ({len(arc_files)} files)")

        for filename in arc_files:
            seen += 1
            in_path = root / filename
            out_path = out_dir / filename
            try:
                wn, sp = read_arc_data(in_path)
                wn_u, sp_u = cut_badband_interp_single(wn, sp, wn_ref)
                if wn_u is None:
                    skipped += 1
                    print(f"[SKIP] {in_path.relative_to(INPUT_DIR).as_posix()}")
                    continue

                write_arc_data(out_path, wn_u, sp_u, fmt=OUTPUT_FMT)
                processed += 1
            except Exception as exc:
                errored += 1
                rel_path = in_path.relative_to(INPUT_DIR).as_posix()
                print(f"[ERROR] {rel_path}: {exc}")

            if seen % PROGRESS_EVERY == 0 or seen == total_files:
                print(
                    f"[PROGRESS] {seen}/{total_files} "
                    f"processed={processed}, skipped={skipped}, error={errored}"
                )

    print("Reference cut + bad-band removal + interpolation finished")
    print(f"- input:   {INPUT_DIR}")
    print(f"- output:  {OUTPUT_DIR}")
    print(f"- processed={processed}, skipped={skipped}, error={errored}")


if __name__ == "__main__":
    process_reference()
