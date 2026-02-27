import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from preprocess_common import (
    build_wn_ref,
    preprocess_single_spectrum,
    read_arc_data,
    save_mean_plot,
)

# Preprocess parameters
CUT_MIN = 600
CUT_MAX = 1800
TARGET_POINTS = 896
WN_REF = build_wn_ref(CUT_MIN, CUT_MAX, TARGET_POINTS)

ASLS_LAM = 1e5
ASLS_P = 0.01
ASLS_MAX_ITER = 10
BAD_BANDS = [(905, 940.0)]

NORM_METHOD = "snv"

# Paths
ROOT_TEST_RAW = "测试菌"
ROOT_TEST_PROC = "../dataset_test_耐药菌"
ROOT_TEST_FIG = "dataset_test_fig"

def iter_class_dirs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [f for f in files if f.lower().endswith(".arc_data")]
        if arc_files:
            yield root, arc_files


def preprocess_test_dataset(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ROOT_TEST_FIG, exist_ok=True)

    processed = 0
    skipped = 0
    errored = 0

    for root, arc_files in iter_class_dirs(input_dir):
        spectra = []
        wn_list = []

        for fname in arc_files:
            in_path = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, input_dir)
            out_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)

            try:
                wn, sp = read_arc_data(in_path)
                wn_u, sp_u = preprocess_single_spectrum(
                    wn,
                    sp,
                    cut_min=CUT_MIN,
                    cut_max=CUT_MAX,
                    wn_ref=WN_REF,
                    bad_bands=BAD_BANDS,
                    asls_lam=ASLS_LAM,
                    asls_p=ASLS_P,
                    asls_max_iter=ASLS_MAX_ITER,
                )
                if wn_u is None:
                    print(f"[SKIP] {os.path.relpath(in_path, input_dir)} (empty after cut)")
                    skipped += 1
                    continue

                with open(out_path, "w", encoding="utf-8") as f:
                    for w, s in zip(wn_u, sp_u):
                        f.write(f"{w:.3f} {s:.3f}\n")

                spectra.append(sp_u)
                wn_list.append(wn_u)
                processed += 1
            except Exception as e:
                print(f"[ERROR] {os.path.relpath(in_path, input_dir)}: {e}")
                errored += 1

        if spectra:
            spectra_arr = np.vstack(spectra)
            wn_ref = wn_list[0]

            rel_dir = os.path.relpath(root, input_dir)
            rel_parent = os.path.dirname(rel_dir)
            fig_dir = (
                ROOT_TEST_FIG
                if rel_parent == "."
                else os.path.join(ROOT_TEST_FIG, rel_parent)
            )
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{os.path.basename(root)}.png")

            save_mean_plot(
                wn=wn_ref,
                spectra=spectra_arr,
                out_path=fig_path,
                norm_method=NORM_METHOD,
                bad_bands=BAD_BANDS,
                title=f"{rel_dir} (mean +/- std)",
            )

            print(f"  Mean spectrum saved: {fig_path}")

    print(
        "Test dataset preprocessing finished. "
        f"Processed={processed}, Skipped={skipped}, Error={errored}"
    )


if __name__ == "__main__":
    preprocess_test_dataset(input_dir=ROOT_TEST_RAW, output_dir=ROOT_TEST_PROC)
