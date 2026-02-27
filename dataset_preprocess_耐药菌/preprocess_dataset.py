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

MIN_SAMPLES_PER_CLASS = 8
NORM_METHOD = "snv"

# PCA outlier removal (set PCA_ENABLED = False to skip)
PCA_ENABLED = True
# float in (0,1] keeps variance; int keeps components
PCA_COMPONENTS = 0.95
PCA_CENTER = True
# remove top X% by reconstruction error (e.g. 0.05 = top 5%)
PCA_OUTLIER_RATIO = 0.01

# Paths
ROOT_PROCESS_RAW = "dataset_raw"
ROOT_PROCESS_CLEAN = "../dataset_train_耐药菌"
ROOT_FIGURE = "dataset_train_fig"
LOG_PATH = os.path.join(CURRENT_DIR, "log.txt")

def iter_class_dirs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [f for f in files if f.lower().endswith(".arc_data")]
        if arc_files:
            yield root, arc_files

def pca_reconstruct_and_error(spectra, n_components=0.95, center=True):
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[0] < 2:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    mean = spectra.mean(axis=0, keepdims=True) if center else 0.0
    spectra_c = spectra - mean

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(spectra_c, full_matrices=False)
    if S.size == 0:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    var = (S ** 2) / max(spectra_c.shape[0] - 1, 1)
    total_var = var.sum()
    if total_var <= 0:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    if isinstance(n_components, float) and 0 < n_components <= 1:
        ratio_cum = np.cumsum(var) / total_var
        k = int(np.searchsorted(ratio_cum, n_components) + 1)
    else:
        try:
            k = int(n_components)
        except Exception:
            k = 1

    k = max(1, min(k, Vt.shape[0]))
    spectra_rec = (U[:, :k] * S[:k]) @ Vt[:k, :]
    if center:
        spectra_rec = spectra_rec + mean

    errors = np.mean((spectra - spectra_rec) ** 2, axis=1)

    return spectra_rec, k, errors

def log_removed_samples(label, filenames, errors, threshold, log_path):
    if not filenames:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{label}] removed {len(filenames)} samples, "
            f"threshold={threshold:.6f}\n"
        )
        for fname, err in zip(filenames, errors):
            f.write(f"  {fname}\t{float(err):.6f}\n")

def preprocess_train_dataset():
    if not os.path.isdir(ROOT_PROCESS_RAW):
        raise FileNotFoundError(f"Missing input dir: {ROOT_PROCESS_RAW}")

    os.makedirs(ROOT_PROCESS_CLEAN, exist_ok=True)
    os.makedirs(ROOT_FIGURE, exist_ok=True)

    for cls_raw_dir, arc_files in iter_class_dirs(ROOT_PROCESS_RAW):
        rel_dir = os.path.relpath(cls_raw_dir, ROOT_PROCESS_RAW)
        label = rel_dir if rel_dir != "." else os.path.basename(ROOT_PROCESS_RAW)
        label_display = label.replace(os.sep, "/")

        print(f"\n=== Processing: {label_display} ===")

        spectra = []
        wn_list = []
        filenames = []

        for fname in arc_files:
            wn, sp = read_arc_data(os.path.join(cls_raw_dir, fname))
            if wn.size == 0 or sp.size == 0:
                continue

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
                continue

            spectra.append(sp_u)
            wn_list.append(wn_u)
            filenames.append(fname)

        if len(spectra) < MIN_SAMPLES_PER_CLASS:
            print(f"  Skip: too few samples ({len(spectra)}) in {label_display}")
            continue

        spectra_arr = np.vstack(spectra)
        if PCA_ENABLED and spectra_arr.shape[0] > 1:
            _, pca_k, errors = pca_reconstruct_and_error(
                spectra_arr,
                n_components=PCA_COMPONENTS,
                center=PCA_CENTER,
            )
            if pca_k > 0:
                ratio = float(PCA_OUTLIER_RATIO)
                ratio = max(0.0, min(ratio, 1.0))
                if ratio <= 0.0:
                    thresh = float("inf")
                    keep_mask = np.ones_like(errors, dtype=bool)
                else:
                    thresh = float(np.quantile(errors, 1.0 - ratio))
                    keep_mask = errors <= thresh
                removed = int((~keep_mask).sum())
                print(
                    f"  PCA outlier removal: k={pca_k}, "
                    f"threshold={thresh:.6f}, removed={removed}"
                )

                if removed > 0:
                    removed_mask = ~keep_mask
                    removed_files = [f for f, k in zip(filenames, removed_mask) if k]
                    removed_errors = errors[removed_mask]
                    log_removed_samples(
                        label_display,
                        removed_files,
                        removed_errors,
                        thresh,
                        LOG_PATH,
                    )
                    spectra_arr = spectra_arr[keep_mask]
                    filenames = [f for f, k in zip(filenames, keep_mask) if k]
                    wn_list = [w for w, k in zip(wn_list, keep_mask) if k]

        if len(spectra_arr) < MIN_SAMPLES_PER_CLASS:
            print(f"  Skip: too few samples ({len(spectra_arr)}) in {label_display}")
            continue
        save_dir = os.path.join(ROOT_PROCESS_CLEAN, rel_dir)
        os.makedirs(save_dir, exist_ok=True)

        for fname, wn_u, sp_u in zip(filenames, wn_list, spectra_arr):
            out_path = os.path.join(save_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                for wv, val in zip(wn_u, sp_u):
                    f.write(f"{wv:.3f} {val:.3f}\n")

        rel_parent = os.path.dirname(rel_dir)
        fig_dir = (
            ROOT_FIGURE if rel_parent == "." else os.path.join(ROOT_FIGURE, rel_parent)
        )
        os.makedirs(fig_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_dir, f"{os.path.basename(cls_raw_dir)}.png")
        title = " - ".join(label.split(os.sep)) + " (mean +/- std)"
        save_mean_plot(
            wn=wn_list[0],
            spectra=spectra_arr,
            out_path=fig_save_path,
            norm_method=NORM_METHOD,
            bad_bands=BAD_BANDS,
            title=title,
        )

        print(f"  Mean spectrum saved: {fig_save_path}")

    print("\nTraining dataset preprocessing finished:")
    print(f"- Clean spectra: {ROOT_PROCESS_CLEAN}")
    print(f"- Mean plots: {ROOT_FIGURE}")


if __name__ == "__main__":
    preprocess_train_dataset()
