import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EPS = 1e-8

def read_arc_data(path):
    """Read .arc_data/.txt: two columns (wavenumber, intensity)."""
    wn, sp = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                wn.append(float(parts[0]))
                sp.append(float(parts[1]))
            except Exception:
                continue
    return np.array(wn), np.array(sp)


def build_wn_ref(cut_min, cut_max, target_points):
    return np.linspace(cut_min, cut_max, target_points)


def build_valid_mask(wn, bad_bands):
    if not bad_bands:
        return None
    valid_mask = np.ones_like(wn, dtype=bool)
    for b0, b1 in bad_bands:
        valid_mask &= ~((wn >= b0) & (wn <= b1))
    return valid_mask


def asls_baseline(spectrum, lam=1e5, p=0.01, niter=10, valid_mask=None):
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        w[~valid_mask] = 0.0

    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D @ D.T
        baseline = spsolve(Z, w * spectrum)
        w = p * (spectrum > baseline) + (1 - p) * (spectrum < baseline)
        if valid_mask is not None:
            w[~valid_mask] = 0.0

    return baseline


def snv(X, eps=EPS):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        mean = X.mean()
        std = max(X.std(), eps)
        return (X - mean) / std
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / (std + eps)


def snv_masked(spectra, masks, eps=EPS):
    spectra = spectra.astype(np.float32)
    masks = masks.astype(bool)
    out = np.full_like(spectra, np.nan)

    for i in range(spectra.shape[0]):
        valid = masks[i]
        if not np.any(valid):
            continue
        mean = spectra[i, valid].mean()
        std = max(spectra[i, valid].std(), eps)
        out[i, valid] = (spectra[i, valid] - mean) / std

    return out


def minmax_normalize(X, eps=EPS):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        min_v = np.min(X)
        max_v = np.max(X)
        denom = max(max_v - min_v, eps)
        return (X - min_v) / denom
    min_v = np.min(X, axis=1, keepdims=True)
    max_v = np.max(X, axis=1, keepdims=True)
    denom = np.maximum(max_v - min_v, eps)
    return (X - min_v) / denom


def normalize_for_plot(spectra, method):
    method = method.lower()
    if method == "minmax":
        return minmax_normalize(spectra)
    if method == "snv":
        return snv(spectra)
    raise ValueError(f"Unknown norm method: {method}")


def preprocess_single_spectrum(
    wn,
    sp,
    cut_min,
    cut_max,
    wn_ref,
    bad_bands,
    asls_lam,
    asls_p,
    asls_max_iter,
):
    valid_mask = build_valid_mask(wn, bad_bands)

    baseline = asls_baseline(
        sp,
        lam=asls_lam,
        p=asls_p,
        niter=asls_max_iter,
        valid_mask=valid_mask,
    )
    sp_bc = sp - baseline

    mask_cut = (wn >= cut_min) & (wn <= cut_max)
    wn_cut = wn[mask_cut]
    sp_cut = sp_bc[mask_cut]

    if wn_cut.size < 10:
        return None, None

    sp_interp = np.interp(wn_ref, wn_cut, sp_cut)

    if bad_bands:
        keep_mask = build_valid_mask(wn_ref, bad_bands)
        wn_ref = wn_ref[keep_mask]
        sp_interp = sp_interp[keep_mask]

    return wn_ref, sp_interp


def save_mean_plot(wn, spectra, out_path, norm_method, bad_bands, title):
    spectra_norm = normalize_for_plot(spectra, norm_method)
    mean_spec = np.mean(spectra_norm, axis=0)
    std_spec = np.std(spectra_norm, axis=0)

    plt.figure(figsize=(10, 5))
    std_proxy = Patch(facecolor="C0", alpha=0.3, label="+/-1 std range")

    if bad_bands:
        for b0, b1 in bad_bands:
            plt.axvspan(
                b0,
                b1,
                color="gray",
                alpha=0.2,
                label="CCD-affected region" if (b0, b1) == bad_bands[0] else None,
            )

    plt.plot(wn, mean_spec, label=f"Mean spectrum {norm_method}")
    plt.fill_between(wn, mean_spec - std_spec, mean_spec + std_spec, alpha=0.3)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel(f"{norm_method} intensity")
    plt.xlim([wn.min(), wn.max()])
    plt.legend(handles=[std_proxy] + plt.gca().get_legend_handles_labels()[0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
