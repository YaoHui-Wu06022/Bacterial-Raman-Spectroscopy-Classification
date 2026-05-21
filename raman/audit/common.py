"""数据审核公共工具"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from raman.data.build import DEFAULT_PIPELINE_CONFIG, _cosmic_ray_kwargs, resolve_pipeline_config
from raman.data.offline import preprocess_single_spectrum
from raman.data.profiles import get_dataset_dir, get_profile
from raman.data.spectrum import build_valid_mask, read_arc_data, snv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def safe_name(name: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("_") or "sample"


def prefix_of(name: str) -> str:
    """提取小文件夹名前缀，例如 KAE01 -> KAE"""
    letters = []
    for char in str(name):
        if char.isalpha():
            letters.append(char)
        else:
            break
    return "".join(letters) or str(name)


def resolve_dataset(dataset: str, project_root: Path = PROJECT_ROOT):
    profile = get_profile(dataset)
    return profile, get_dataset_dir(profile, project_root)


def robust_scale(values):
    """返回中位数和 MAD 鲁棒尺度"""
    values = np.asarray(values, dtype=np.float32)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(values))
    return center, max(scale, 1e-8)


def robust_mad_scale(values, floor=1e-8):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(floor)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= floor:
        scale = float(np.std(values))
    return max(scale, float(floor))


def spectral_corr(a, b):
    """计算光谱相关系数，近似常数谱返回 0 避免 NaN"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.std() <= 1e-8 or b.std() <= 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def corr_many_to_one(arr, spec):
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    return (arr @ spec / max(spec.size, 1)).astype(np.float32, copy=False)


def robust_wave_stats(spectra, min_scale=0.05, floor_fraction=0.25):
    """逐波数计算中位谱和鲁棒尺度"""
    spectra = np.asarray(spectra, dtype=np.float32)
    center = np.median(spectra, axis=0)
    mad = np.median(np.abs(spectra - center), axis=0)
    scale = 1.4826 * mad
    if np.any(scale > 1e-8):
        floor = max(float(np.median(scale[scale > 1e-8])) * float(floor_fraction), float(min_scale))
    else:
        floor = float(min_scale)
    return center, np.maximum(scale, floor)


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float32)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def median_step_cm(wn):
    wn = np.asarray(wn, dtype=np.float32)
    if wn.size < 2:
        return 1.0
    diffs = np.abs(np.diff(wn))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-8)]
    return float(np.median(diffs)) if diffs.size else 1.0


def region_width_cm(wn, start, end):
    if end <= start:
        return 0.0
    step = median_step_cm(wn)
    return float(abs(wn[end - 1] - wn[start]) + step)


def output_wn(cfg):
    wn = cfg.build_wn_ref()
    keep = build_valid_mask(wn, cfg.bad_bands)
    return wn[keep] if keep is not None else wn


def add_bad_band_spans(ax, bad_bands, alpha=0.14):
    for band_min, band_max in bad_bands:
        ax.axvspan(band_min, band_max, color="gray", alpha=alpha)


def _keep_mask_without_bad_bands(wn, bad_bands):
    wn = np.asarray(wn, dtype=np.float32)
    keep = np.ones_like(wn, dtype=bool)
    for band_min, band_max in bad_bands:
        keep &= ~((wn >= band_min) & (wn <= band_max))
    if wn.size >= 2:
        step = median_step_cm(wn)
        gap_breaks = np.where(np.abs(np.diff(wn)) > step * 1.8)[0]
        keep[gap_breaks + 1] = False
    return keep


def plot_segments_without_bad_bands(ax, wn, values, bad_bands, **kwargs):
    """绘制断线光谱，避免坏段灰色区域内出现连线"""
    wn = np.asarray(wn, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    keep = _keep_mask_without_bad_bands(wn, bad_bands)
    label = kwargs.pop("label", None)
    labeled = False
    for start, end in contiguous_regions(keep):
        if end - start >= 2:
            line_label = label if label and not labeled else None
            ax.plot(wn[start:end], values[start:end], label=line_label, **kwargs)
            labeled = True
    add_bad_band_spans(ax, bad_bands)


def fill_between_segments_without_bad_bands(ax, wn, lower, upper, bad_bands, **kwargs):
    wn = np.asarray(wn, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    keep = _keep_mask_without_bad_bands(wn, bad_bands)
    label = kwargs.pop("label", None)
    labeled = False
    for start, end in contiguous_regions(keep):
        if end - start >= 2:
            band_label = label if label and not labeled else None
            ax.fill_between(wn[start:end], lower[start:end], upper[start:end], label=band_label, **kwargs)
            labeled = True
    add_bad_band_spans(ax, bad_bands)


def preprocess_spectrum_for_audit(path, profile, cfg=None, wn_ref=None, include_raw=False):
    """按当前离线流程预处理单谱并返回审核载荷"""
    cfg = resolve_pipeline_config(cfg or DEFAULT_PIPELINE_CONFIG)
    path = Path(path)
    payload = {"path": path, "skip_reason": ""}
    wn, sp = read_arc_data(path)

    if include_raw:
        payload["raw_wn"] = wn
        payload["raw_sp"] = sp

    if wn.size == 0 or sp.size == 0:
        payload["skip_reason"] = "read_failed"
        return payload

    if wn_ref is None:
        wn_ref = cfg.build_wn_ref()

    wn_u, sp_u, cosmic_stats = preprocess_single_spectrum(
        wn,
        sp,
        cut_min=cfg.cut_min,
        cut_max=cfg.cut_max,
        wn_ref=wn_ref,
        bad_bands=cfg.bad_bands,
        baseline_method=cfg.baseline_method,
        baseline_lam=cfg.baseline_lam,
        baseline_asls_p=cfg.baseline_asls_p,
        baseline_max_iter=cfg.baseline_max_iter,
        baseline_fit_min=cfg.baseline_fit_min,
        baseline_fit_max=cfg.baseline_fit_max,
        **_cosmic_ray_kwargs(profile, cfg),
    )
    if wn_u is None or sp_u is None:
        payload["skip_reason"] = "preprocess_failed"
        return payload

    payload.update(
        {
            "wn": wn_u,
            "sp": sp_u,
            "z": snv(sp_u),
            "cosmic_replaced": int(cosmic_stats),
            "cosmic_stats": cosmic_stats,
        }
    )
    return payload


def relative_to_init(path, dataset_dir, init_root, profile):
    path = Path(path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    init_root = Path(init_root).resolve()
    try:
        return path.relative_to(init_root)
    except ValueError:
        pass

    try:
        rel = path.relative_to(dataset_dir)
    except ValueError:
        return Path(path.name)
    if rel.parts and rel.parts[0] == profile.root_init:
        return Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")
    return rel


def resolve_audit_folder(folder, dataset_dir, profile, init_root):
    """解析绝对路径、属名/文件夹名或唯一末级文件夹名"""
    dataset_dir = Path(dataset_dir)
    folder = Path(str(folder).strip().strip('"').strip("'"))
    candidates = []

    if folder.is_absolute() and folder.is_dir():
        candidates.append(folder.resolve())
    elif folder.is_dir():
        candidates.append(folder.resolve())
    else:
        candidates.extend(
            path.resolve()
            for path in (
                dataset_dir / folder,
                init_root / folder,
            )
            if path.is_dir()
        )
        if len(folder.parts) == 1:
            candidates.extend(path.resolve() for path in sorted(init_root.glob(f"*/{folder.name}")) if path.is_dir())

    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        joined = "\n".join(str(path) for path in unique)
        raise ValueError(f"Audit folder name is ambiguous. Use Genus/Folder:\n{joined}")
    raise FileNotFoundError(f"Audit folder not found under init: {folder}")


def resolve_audit_input(dataset_dir, profile, subdir=None, folder=None):
    dataset_dir = Path(dataset_dir).resolve()
    init_root = (dataset_dir / (subdir or profile.root_init)).resolve()
    if folder is None:
        return init_root, Path(".")
    input_root = resolve_audit_folder(folder, dataset_dir, profile, init_root)
    rel_base = relative_to_init(input_root, dataset_dir, init_root, profile)
    return input_root, rel_base


def join_rel(base, child):
    if base == Path("."):
        return child
    if child == Path("."):
        return base
    return base / child
