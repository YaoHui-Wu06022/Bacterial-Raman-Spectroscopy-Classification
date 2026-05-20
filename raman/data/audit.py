import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.data.archive import iter_arc_dirs
from raman.data.build import DEFAULT_PIPELINE_CONFIG, resolve_pipeline_config
from raman.data.offline import preprocess_single_spectrum
from raman.data.spectrum import read_arc_data, snv


@dataclass(frozen=True)
class SpectrumAuditConfig:
    """单谱质量审核参数，默认偏保守，先用于人工复核前的候选筛选"""

    min_samples: int = 5
    score_threshold: float = 3.5
    corr_threshold: float = 0.92
    point_z_threshold: float = 8.0
    max_bad_point_ratio: float = 0.03
    max_plots_per_group: int = 20


def _cosmic_ray_enabled(profile, cfg):
    return profile.profile_id in set(cfg.cosmic_ray_enabled_profile_ids)


def _safe_name(name):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("_") or "sample"


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float32)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(values))
    return center, max(scale, 1e-8)


def _corrcoef(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.std() <= 1e-8 or b.std() <= 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _add_bad_band_spans(ax, bad_bands):
    for band_min, band_max in bad_bands:
        ax.axvspan(band_min, band_max, color="gray", alpha=0.15)


def _plot_flagged_spectrum(path, payload, group_stats, out_path, cfg, audit_cfg):
    """输出异常单谱处理图，方便人工判断是否需要剔除原文件"""
    wn = payload["wn"]
    z = payload["z"]
    robust_z = payload["robust_z"]
    q10 = group_stats["q10"]
    q90 = group_stats["q90"]
    center = group_stats["center"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    axes[0].plot(payload["raw_wn"], payload["raw_sp"], linewidth=1.0)
    _add_bad_band_spans(axes[0], cfg.bad_bands)
    axes[0].set_title("Raw spectrum")
    axes[0].set_ylabel("Intensity")

    axes[1].fill_between(wn, q10, q90, color="C0", alpha=0.18, label="group q10-q90")
    axes[1].plot(wn, center, color="C0", linewidth=1.5, label="group median")
    axes[1].plot(wn, z, color="C3", linewidth=1.0, label="sample after preprocessing")
    _add_bad_band_spans(axes[1], cfg.bad_bands)
    axes[1].set_title(
        f"Shape compare | score={payload['score']:.2f}, "
        f"corr={payload['corr']:.3f}, bad_ratio={payload['bad_point_ratio']:.3f}"
    )
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    axes[2].plot(wn, robust_z, color="C4", linewidth=1.0)
    axes[2].axhline(audit_cfg.point_z_threshold, color="C3", linestyle="--", linewidth=1.0)
    axes[2].axhline(-audit_cfg.point_z_threshold, color="C3", linestyle="--", linewidth=1.0)
    axes[2].set_title("Robust residual z-score")
    axes[2].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[2].set_ylabel("z")

    fig.suptitle(path.name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _preprocess_group_samples(profile, cfg, root, arc_files):
    wn_ref = cfg.build_wn_ref()
    cosmic_ray_remove = _cosmic_ray_enabled(profile, cfg)
    payloads = []

    for filename in arc_files:
        path = root / filename
        wn, sp = read_arc_data(path)
        if wn.size == 0 or sp.size == 0:
            payloads.append({"path": path, "skip_reason": "read_failed"})
            continue

        wn_u, sp_u, cosmic_replaced = preprocess_single_spectrum(
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
            cosmic_ray_remove=cosmic_ray_remove,
            cosmic_ray_window_cm=cfg.cosmic_ray_narrow_window_cm,
            cosmic_ray_threshold=cfg.cosmic_ray_threshold,
            cosmic_ray_max_iter=cfg.cosmic_ray_max_iter,
            cosmic_ray_peak_prominence_z=cfg.cosmic_ray_peak_prominence_z,
            cosmic_ray_peak_width_max_cm=cfg.cosmic_ray_peak_width_max_cm,
            cosmic_ray_peak_ratio_z_per_cm=cfg.cosmic_ray_peak_ratio_z_per_cm,
            cosmic_ray_peak_pad_cm=cfg.cosmic_ray_peak_pad_cm,
            cosmic_ray_peak_rel_height=cfg.cosmic_ray_peak_rel_height,
        )
        if wn_u is None or sp_u is None:
            payloads.append({"path": path, "skip_reason": "preprocess_failed"})
            continue

        payloads.append(
            {
                "path": path,
                "raw_wn": wn,
                "raw_sp": sp,
                "wn": wn_u,
                "sp": sp_u,
                "z": snv(sp_u),
                "cosmic_replaced": int(cosmic_replaced),
                "skip_reason": "",
            }
        )

    return payloads


def _score_group(payloads, audit_cfg):
    valid_payloads = [item for item in payloads if not item["skip_reason"]]
    if len(valid_payloads) < audit_cfg.min_samples:
        return valid_payloads, None

    spectra = np.vstack([item["z"] for item in valid_payloads])
    center = np.median(spectra, axis=0)
    q10 = np.quantile(spectra, 0.10, axis=0)
    q90 = np.quantile(spectra, 0.90, axis=0)
    residual = spectra - center
    wave_mad = np.median(np.abs(residual), axis=0)
    wave_scale = 1.4826 * wave_mad
    fallback = float(np.median(wave_scale[wave_scale > 1e-8])) if np.any(wave_scale > 1e-8) else 1.0
    wave_scale = np.where(wave_scale > 1e-8, wave_scale, fallback)

    rmse_values = np.sqrt(np.mean(residual * residual, axis=1))
    rmse_center, rmse_scale = _robust_scale(rmse_values)

    for item, rmse, diff in zip(valid_payloads, rmse_values, residual):
        robust_z = diff / wave_scale
        abs_robust_z = np.abs(robust_z)
        item["rmse"] = float(rmse)
        item["score"] = float((rmse - rmse_center) / rmse_scale)
        item["corr"] = _corrcoef(item["z"], center)
        item["max_abs_z"] = float(np.max(abs_robust_z))
        item["p95_abs_z"] = float(np.quantile(abs_robust_z, 0.95))
        item["bad_point_ratio"] = float(np.mean(abs_robust_z > audit_cfg.point_z_threshold))
        item["robust_z"] = robust_z
        item["flagged"] = (
            item["score"] >= audit_cfg.score_threshold
            or item["corr"] <= audit_cfg.corr_threshold
            or item["bad_point_ratio"] >= audit_cfg.max_bad_point_ratio
        )

    return valid_payloads, {
        "center": center,
        "q10": q10,
        "q90": q90,
    }


def _resolve_audit_input(dataset_dir, profile, subdir, folder):
    dataset_dir = Path(dataset_dir)
    if folder is None:
        input_root = dataset_dir / (subdir or profile.root_init)
        rel_base = input_root.relative_to(dataset_dir)
        return input_root, rel_base

    folder = Path(folder)
    input_root = folder if folder.is_absolute() else dataset_dir / folder
    try:
        rel_base = input_root.relative_to(dataset_dir)
    except ValueError:
        rel_base = Path(input_root.name)
    return input_root, rel_base


def audit_dataset(
    profile,
    dataset_dir,
    subdir=None,
    folder=None,
    output_dir=None,
    pipeline_config=None,
    audit_config=None,
):
    """审核单谱离群样本，输出 summary.csv 和异常谱处理图"""
    cfg = resolve_pipeline_config(pipeline_config)
    audit_cfg = audit_config or SpectrumAuditConfig()
    input_root, rel_base = _resolve_audit_input(dataset_dir, profile, subdir, folder)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing audit input folder: {input_root}")

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / "audit" / rel_base
    figure_root = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_groups = 0
    total_files = 0
    total_flagged = 0

    for root, arc_files in iter_arc_dirs(input_root):
        total_groups += 1
        rel_dir = root.relative_to(input_root)
        rel_group = rel_base / rel_dir if rel_dir != Path(".") else rel_base
        label = rel_group.as_posix()
        print(f"\n=== Audit: {label} ===")

        payloads = _preprocess_group_samples(profile, cfg, root, arc_files)
        valid_payloads, group_stats = _score_group(payloads, audit_cfg)
        total_files += len(payloads)

        if group_stats is None:
            print(f"  Skip scoring: valid samples < {audit_cfg.min_samples}")
            for item in payloads:
                rows.append(
                    {
                        "group": label,
                        "file": item["path"].name,
                        "flagged": "",
                        "reason": item.get("skip_reason") or "too_few_group_samples",
                        "score": "",
                        "corr": "",
                        "rmse": "",
                        "max_abs_z": "",
                        "p95_abs_z": "",
                        "bad_point_ratio": "",
                        "cosmic_replaced": item.get("cosmic_replaced", ""),
                    }
                )
            continue

        flagged = [item for item in valid_payloads if item["flagged"]]
        total_flagged += len(flagged)
        print(f"  Files={len(payloads)}, valid={len(valid_payloads)}, flagged={len(flagged)}")

        flagged_for_plot = sorted(
            flagged,
            key=lambda item: (item["score"], item["bad_point_ratio"], item["max_abs_z"]),
            reverse=True,
        )[: audit_cfg.max_plots_per_group]
        for item in flagged_for_plot:
            out_name = f"{_safe_name(item['path'].stem)}.png"
            out_path = figure_root / rel_group / out_name
            _plot_flagged_spectrum(item["path"], item, group_stats, out_path, cfg, audit_cfg)

        for item in payloads:
            if item.get("skip_reason"):
                rows.append(
                    {
                        "group": label,
                        "file": item["path"].name,
                        "flagged": "",
                        "reason": item["skip_reason"],
                        "score": "",
                        "corr": "",
                        "rmse": "",
                        "max_abs_z": "",
                        "p95_abs_z": "",
                        "bad_point_ratio": "",
                        "cosmic_replaced": "",
                    }
                )
                continue
            reasons = []
            if item["score"] >= audit_cfg.score_threshold:
                reasons.append("shape_score")
            if item["corr"] <= audit_cfg.corr_threshold:
                reasons.append("low_corr")
            if item["bad_point_ratio"] >= audit_cfg.max_bad_point_ratio:
                reasons.append("bad_point_ratio")
            rows.append(
                {
                    "group": label,
                    "file": item["path"].name,
                    "flagged": int(item["flagged"]),
                    "reason": ";".join(reasons),
                    "score": f"{item['score']:.6f}",
                    "corr": f"{item['corr']:.6f}",
                    "rmse": f"{item['rmse']:.6f}",
                    "max_abs_z": f"{item['max_abs_z']:.6f}",
                    "p95_abs_z": f"{item['p95_abs_z']:.6f}",
                    "bad_point_ratio": f"{item['bad_point_ratio']:.6f}",
                    "cosmic_replaced": item["cosmic_replaced"],
                }
            )

    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "group",
        "file",
        "flagged",
        "reason",
        "score",
        "corr",
        "rmse",
        "max_abs_z",
        "p95_abs_z",
        "bad_point_ratio",
        "cosmic_replaced",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSingle-spectrum audit finished:")
    print(f"- Input: {input_root}")
    print(f"- Output: {output_dir}")
    print(f"- Groups={total_groups}, Files={total_files}, Flagged={total_flagged}")
    print(f"- Summary: {summary_path}")
    print(f"- Figures: {figure_root}")

    return {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "figure_root": str(figure_root),
        "groups": total_groups,
        "files": total_files,
        "flagged": total_flagged,
    }
