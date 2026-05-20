"""Audit single-spectrum outliers inside one init folder or a dataset stage.

Usage examples:
    python dataset/audit_single_spectra.py 细菌 --subdir init
    python dataset/audit_single_spectra.py 细菌 --folder Acinetobacter/AB01
    python dataset/audit_single_spectra.py 细菌 --folder SA03 --max-plots-per-group 30

The script does not move or delete files. It writes summary.csv and review
figures under dataset/<dataset>/audit/.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman.data.archive import iter_arc_dirs
from raman.data.build import DEFAULT_PIPELINE_CONFIG, _cosmic_ray_kwargs, resolve_pipeline_config
from raman.data.offline import preprocess_single_spectrum
from raman.data.profiles import get_dataset_dir, get_profile
from raman.data.spectrum import read_arc_data, snv


SUMMARY_FIELDNAMES = [
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


@dataclass(frozen=True)
class SpectrumAuditConfig:
    """单谱质量审核参数，默认偏保守，先用于人工复核前的候选筛选"""

    min_samples: int = 5
    score_threshold: float = 3.5
    corr_threshold: float = 0.92
    point_z_threshold: float = 8.0
    max_bad_point_ratio: float = 0.03
    max_plots_per_group: int = 20


def _safe_name(name):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("_") or "sample"


def robust_scale(values):
    """返回一组数值的 median 和 MAD 尺度，用于把 RMSE 转成稳健 z-score"""
    values = np.asarray(values, dtype=np.float32)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(values))
    return center, max(scale, 1e-8)


def spectral_corr(a, b):
    """计算两条光谱的相关系数；近似常数谱直接返回 0，避免 NaN 干扰评分"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.std() <= 1e-8 or b.std() <= 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def robust_wave_stats(spectra, min_scale=0.05, floor_fraction=0.25):
    """逐波数估计组内中位谱和鲁棒尺度，供组内/跨组审核共同使用"""
    spectra = np.asarray(spectra, dtype=np.float32)
    center = np.median(spectra, axis=0)
    mad = np.median(np.abs(spectra - center), axis=0)
    scale = 1.4826 * mad
    if np.any(scale > 1e-8):
        floor = max(float(np.median(scale[scale > 1e-8])) * float(floor_fraction), float(min_scale))
    else:
        floor = float(min_scale)
    return center, np.maximum(scale, floor)


def preprocess_spectrum_for_audit(path, profile, cfg, wn_ref=None, include_raw=False):
    """按当前离线流程处理单条原始谱，返回审核需要的 SNV 光谱和宇宙射线统计"""
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
            "cosmic_replaced": int(cosmic_replaced),
            "cosmic_stats": cosmic_replaced,
        }
    )
    return payload


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
    """预处理同一文件夹内所有单谱；保留 raw 数据用于后续异常图展示"""
    wn_ref = cfg.build_wn_ref()
    return [
        preprocess_spectrum_for_audit(root / filename, profile, cfg, wn_ref=wn_ref, include_raw=True)
        for filename in arc_files
    ]


def _score_group(payloads, audit_cfg):
    """在同一小文件夹内部找离群谱，不做跨文件夹类别判断"""
    valid_payloads = [item for item in payloads if not item["skip_reason"]]
    if len(valid_payloads) < audit_cfg.min_samples:
        return valid_payloads, None

    spectra = np.vstack([item["z"] for item in valid_payloads])
    center, wave_scale = robust_wave_stats(spectra, min_scale=1e-8, floor_fraction=1.0)
    q10 = np.quantile(spectra, 0.10, axis=0)
    q90 = np.quantile(spectra, 0.90, axis=0)
    residual = spectra - center

    rmse_values = np.sqrt(np.mean(residual * residual, axis=1))
    rmse_center, rmse_scale = robust_scale(rmse_values)

    for item, rmse, diff in zip(valid_payloads, rmse_values, residual):
        robust_z = diff / wave_scale
        abs_robust_z = np.abs(robust_z)
        item["rmse"] = float(rmse)
        item["score"] = float((rmse - rmse_center) / rmse_scale)
        item["corr"] = spectral_corr(item["z"], center)
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


def _relative_to_init(path, dataset_dir, init_root, profile):
    """返回去掉 init 前缀后的数据相对路径，用于输出目录和报告标签"""
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


def _resolve_audit_folder(folder, dataset_dir, profile, init_root):
    """支持 init/Genus/Folder、Genus/Folder、末级 Folder 名和绝对路径"""
    dataset_dir = Path(dataset_dir)
    folder = Path(folder)
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
            candidates.extend(
                path.resolve()
                for path in sorted(init_root.glob(f"*/{folder.name}"))
                if path.is_dir()
            )

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


def _resolve_audit_input(dataset_dir, profile, subdir, folder):
    dataset_dir = Path(dataset_dir).resolve()
    init_root = (dataset_dir / (subdir or profile.root_init)).resolve()
    if folder is None:
        return init_root, Path(".")

    input_root = _resolve_audit_folder(folder, dataset_dir, profile, init_root)
    rel_base = _relative_to_init(input_root, dataset_dir, init_root, profile)
    return input_root, rel_base


def _skip_summary_row(label, item, reason):
    """生成无法评分样本的 summary 行"""
    return {
        "group": label,
        "file": item["path"].name,
        "flagged": "",
        "reason": reason,
        "score": "",
        "corr": "",
        "rmse": "",
        "max_abs_z": "",
        "p95_abs_z": "",
        "bad_point_ratio": "",
        "cosmic_replaced": item.get("cosmic_replaced", ""),
    }


def _flag_reasons(item, audit_cfg):
    """把触发的阈值条件记录成可读原因，方便回看 summary.csv"""
    reasons = []
    if item["score"] >= audit_cfg.score_threshold:
        reasons.append("shape_score")
    if item["corr"] <= audit_cfg.corr_threshold:
        reasons.append("low_corr")
    if item["bad_point_ratio"] >= audit_cfg.max_bad_point_ratio:
        reasons.append("bad_point_ratio")
    return reasons


def _summary_row(label, item, audit_cfg):
    reasons = _flag_reasons(item, audit_cfg)
    return {
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


def _join_rel(base, child):
    if base == Path("."):
        return child
    if child == Path("."):
        return base
    return base / child


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
    if output_dir is None:
        output_dir = dataset_dir / "audit"
        if folder is not None and rel_base != Path("."):
            output_dir = output_dir / rel_base
    else:
        output_dir = Path(output_dir)
    figure_root = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_groups = 0
    total_files = 0
    total_flagged = 0

    for root, arc_files in iter_arc_dirs(input_root):
        total_groups += 1
        rel_dir = root.relative_to(input_root)
        rel_group = _join_rel(rel_base, rel_dir)
        label = rel_group.as_posix()
        print(f"\n=== Audit: {label} ===")

        payloads = _preprocess_group_samples(profile, cfg, root, arc_files)
        valid_payloads, group_stats = _score_group(payloads, audit_cfg)
        total_files += len(payloads)

        if group_stats is None:
            print(f"  Skip scoring: valid samples < {audit_cfg.min_samples}")
            for item in payloads:
                rows.append(_skip_summary_row(label, item, item.get("skip_reason") or "too_few_group_samples"))
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
            figure_group = rel_dir if rel_dir != Path(".") else Path()
            out_path = figure_root / figure_group / out_name
            _plot_flagged_spectrum(item["path"], item, group_stats, out_path, cfg, audit_cfg)

        for item in payloads:
            if item.get("skip_reason"):
                rows.append(_skip_summary_row(label, item, item["skip_reason"]))
                continue
            rows.append(_summary_row(label, item, audit_cfg))

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDNAMES)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Audit single-spectrum outliers inside dataset init folders.")
    parser.add_argument("dataset", nargs="?", default="细菌", help="Dataset name or profile id, e.g. 细菌 / bacteria.")
    parser.add_argument("--subdir", default="init", help="Dataset stage to audit. Default: init.")
    parser.add_argument("--folder", default=None, help="Folder to audit, e.g. Staphylococcus/SA03 or SA03.")
    parser.add_argument("--output-dir", default=None, help="Override audit output dir.")
    parser.add_argument("--score-threshold", type=float, default=3.5)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--point-z-threshold", type=float, default=8.0)
    parser.add_argument("--max-bad-point-ratio", type=float, default=0.03)
    parser.add_argument("--max-plots-per-group", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    profile = get_profile(args.dataset)
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    audit_cfg = SpectrumAuditConfig(
        score_threshold=args.score_threshold,
        corr_threshold=args.corr_threshold,
        point_z_threshold=args.point_z_threshold,
        max_bad_point_ratio=args.max_bad_point_ratio,
        max_plots_per_group=args.max_plots_per_group,
    )
    audit_dataset(
        profile,
        dataset_dir,
        subdir=args.subdir,
        folder=args.folder,
        output_dir=args.output_dir,
        audit_config=audit_cfg,
    )


if __name__ == "__main__":
    main()
