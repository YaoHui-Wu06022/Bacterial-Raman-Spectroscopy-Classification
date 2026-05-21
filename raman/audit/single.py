"""组内单谱离群审核入口"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.audit.common import (
    PROJECT_ROOT,
    fill_between_segments_without_bad_bands,
    join_rel,
    plot_segments_without_bad_bands,
    preprocess_spectrum_for_audit,
    resolve_audit_input,
    resolve_dataset,
    safe_name,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.scoring import score_group_payloads
from raman.data.archive import iter_arc_dirs
from raman.data.build import DEFAULT_PIPELINE_CONFIG, resolve_pipeline_config


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


def _plot_flagged_spectrum(path, payload, group_stats, out_path, cfg, audit_cfg):
    """输出异常单谱复核图"""
    wn = payload["wn"]
    z = payload["z"]
    robust_z = payload["robust_z"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    plot_segments_without_bad_bands(
        axes[0],
        payload["raw_wn"],
        payload["raw_sp"],
        cfg.bad_bands,
        color="0.55",
        linewidth=1.0,
        label="raw",
    )
    axes[0].set_title("Raw spectrum")
    axes[0].set_ylabel("Intensity")

    fill_between_segments_without_bad_bands(
        axes[1],
        wn,
        group_stats["q10"],
        group_stats["q90"],
        cfg.bad_bands,
        color="C0",
        alpha=0.18,
        label="group q10-q90",
    )
    plot_segments_without_bad_bands(axes[1], wn, group_stats["center"], cfg.bad_bands, color="C0", linewidth=1.5, label="group median")
    plot_segments_without_bad_bands(axes[1], wn, z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample after preprocessing")
    axes[1].set_title(
        f"Shape compare | score={payload['score']:.2f}, "
        f"corr={payload['corr']:.3f}, bad_ratio={payload['bad_point_ratio']:.3f}"
    )
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    plot_segments_without_bad_bands(axes[2], wn, robust_z, cfg.bad_bands, color="C4", linewidth=1.0)
    axes[2].axhline(audit_cfg.group_point_z_threshold, color="C3", linestyle="--", linewidth=1.0)
    axes[2].axhline(-audit_cfg.group_point_z_threshold, color="C3", linestyle="--", linewidth=1.0)
    axes[2].set_title("Robust residual z-score")
    axes[2].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[2].set_ylabel("z")

    fig.suptitle(path.name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _flag_reasons(item, audit_cfg):
    reasons = []
    if item["score"] >= audit_cfg.group_score_threshold:
        reasons.append("shape_score")
    if item["corr"] <= audit_cfg.group_corr_threshold:
        reasons.append("low_corr")
    if item["bad_point_ratio"] >= audit_cfg.group_bad_ratio_threshold:
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


def _skip_summary_row(label, item, reason):
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


def audit_dataset(
    profile,
    dataset_dir,
    subdir=None,
    folder=None,
    output_dir=None,
    pipeline_config=None,
    audit_config=None,
    max_plots_per_group=20,
):
    """执行组内单谱离群审核"""
    cfg = resolve_pipeline_config(pipeline_config or DEFAULT_PIPELINE_CONFIG)
    audit_cfg = audit_config or DEFAULT_AUDIT_CONFIG
    input_root, rel_base = resolve_audit_input(dataset_dir, profile, subdir, folder)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing audit input folder: {input_root}")

    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        output_dir = dataset_dir / "audit_single"
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
    wn_ref = cfg.build_wn_ref()

    for root, arc_files in iter_arc_dirs(input_root):
        total_groups += 1
        rel_dir = root.relative_to(input_root)
        rel_group = join_rel(rel_base, rel_dir)
        label = rel_group.as_posix()
        print(f"\n=== Audit: {label} ===")

        payloads = [
            preprocess_spectrum_for_audit(root / filename, profile, cfg, wn_ref=wn_ref, include_raw=True)
            for filename in arc_files
        ]
        valid_payloads, group_stats = score_group_payloads(payloads, audit_cfg)
        total_files += len(payloads)

        if group_stats is None:
            print(f"  Skip scoring: valid samples < {audit_cfg.min_group_samples}")
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
        )[:max_plots_per_group]
        for item in flagged_for_plot:
            out_name = f"{safe_name(item['path'].stem)}.png"
            figure_group = rel_dir if rel_dir != Path(".") else Path()
            _plot_flagged_spectrum(item["path"], item, group_stats, figure_root / figure_group / out_name, cfg, audit_cfg)

        for item in payloads:
            if item.get("skip_reason"):
                rows.append(_skip_summary_row(label, item, item["skip_reason"]))
            else:
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


def build_parser():
    parser = argparse.ArgumentParser(description="审核同一文件夹内的单谱离群")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id，例如 细菌 / bacteria")
    parser.add_argument("--subdir", default="init", help="要审核的数据阶段，默认 init")
    parser.add_argument("--folder", default=None, help="指定小文件夹，例如 Staphylococcus/SA03 或 SA03")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--max-plots-per-group", type=int, default=20, help="每个小文件夹最多输出多少张复核图")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    profile, dataset_dir = resolve_dataset(args.dataset, PROJECT_ROOT)
    audit_dataset(
        profile,
        dataset_dir,
        subdir=args.subdir,
        folder=args.folder,
        output_dir=args.output_dir,
        max_plots_per_group=args.max_plots_per_group,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
