"""指定小文件夹的前缀池审核入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raman.audit.common import (
    PROJECT_ROOT,
    output_wn,
    plot_segments_without_bad_bands,
    prefix_of,
    resolve_audit_folder,
    resolve_dataset,
    write_csv,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.full_scan import run_two_stage_scan
from raman.audit.scoring import reason_labels, record_to_row
from raman.data.build import DEFAULT_PIPELINE_CONFIG


def _target_scope(target_folder: Path, init_root: Path):
    rel = target_folder.relative_to(init_root)
    return rel, f"{rel.parts[0]}/{prefix_of(rel.parts[1])}"


def _write_review_plot(out_dir, rel, target_records, prefix_stats, cfg):
    if not target_records:
        return None
    stats = prefix_stats.get(target_records[0].prefix_scope)
    if stats is None:
        return None

    png_path = out_dir / "review.png"
    wn = output_wn(cfg)
    plot_items = sorted(
        [record for record in target_records if record.z is not None],
        key=lambda item: (item.decision != "remove_candidate", item.decision != "review_candidate", -item.risk_score, item.file),
    )
    if not plot_items:
        return None
    cols = 3
    nrows = int(np.ceil(len(plot_items) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(15, max(3, nrows * 2.5)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {"remove_candidate": "#D62728", "review_candidate": "#FF7F0E", "keep": "#1F77B4"}

    for ax, record in zip(axes, plot_items):
        color = colors.get(record.decision, "#666666")
        plot_segments_without_bad_bands(ax, wn, stats["center"], cfg.bad_bands, color="#444444", lw=1.1, label="prefix mean")
        plot_segments_without_bad_bands(ax, wn, record.z, cfg.bad_bands, color=color, lw=0.9)
        ax.set_title(
            f"{record.file} | {record.decision}\n"
            f"score={record.prefix_outlier_score:.1f}, r={record.corr_species_mean:.2f}, "
            f"ro={record.nearest_prefix_other_corr:.2f}, bump={record.local_bump_max_z:.1f}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)

    for ax in axes[len(plot_items) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{rel.as_posix()} prefix-pool review", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def audit_folder(dataset, folder, dataset_root=None, out_dir=None, no_plot=False, audit_config=None):
    """按同属同前缀合并池审核一个小文件夹。"""
    audit_cfg = audit_config or DEFAULT_AUDIT_CONFIG
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    cfg = DEFAULT_PIPELINE_CONFIG
    init_root = Path(dataset_root).resolve() if dataset_root else (dataset_dir / profile.root_init).resolve()
    output_root = Path(out_dir).resolve() if out_dir else dataset_dir / "audit_folder"
    target_folder = resolve_audit_folder(folder, dataset_dir, profile, init_root)
    rel, scope = _target_scope(target_folder, init_root)

    records, prefix_stats = run_two_stage_scan(profile, cfg, audit_cfg, init_root)
    target_records = [record for record in records if record.group == rel.as_posix()]
    if not target_records:
        raise RuntimeError(f"No valid spectra were found for {rel.as_posix()}")

    out_path = output_root / rel
    out_path.mkdir(parents=True, exist_ok=True)
    rows = [record_to_row(record) for record in target_records]
    fieldnames = list(rows[0].keys()) if rows else None
    candidate_rows = [row for row in rows if row["decision"] in {"remove_candidate", "review_candidate", "skip"}]
    delete_rows = [row for row in rows if row["decision"] == "remove_candidate"]
    review_rows = [row for row in rows if row["decision"] == "review_candidate"]
    write_csv(out_path / "scores.csv", rows, fieldnames)
    write_csv(out_path / "candidates.csv", candidate_rows, fieldnames)
    write_csv(out_path / "delete_candidates.csv", delete_rows, fieldnames)
    write_csv(out_path / "review_candidates.csv", review_rows, fieldnames)

    prefix_records = [record for record in records if record.prefix_scope == scope and record.z is not None]
    ref_dirs = sorted({record.folder for record in prefix_records if record.group != rel.as_posix()})
    payload = {
        "folder": rel.as_posix(),
        "prefix_scope": scope,
        "reference_dirs": ref_dirs,
        "files": len(rows),
        "delete_candidates": len(delete_rows),
        "review_candidates": len(review_rows),
        "reason_counts": {},
    }
    for record in target_records:
        for label in reason_labels(record.reasons):
            payload["reason_counts"][label] = payload["reason_counts"].get(label, 0) + 1

    png_path = None if no_plot else _write_review_plot(out_path, rel, target_records, prefix_stats, cfg)
    payload["png"] = str(png_path) if png_path else ""
    (out_path / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Folder: {rel.as_posix()}")
    print(f"Prefix scope: {scope}")
    print(f"Reference dirs: {', '.join(ref_dirs)}")
    print(f"Keep: {sum(row['decision'] == 'keep' for row in rows)}")
    print(f"Delete candidates: {len(delete_rows)}")
    print(f"Review candidates: {len(review_rows)}")
    print(f"CSV: {out_path / 'scores.csv'}")
    print(f"JSON: {out_path / 'summary.json'}")
    if png_path is not None:
        print(f"PNG: {png_path}")
    return {"rows": rows, "summary": payload, "csv": str(out_path / "scores.csv"), "png": str(png_path) if png_path else ""}


def build_parser():
    parser = argparse.ArgumentParser(description="审核小文件夹相对同属同前缀合并池的异常情况")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("--folder", required=True, help="指定小文件夹，例如 Acinetobacter/AB01 或 AB01")
    parser.add_argument("--dataset-root", default=None, help="覆盖 init 根目录")
    parser.add_argument("--out-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成复核图")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    audit_folder(args.dataset, args.folder, dataset_root=args.dataset_root, out_dir=args.out_dir, no_plot=args.no_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
