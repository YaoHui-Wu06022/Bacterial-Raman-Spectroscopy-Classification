"""前缀池单谱审核入口。

single 不再按小文件夹独立离群；它复用 full_scan 的两阶段前缀池评分，
用于查看某个小文件夹所属前缀池的候选谱
"""

from __future__ import annotations

import argparse
from pathlib import Path

from raman.audit.common import PROJECT_ROOT, prefix_of, resolve_audit_folder, resolve_dataset, write_csv
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.full_scan import run_two_stage_scan, write_figures
from raman.audit.scoring import record_to_row
from raman.data.build import DEFAULT_PIPELINE_CONFIG, resolve_pipeline_config


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
    """按同属同前缀合并池执行局部审核。"""
    cfg = resolve_pipeline_config(pipeline_config or DEFAULT_PIPELINE_CONFIG)
    audit_cfg = audit_config or DEFAULT_AUDIT_CONFIG
    dataset_dir = Path(dataset_dir)
    init_root = dataset_dir / (subdir or profile.root_init)
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing audit input folder: {init_root}")

    target_scope = None
    if folder:
        target_folder = resolve_audit_folder(folder, dataset_dir, profile, init_root)
        rel = target_folder.relative_to(init_root)
        target_scope = f"{rel.parts[0]}/{prefix_of(rel.parts[1])}"

    if output_dir is None:
        output_dir = dataset_dir / "audit_single"
        if target_scope:
            output_dir = output_dir / target_scope
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, prefix_stats = run_two_stage_scan(profile, cfg, audit_cfg, init_root)
    if target_scope:
        records = [record for record in records if record.prefix_scope == target_scope]

    all_rows = [record_to_row(record) for record in records]
    candidate_rows = [
        row for row in all_rows if row["decision"] in {"remove_candidate", "review_candidate", "skip"}
    ]
    delete_rows = [row for row in candidate_rows if row["decision"] == "remove_candidate"]
    review_rows = [row for row in candidate_rows if row["decision"] == "review_candidate"]
    fieldnames = list(all_rows[0].keys()) if all_rows else None

    write_csv(output_dir / "all_spectra_scores.csv", all_rows, fieldnames)
    write_csv(output_dir / "candidates.csv", candidate_rows, fieldnames)
    write_csv(output_dir / "delete_candidates.csv", delete_rows, fieldnames)
    write_csv(output_dir / "review_candidates.csv", review_rows, fieldnames)
    fig_count = write_figures(output_dir, records, profile, cfg, prefix_stats, max_plots_per_group, audit_cfg)

    print("\nPrefix-pool single audit finished:")
    print(f"- Scope: {target_scope or 'all'}")
    print(f"- Output: {output_dir}")
    print(f"- Files={len(all_rows)}, Delete candidates={len(delete_rows)}, Review candidates={len(review_rows)}")
    print(f"- Figures={fig_count}")
    return {
        "input_root": str(init_root),
        "output_dir": str(output_dir),
        "scope": target_scope or "all",
        "files": len(all_rows),
        "delete_candidates": len(delete_rows),
        "review_candidates": len(review_rows),
        "figures": fig_count,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="按同属同前缀合并池审核单谱候选")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id，例如 细菌 / bacteria")
    parser.add_argument("--subdir", default="init", help="要审核的数据阶段，默认 init")
    parser.add_argument("--folder", default=None, help="指定小文件夹，例如 Staphylococcus/SA03 或 SA03")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--max-plots-per-group", type=int, default=20, help="最多输出多少张候选谱复核图")
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
