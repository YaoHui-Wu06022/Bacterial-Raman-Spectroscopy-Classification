"""全库两阶段强异常谱清理入口"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.audit.common import (
    PROJECT_ROOT,
    cosmic_clean_for_plot,
    fill_between_segments_without_bad_bands,
    load_audit_records,
    output_wn,
    plot_segments_without_bad_bands,
    resolve_dataset,
    select_limited,
    write_csv,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.scoring import (
    SpectrumRecord,
    classify_prefix_cleanup,
    classify_species_prefilter,
    reason_labels,
    record_to_row,
    score_prefix_pools,
)
from raman.data.build import DEFAULT_PIPELINE_CONFIG
from raman.data.spectrum import read_arc_data


def plot_spectrum_candidate(record, out_path, profile, cfg, prefix_stats, audit_cfg):
    """输出单谱复核图。"""
    stats = prefix_stats.get(record.prefix_scope)
    if stats is None or record.z is None:
        return

    raw_wn, raw_sp = read_arc_data(record.path)
    raw_cosmic = cosmic_clean_for_plot(raw_wn, raw_sp, profile, cfg)
    wn = output_wn(cfg)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
    plot_segments_without_bad_bands(axes[0], raw_wn, raw_sp, cfg.bad_bands, color="0.60", linewidth=0.9, label="raw")
    plot_segments_without_bad_bands(axes[0], raw_wn, raw_cosmic, cfg.bad_bands, color="C0", linewidth=0.9, label="cosmic cleaned")
    axes[0].set_title("Raw / cosmic cleaned")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="best")

    fill_between_segments_without_bad_bands(axes[1], wn, stats["q10"], stats["q90"], cfg.bad_bands, color="C0", alpha=0.16, label="prefix q10-q90")
    plot_segments_without_bad_bands(axes[1], wn, stats["center"], cfg.bad_bands, color="C0", linewidth=1.4, label="prefix mean")
    plot_segments_without_bad_bands(axes[1], wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample SNV")
    axes[1].set_title("After preprocessing vs merged prefix mean")
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    prefix_z = (record.z - stats["center"]) / stats["wave_scale"]
    plot_segments_without_bad_bands(axes[2], wn, prefix_z, cfg.bad_bands, color="C4", linewidth=0.9, label="prefix residual z")
    axes[2].axhline(audit_cfg.local_bump_z_threshold, color="C3", linestyle="--", linewidth=0.8, label="local bump z")
    axes[2].axhline(-audit_cfg.group_point_z_threshold, color="0.5", linestyle=":", linewidth=0.8)
    for pos in record.step_positions:
        axes[2].axvline(pos, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_title("Prefix residual z-score and detected steps")
    axes[2].set_ylabel("z")
    axes[2].legend(loc="best")

    axes[3].axis("off")
    text = (
        f"Decision: {record.decision}\n"
        f"Stage: {record.clean_stage}\n"
        f"Reasons: {'; '.join(record.reasons)}\n"
        f"Labels: {'; '.join(reason_labels(record.reasons))}\n"
        f"prefix={record.prefix_scope}, score={record.prefix_outlier_score:.3f}\n"
        f"corr_species_mean={record.corr_species_mean:.3f}, nearest_prefix_corr={record.nearest_prefix_corr:.3f}\n"
        f"nearest_other_folder_corr={record.nearest_prefix_other_corr:.3f}, other_ref_pool={record.other_ref_pool_size}\n"
        f"local_bump_regions={record.species_bump_regions}, bump_z={record.local_bump_max_z:.3f}, area={record.local_bump_area:.3f}\n"
        f"variance_share={record.prefix_variance_share:.5f}, variance_ratio={record.prefix_variance_ratio:.3f}\n"
        f"bad_ratio_z6={record.bad_ratio_z6:.3f} (diagnostic only)\n"
        f"step_count={record.step_count}, bad_band_edge_step={record.bad_band_edge_step_count}\n"
        f"roughness_z={record.roughness_z:.3f}, cosmic_total={record.cosmic_total}"
    )
    axes[3].text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(record.rel_path)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_figures(out_dir, records, profile, cfg, prefix_stats, max_spectrum_figures, audit_cfg):
    """按数量上限输出候选谱复核图。"""
    candidates = [record for record in records if record.decision in {"remove_candidate", "review_candidate"} and record.z is not None]
    candidates = sorted(candidates, key=lambda item: (item.decision != "remove_candidate", -item.risk_score, item.rel_path))
    spectrum_to_plot = select_limited(candidates, max_spectrum_figures)
    for record in spectrum_to_plot:
        out_path = out_dir / "figures" / "spectra" / record.genus / record.folder / f"{Path(record.file).stem}.png"
        plot_spectrum_candidate(record, out_path, profile, cfg, prefix_stats, audit_cfg)
    return len(spectrum_to_plot)


def write_summary(out_dir, dataset_name, records, cfg, dataset_dir, init_root, fig_count, move_strong):
    """写中文报告和机器可读摘要。"""
    valid = [record for record in records if record.z is not None]
    skipped = [record for record in records if record.z is None]
    remove_records = [record for record in valid if record.decision == "remove_candidate"]
    review_records = [record for record in valid if record.decision == "review_candidate"]
    stage_counts = {}
    for record in valid:
        stage_counts[record.clean_stage or "keep"] = stage_counts.get(record.clean_stage or "keep", 0) + 1

    lines = [
        f"# dataset/{dataset_name} 两阶段强异常清理报告",
        "",
        "## 总体",
        "",
        f"- 数据目录：`{init_root}`",
        f"- 输出目录：`{out_dir}`",
        f"- 总光谱数：{len(records)}",
        f"- 有效光谱数：{len(valid)}",
        f"- 跳过光谱数：{len(skipped)}",
        f"- 强异常移除候选：{len(remove_records)}",
        f"- 仅复核候选：{len(review_records)}",
        f"- 已输出候选谱图：{fig_count}",
        f"- 自动移动强异常：{'是' if move_strong else '否'}",
        "- 第一阶段使用同属同前缀合并均值谱筛掉局部异常凸起、阶梯样异常和完全不贴合的无效谱",
        "- 第二阶段排除第一阶段强异常后，按方差贡献、相关性和排除同小文件夹后的最近邻相关性做强离群清洗",
        "- bad_ratio_z6 / bad_ratio_z8 仅作为诊断字段保留，不再作为删除必要条件",
        "- review_candidate 只写报告，不会自动移动",
        "",
        "## 阶段统计",
        "",
    ]
    for stage, count in sorted(stage_counts.items()):
        lines.append(f"- {stage}: {count}")

    lines.extend(["", "## 强异常移除候选", ""])
    if remove_records:
        for record in sorted(remove_records, key=lambda item: (-item.risk_score, item.rel_path))[:120]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(reason_labels(record.reasons))}（{'; '.join(record.reasons)}）")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 仅复核候选", ""])
    if review_records:
        for record in sorted(review_records, key=lambda item: (-item.risk_score, item.rel_path))[:120]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(reason_labels(record.reasons))}（{'; '.join(record.reasons)}）")
    else:
        lines.append("- 暂无")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "dataset_dir": str(dataset_dir),
        "init_root": str(init_root),
        "output_dir": str(out_dir),
        "pipeline_config": asdict(cfg),
        "audit_config": asdict(DEFAULT_AUDIT_CONFIG),
        "total_spectra": len(records),
        "valid_spectra": len(valid),
        "skipped_spectra": len(skipped),
        "remove_candidates": len(remove_records),
        "review_candidates": len(review_records),
        "stage_counts": stage_counts,
        "figures": {"spectra": fig_count},
        "move_strong": move_strong,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_two_stage_scan(profile, cfg, audit_cfg, init_root):
    records = load_audit_records(profile, cfg, init_root, SpectrumRecord)
    first_stats = score_prefix_pools(records, cfg, audit_cfg)
    classify_species_prefilter(records, audit_cfg)
    first_remove_ids = {id(record) for record in records if record.decision == "remove_candidate"}
    final_stats = score_prefix_pools(records, cfg, audit_cfg, exclude_ids=first_remove_ids)
    classify_prefix_cleanup(records, audit_cfg)
    return records, final_stats or first_stats


def full_scan(
    dataset,
    output_root=None,
    timestamp=None,
    max_remove_candidates=0,
    max_folder_candidates=0,
    max_spectrum_figures=0,
    max_folder_figures=0,
    move_strong=False,
):
    """执行全库两阶段强异常清理。"""
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    cfg = DEFAULT_PIPELINE_CONFIG
    audit_cfg = DEFAULT_AUDIT_CONFIG
    init_root = dataset_dir / profile.root_init
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init folder: {init_root}")

    if timestamp is None:
        mode_name = "full_scan_move_strong" if move_strong else "full_scan_dry_run"
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode_name}"
    output_root = Path(output_root) if output_root else dataset_dir / "audit_full_scan"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    out_dir = output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_dir}")
    print(f"Input: {init_root}")
    print(f"Output: {out_dir}")

    records, prefix_stats = run_two_stage_scan(profile, cfg, audit_cfg, init_root)

    candidate_rows = [
        record_to_row(record)
        for record in records
        if record.decision in {"remove_candidate", "review_candidate", "skip"}
    ]
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["decision"] != "remove_candidate", row["prefix_scope"], row["group"], row["file"]))
    delete_rows = [row for row in candidate_rows if row["decision"] == "remove_candidate"]
    review_rows = [row for row in candidate_rows if row["decision"] == "review_candidate"]
    all_rows = [record_to_row(record) for record in records]
    spectrum_fieldnames = list(all_rows[0].keys()) if all_rows else None

    delete_csv = out_dir / "delete_candidates.csv"
    write_csv(out_dir / "spectrum_candidates.csv", candidate_rows, spectrum_fieldnames)
    write_csv(delete_csv, delete_rows, spectrum_fieldnames)
    write_csv(out_dir / "review_candidates.csv", review_rows, spectrum_fieldnames)
    write_csv(out_dir / "all_spectra_scores.csv", all_rows, spectrum_fieldnames)
    (out_dir / "delete_manifest.txt").write_text("".join(f"{row['rel_path']}\n" for row in delete_rows), encoding="utf-8")

    fig_count = write_figures(out_dir, records, profile, cfg, prefix_stats, max_spectrum_figures, audit_cfg)
    write_summary(out_dir, profile.dataset_name, records, cfg, dataset_dir, init_root, fig_count, move_strong)

    if move_strong and delete_rows:
        from raman.audit.move import move_items

        log_path = out_dir / "move_strong_log.txt"
        with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file):
            move_items(dataset, from_list=str(delete_csv), dry_run=False)
        print(f"- Strong candidates moved: {len(delete_rows)}")
        print(f"- Move log: {log_path}")

    print("\nFull scan finished:")
    print(f"- Summary: {out_dir / 'summary.md'}")
    print(f"- Delete candidates: {delete_csv}")
    print(f"- Review candidates: {out_dir / 'review_candidates.csv'}")
    print(f"- All scores: {out_dir / 'all_spectra_scores.csv'}")
    print(f"- Figures: {out_dir / 'figures'}")
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description="全库两阶段强异常谱清理")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("--output-root", default=None, help="覆盖输出根目录")
    parser.add_argument("--timestamp", default=None, help="覆盖时间戳目录名")
    parser.add_argument("--max-remove-candidates", type=int, default=0, help="兼容旧参数；新流程不按数量截断强异常")
    parser.add_argument("--max-folder-candidates", type=int, default=0, help="兼容旧参数；新流程不输出文件夹候选")
    parser.add_argument("--max-spectrum-figures", type=int, default=0, help="最多输出多少张单谱复核图，0 表示全部，负数表示不输出")
    parser.add_argument("--max-folder-figures", type=int, default=0, help="兼容旧参数；新流程不输出文件夹候选图")
    parser.add_argument("--move-strong", action="store_true", help="将强异常 delete_candidates 自动移动到 delete；review 不会移动")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    full_scan(
        args.dataset,
        output_root=args.output_root,
        timestamp=args.timestamp,
        max_remove_candidates=args.max_remove_candidates,
        max_folder_candidates=args.max_folder_candidates,
        max_spectrum_figures=args.max_spectrum_figures,
        max_folder_figures=args.max_folder_figures,
        move_strong=args.move_strong,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
