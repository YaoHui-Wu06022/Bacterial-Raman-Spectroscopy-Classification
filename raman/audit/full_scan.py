"""全库分阶段清洗入口。"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.audit.common import (
    PROJECT_ROOT,
    cosmic_clean_for_plot,
    fill_between_segments_without_bad_bands,
    load_audit_records,
    moving_average,
    output_wn,
    plot_segments_without_bad_bands,
    resolve_dataset,
    write_csv,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.scoring import (
    SpectrumRecord,
    STAGE_DELETE_CATEGORY,
    reason_labels,
    record_to_row,
    score_stage,
    stage_title,
    validate_stage,
)
from raman.data.build import DEFAULT_PIPELINE_CONFIG
from raman.data.spectrum import read_arc_data


def _class_folder_review_targets(records, audit_cfg):
    """挑出需要生成文件夹集中度复核图的记录。"""
    targets = []
    seen = set()
    for record in records:
        if record.stage != "class-similarity" or record.z is None:
            continue
        if record.folder_candidate_count <= 0:
            continue
        if (
            record.folder_candidate_count <= audit_cfg.class_folder_candidate_max_count
            and record.folder_candidate_fraction <= audit_cfg.class_folder_candidate_max_fraction
        ):
            continue
        key = (record.genus, record.folder, record.prefix_scope)
        if key in seen:
            continue
        seen.add(key)
        targets.append(record)
    return sorted(targets, key=lambda item: (-item.folder_candidate_count, -item.folder_candidate_fraction, item.rel_path))


def plot_class_folder_review(record, out_path, profile, cfg, prefix_stats, audit_cfg):
    """绘制小文件夹均值和其它同前缀均值对比图。"""
    stats = prefix_stats.get(record.ref_pool_scope or record.prefix_scope)
    if stats is None:
        return
    scope_records = [item for item in stats["records"] if item.z is not None]
    folder_records = [item for item in scope_records if item.folder == record.folder]
    other_records = [item for item in scope_records if item.folder != record.folder]
    if not folder_records or not other_records:
        return

    folder_mean = np.mean(np.stack([item.z for item in folder_records]).astype(np.float32), axis=0)
    other_mean = np.mean(np.stack([item.z for item in other_records]).astype(np.float32), axis=0)
    folder_count = len(folder_records)
    other_count = len(other_records)
    wn = output_wn(cfg)[: folder_mean.size]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plot_segments_without_bad_bands(axes[0], wn, other_mean, cfg.bad_bands, color="0.35", linewidth=1.0, label="same prefix other folders mean")
    plot_segments_without_bad_bands(axes[0], wn, folder_mean, cfg.bad_bands, color="C3", linewidth=1.1, label="this folder mean")
    axes[0].set_title("Folder concentration review")
    axes[0].set_ylabel("SNV intensity")
    axes[0].legend(loc="best")

    diff = folder_mean - other_mean
    plot_segments_without_bad_bands(axes[1], wn, diff, cfg.bad_bands, color="C4", linewidth=1.0, label="folder minus others")
    axes[1].axhline(0.0, color="0.3", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("Difference")
    axes[1].set_xlabel("Wavenumber")
    axes[1].legend(loc="best")

    fig.suptitle(
        f"{record.genus}/{record.folder} | candidates={record.folder_candidate_count}/{folder_count} "
        f"({record.folder_candidate_fraction:.3f}) | other folders={other_count}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _shade_ranges(ax, wn, ranges, color="darkorange", alpha=0.16):
    """在图上标出若干索引区间。"""
    for start, end in ranges:
        ax.axvspan(float(wn[start]), float(wn[end - 1]), color=color, alpha=alpha)


def _plot_raw_panel(ax, record, profile, cfg):
    """绘制原始谱和宇宙射线清理后谱。"""
    raw_wn, raw_sp = read_arc_data(record.path)
    raw_cosmic = cosmic_clean_for_plot(raw_wn, raw_sp, profile, cfg)
    plot_segments_without_bad_bands(ax, raw_wn, raw_sp, cfg.bad_bands, color="0.60", linewidth=0.9, label="raw")
    plot_segments_without_bad_bands(ax, raw_wn, raw_cosmic, cfg.bad_bands, color="C0", linewidth=0.9, label="cosmic cleaned")
    ax.set_title("Raw / cosmic cleaned")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best")


def _plot_anomalous_panels(axes, record, wn, cfg, audit_cfg):
    """绘制第二阶段宽平台/阶梯异常视图。"""
    plot_segments_without_bad_bands(axes[1], wn, record.sp, cfg.bad_bands, color="C2", linewidth=0.9, label="baseline corrected")
    if record.wide_smooth is not None:
        plot_segments_without_bad_bands(axes[1], wn, record.wide_smooth, cfg.bad_bands, color="C1", linewidth=1.1, label="short smooth")
    if record.wide_floor is not None:
        plot_segments_without_bad_bands(axes[1], wn, record.wide_floor, cfg.bad_bands, color="0.35", linewidth=1.0, label="local floor")
    _shade_ranges(axes[1], wn, record.wide_bump_ranges)
    axes[1].set_title("Baseline corrected / wide platform detection")
    axes[1].set_ylabel("Corrected intensity")
    axes[1].legend(loc="best")

    if record.wide_z is not None:
        plot_segments_without_bad_bands(axes[2], wn, record.wide_z, cfg.bad_bands, color="C4", linewidth=1.0, label="wide residual z")
        axes[2].axhline(audit_cfg.anomalous_wide_z_min, color="darkorange", linestyle="--", linewidth=1.0, label="wide z threshold")
    _shade_ranges(axes[2], wn, record.wide_bump_ranges)
    axes[2].set_title("Wide rising platform / step z-score")
    axes[2].set_ylabel("z")
    axes[2].legend(loc="best")


def _plot_class_panels(axes, record, wn, cfg, prefix_stats, audit_cfg):
    """绘制第三阶段类内相似性视图。"""
    stats = prefix_stats.get(record.ref_pool_scope or record.prefix_scope)
    if stats is not None:
        fill_between_segments_without_bad_bands(axes[1], wn, stats["q10"], stats["q90"], cfg.bad_bands, color="0.85", alpha=0.55, label="reference q10-q90")
        plot_segments_without_bad_bands(axes[1], wn, stats["center"], cfg.bad_bands, color="0.35", linewidth=1.0, label="reference median")
    plot_segments_without_bad_bands(axes[1], wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample SNV")
    axes[1].set_title("Sample vs class reference pool")
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    if record.local_pos_z is not None:
        plot_segments_without_bad_bands(axes[2], wn, record.local_pos_z, cfg.bad_bands, color="C4", linewidth=1.0, label="local positive z")
        axes[2].axhline(audit_cfg.class_local_z_min, color="darkorange", linestyle="--", linewidth=1.0, label="local z threshold")
    _shade_ranges(axes[2], wn, record.local_pos_ranges)
    axes[2].set_title("Local positive residual anomaly")
    axes[2].set_ylabel("z")
    axes[2].legend(loc="best")


def _plot_invalid_panels(axes, record, wn, cfg, audit_cfg):
    """绘制第一阶段无效谱自身质量视图。"""
    plot_segments_without_bad_bands(axes[1], wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample SNV")
    axes[1].set_title("Preprocessed sample")
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    smooth_window = audit_cfg.invalid_noise_smooth_points
    smooth = moving_average(record.z, smooth_window)
    detail = record.z - smooth
    plot_segments_without_bad_bands(axes[2], wn, record.z, cfg.bad_bands, color="0.70", linewidth=0.7, label="SNV")
    plot_segments_without_bad_bands(axes[2], wn, smooth, cfg.bad_bands, color="C1", linewidth=1.1, label=f"smooth {smooth_window}pt")
    plot_segments_without_bad_bands(axes[2], wn, detail, cfg.bad_bands, color="C4", linewidth=0.7, label="detail")
    axes[2].set_title("Self noise / structure view")
    axes[2].set_ylabel("SNV")
    axes[2].legend(loc="best")


def _record_summary_text(record):
    """生成候选谱图底部的文本摘要。"""
    return (
        f"Stage: {record.stage}\n"
        f"Decision: {record.decision}\n"
        f"Reasons: {'; '.join(record.reasons)}\n"
        f"Labels: {'; '.join(reason_labels(record.reasons))}\n"
        f"raw_wn={record.raw_wn_min:.1f}-{record.raw_wn_max:.1f}, coverage={record.coverage_ratio:.3f}\n"
        f"long_flat_points={record.long_flat_points}, flat_fraction={record.flat_fraction:.3f}, roughness={record.roughness:.4f}\n"
        f"smooth_range={record.smooth_range:.4f}, detail_noise={record.detail_noise:.4f}, structure_ratio={record.structure_ratio:.3f}\n"
        f"wide_bump_count={record.wide_bump_count}, z={record.wide_bump_max_z:.3f}, "
        f"area={record.wide_bump_area:.3f}, width_points={record.wide_bump_width_points}, "
        f"width_cm={record.wide_bump_width_cm:.1f}, center={record.wide_bump_center_cm:.1f}\n"
        f"edge_jump_z={record.wide_edge_jump_z:.3f}, left={record.wide_left_edge_jump_z:.3f}, "
        f"right={record.wide_right_edge_jump_z:.3f}\n"
        f"rising_area={record.rising_region_area:.3f}, rising_width={record.rising_region_width_cm:.1f}\n"
        f"ref_pool_size={record.ref_pool_size}, other_ref_pool_size={record.other_ref_pool_size}, "
        f"corr_ref={record.corr_ref:.3f}, nearest_ref_corr={record.nearest_ref_corr:.3f}, rmse_to_ref={record.rmse_to_ref:.3f}\n"
        f"local_pos_count={record.local_pos_count}, local_pos_max_z={record.local_pos_max_z:.3f}, "
        f"local_pos_area={record.local_pos_area:.3f}, local_pos_width_points={record.local_pos_width_points}, "
        f"local_pos_center={record.local_pos_center_cm:.1f}\n"
        f"folder_candidate_count={record.folder_candidate_count}, folder_candidate_fraction={record.folder_candidate_fraction:.3f}\n"
        f"cosmic_total={record.cosmic_total}, narrow={record.cosmic_narrow}, peak={record.cosmic_peak}\n"
        f"risk_score={record.risk_score:.3f}"
    )


def plot_stage_candidate(record, out_path, profile, cfg, prefix_stats, audit_cfg):
    """绘制单条候选谱的阶段复核图。"""
    if record.z is None:
        return

    wn = output_wn(cfg)[: record.z.size]
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
    _plot_raw_panel(axes[0], record, profile, cfg)

    if record.stage == "anomalous-cosmic" and record.sp is not None:
        _plot_anomalous_panels(axes, record, wn, cfg, audit_cfg)
    elif record.stage == "class-similarity":
        _plot_class_panels(axes, record, wn, cfg, prefix_stats, audit_cfg)
    else:
        _plot_invalid_panels(axes, record, wn, cfg, audit_cfg)

    axes[3].axis("off")
    text = _record_summary_text(record)
    axes[3].text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(record.rel_path)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _select_figure_targets(delete_records, review_records, folder_targets, max_figures):
    """按总图数上限分配单谱图和文件夹图名额。"""
    if max_figures < 0:
        return [], []
    if max_figures == 0:
        return delete_records + review_records, folder_targets[:100]

    folder_quota = min(len(folder_targets), 100, max(1, max_figures // 5)) if folder_targets else 0
    spectrum_quota = max(0, max_figures - folder_quota)
    review_quota = min(len(review_records), spectrum_quota // 2)
    delete_quota = min(len(delete_records), spectrum_quota - review_quota)
    review_quota = min(len(review_records), spectrum_quota - delete_quota)
    return delete_records[:delete_quota] + review_records[:review_quota], folder_targets[:folder_quota]


def write_stage_figures(out_dir, records, profile, cfg, prefix_stats, max_spectrum_figures, audit_cfg):
    """按图数量上限输出候选谱和文件夹复核图。"""
    delete_records = sorted(
        [record for record in records if record.decision == "remove_candidate" and record.z is not None],
        key=lambda item: (-item.risk_score, item.rel_path),
    )
    review_records = sorted(
        [record for record in records if record.decision == "review_candidate" and record.z is not None],
        key=lambda item: (-item.risk_score, item.rel_path),
    )
    folder_targets = _class_folder_review_targets(records, audit_cfg) if records and records[0].stage == "class-similarity" else []
    selected, folder_targets = _select_figure_targets(
        delete_records,
        review_records,
        folder_targets,
        max_spectrum_figures,
    )

    for record in selected:
        kind = "delete" if record.decision == "remove_candidate" else "review"
        out_path = out_dir / "figures" / kind / record.genus / record.folder / f"{Path(record.file).stem}.png"
        plot_stage_candidate(record, out_path, profile, cfg, prefix_stats, audit_cfg)

    for record in folder_targets:
        out_path = out_dir / "figures" / "folder_review" / record.genus / f"{record.folder}.png"
        plot_class_folder_review(record, out_path, profile, cfg, prefix_stats, audit_cfg)

    folder_figures = len(folder_targets)
    return {"spectrum": len(selected), "folder": folder_figures, "total": len(selected) + folder_figures}


def write_stage_summary(out_dir, dataset_name, records, cfg, audit_cfg, dataset_dir, init_root, stage, figure_counts, moved):
    """写入阶段扫描 summary.md 和 summary.json。"""
    valid = [record for record in records if record.z is not None]
    skipped = [record for record in records if record.z is None]
    remove_records = [record for record in valid if record.decision == "remove_candidate"]
    review_records = [record for record in valid if record.decision == "review_candidate"]
    reason_counts = {}
    for record in remove_records + review_records:
        for reason in record.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines = [
        f"# dataset/{dataset_name} audit {stage}",
        "",
        "## 总体",
        "",
        f"- 阶段：{stage_title(stage)}",
        f"- 数据目录：`{init_root}`",
        f"- 输出目录：`{out_dir}`",
        f"- 总光谱数：{len(records)}",
        f"- 有效光谱数：{len(valid)}",
        f"- 跳过光谱数：{len(skipped)}",
        f"- 删除候选：{len(remove_records)}",
        f"- 复核候选：{len(review_records)}",
        f"- 单谱候选图：{figure_counts['spectrum']}",
        f"- 文件夹复核图：{figure_counts['folder']}",
        f"- 图总数：{figure_counts['total']}",
        f"- 已执行移动：{'是' if moved else '否'}",
        f"- 删除目录：`delete/{STAGE_DELETE_CATEGORY[stage]}`",
        "",
        "## 原因统计",
        "",
    ]
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 删除候选", ""])
    if remove_records:
        for record in sorted(remove_records, key=lambda item: (-item.risk_score, item.rel_path))[:150]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(record.reasons)}")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 复核候选", ""])
    if review_records:
        for record in sorted(review_records, key=lambda item: (-item.risk_score, item.rel_path))[:150]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(record.reasons)}")
    else:
        lines.append("- 暂无")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "dataset_dir": str(dataset_dir),
        "init_root": str(init_root),
        "output_dir": str(out_dir),
        "stage": stage,
        "delete_category": STAGE_DELETE_CATEGORY[stage],
        "pipeline_config": asdict(cfg),
        "audit_config": asdict(audit_cfg),
        "total_spectra": len(records),
        "valid_spectra": len(valid),
        "skipped_spectra": len(skipped),
        "delete_candidates": len(remove_records),
        "review_candidates": len(review_records),
        "reason_counts": reason_counts,
        "figures": figure_counts,
        "moved": moved,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stage_outputs(out_dir, records, profile, cfg, audit_cfg, prefix_stats, dataset_dir, init_root, stage, max_spectrum_figures, moved=False):
    """写入阶段扫描的 CSV、图和摘要。"""
    all_rows = [record_to_row(record) for record in records]
    fieldnames = list(all_rows[0].keys()) if all_rows else None
    candidate_rows = sorted(
        [record_to_row(record) for record in records if record.decision in {"remove_candidate", "review_candidate", "skip"}],
        key=lambda row: (row["decision"] != "remove_candidate", -float(row["risk_score"] or 0), row["rel_path"]),
    )
    delete_rows = [row for row in candidate_rows if row["decision"] == "remove_candidate"]
    review_rows = [row for row in candidate_rows if row["decision"] == "review_candidate"]

    delete_csv = out_dir / "delete_candidates.csv"
    write_csv(delete_csv, delete_rows, fieldnames)
    write_csv(out_dir / "review_candidates.csv", review_rows, fieldnames)
    write_csv(out_dir / "all_spectra_scores.csv", all_rows, fieldnames)
    (out_dir / "delete_manifest.txt").write_text("".join(f"{row['rel_path']}\n" for row in delete_rows), encoding="utf-8")

    figure_counts = write_stage_figures(out_dir, records, profile, cfg, prefix_stats, max_spectrum_figures, audit_cfg)
    write_stage_summary(out_dir, profile.dataset_name, records, cfg, audit_cfg, dataset_dir, init_root, stage, figure_counts, moved)
    return {"delete_csv": delete_csv, "delete_rows": delete_rows, "review_rows": review_rows, "figures": figure_counts}


def full_scan(
    dataset,
    stage="invalid",
    output_root=None,
    timestamp=None,
    max_spectrum_figures=200,
    move=False,
):
    """执行一次全库分阶段扫描，可选择移动强候选。"""
    stage = validate_stage(stage)
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    cfg = DEFAULT_PIPELINE_CONFIG
    audit_cfg = DEFAULT_AUDIT_CONFIG
    init_root = dataset_dir / profile.root_init
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init folder: {init_root}")

    if timestamp is None:
        mode_name = "move" if move else "dry_run"
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{stage}_{mode_name}"
    output_root = Path(output_root) if output_root else dataset_dir / "audit_full_scan"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    out_dir = output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_dir}")
    print(f"Stage: {stage}")
    print(f"Input: {init_root}")
    print(f"Output: {out_dir}")

    records = load_audit_records(profile, cfg, init_root, SpectrumRecord)
    prefix_stats = score_stage(records, cfg, stage, audit_cfg)
    result = write_stage_outputs(out_dir, records, profile, cfg, audit_cfg, prefix_stats, dataset_dir, init_root, stage, max_spectrum_figures, moved=False)

    if move and result["delete_rows"]:
        from raman.audit.move import move_items

        log_path = out_dir / "move_log.txt"
        with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file):
            move_items(dataset, from_list=str(result["delete_csv"]), dry_run=False, category=STAGE_DELETE_CATEGORY[stage])
        write_stage_summary(out_dir, profile.dataset_name, records, cfg, audit_cfg, dataset_dir, init_root, stage, result["figures"], moved=True)
        print(f"- Moved candidates: {len(result['delete_rows'])}")
        print(f"- Move log: {log_path}")

    print("\nStage scan finished:")
    print(f"- Summary: {out_dir / 'summary.md'}")
    print(f"- Delete candidates: {result['delete_csv']}")
    print(f"- Review candidates: {out_dir / 'review_candidates.csv'}")
    print(f"- All scores: {out_dir / 'all_spectra_scores.csv'}")
    print(f"- Figures: {out_dir / 'figures'}")
    return out_dir


def build_parser():
    """构建 full 子命令参数解析器。"""
    parser = argparse.ArgumentParser(description="分阶段清洗 Raman audit")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("--stage", choices=("invalid", "anomalous-cosmic", "class-similarity"), default="invalid", help="本次执行的清洗阶段")
    parser.add_argument("--output-root", default=None, help="覆盖输出根目录")
    parser.add_argument("--timestamp", default=None, help="覆盖输出目录名")
    parser.add_argument("--max-spectrum-figures", type=int, default=200, help="最多输出多少张候选图；默认 200，0 表示全部，负数表示不输出")
    parser.add_argument("--move", action="store_true", help="移动本阶段 delete_candidates 到对应 delete 分类目录")
    parser.add_argument("--move-strong", dest="move", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-remove-candidates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--max-folder-candidates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--max-folder-figures", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    """执行 full 子命令。"""
    args = build_parser().parse_args(argv)
    full_scan(
        args.dataset,
        stage=args.stage,
        output_root=args.output_root,
        timestamp=args.timestamp,
        max_spectrum_figures=args.max_spectrum_figures,
        move=args.move,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
