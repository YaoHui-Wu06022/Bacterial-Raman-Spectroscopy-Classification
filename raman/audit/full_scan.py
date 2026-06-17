"""全库分阶段清洗入口。"""

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
    cosmic_clean_for_plot,
    load_audit_records,
    write_csv,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.scoring import (
    INVALID_NOISE_SMOOTH_POINTS,
    SpectrumRecord,
    STAGE_DELETE_CATEGORY,
    reason_labels,
    record_to_row,
    score_stage,
    stage_title,
    validate_stage,
)
from raman.data.build import DEFAULT_PIPELINE_CONFIG
from raman.data.io import read_arc_data
from raman.tool.array import moving_average
from raman.tool.dataset import resolve_dataset
from raman.tool.path import PROJECT_ROOT
from raman.tool.plotting import (
    fill_between_segments_without_bad_bands,
    plot_segments_without_bad_bands,
)
from raman.tool.spectrum import output_wavenumbers as output_wn


def _plot_raw_panel(ax, record, profile, cfg):
    """绘制原始谱和宇宙射线清理后谱。"""
    raw_wn, raw_sp = read_arc_data(record.path)
    raw_cosmic = cosmic_clean_for_plot(raw_wn, raw_sp, profile, cfg)
    plot_segments_without_bad_bands(ax, raw_wn, raw_sp, cfg.bad_bands, color="0.60", linewidth=0.9, label="raw")
    plot_segments_without_bad_bands(ax, raw_wn, raw_cosmic, cfg.bad_bands, color="C0", linewidth=0.9, label="cosmic cleaned")
    ax.set_title("Raw / cosmic cleaned")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best")


def _plot_class_panel(ax, record, wn, cfg, prefix_stats):
    """绘制第二阶段类内相似性视图。"""
    stats = prefix_stats.get(record.ref_pool_scope or record.prefix_scope)
    if stats is not None:
        fill_between_segments_without_bad_bands(ax, wn, stats["q10"], stats["q90"], cfg.bad_bands, color="0.85", alpha=0.55, label="reference q10-q90")
        plot_segments_without_bad_bands(ax, wn, stats["center"], cfg.bad_bands, color="0.35", linewidth=1.0, label="reference median")
    plot_segments_without_bad_bands(ax, wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample standardized")
    ax.set_title("Sample vs class reference pool")
    ax.set_ylabel("Standardized intensity")
    ax.legend(loc="best")


def _shade_index_ranges(ax, wn, ranges):
    """按索引区间在图上标出局部异常段。"""
    for start, end in ranges:
        if 0 <= start < end <= len(wn):
            ax.axvspan(float(wn[start]), float(wn[end - 1]), color="darkorange", alpha=0.16)


def _plot_local_residual_panel(ax, record, wn, cfg, audit_cfg):
    """绘制第二阶段局部正残差异常视图"""
    if record.local_residual_curve is not None:
        plot_segments_without_bad_bands(ax, wn, record.local_residual_curve, cfg.bad_bands, color="C4", linewidth=0.9, label="sample - reference")
    threshold = audit_cfg.class_local_residual_min
    ax.axhline(threshold, color="darkorange", linestyle="--", linewidth=0.9, label="positive threshold")
    _shade_index_ranges(ax, wn, record.local_residual_ranges)
    ax.set_title("Local positive residual anomaly")
    ax.set_ylabel("Residual")
    ax.legend(loc="best")


def _plot_invalid_panels(axes, record, wn, cfg, audit_cfg):
    """绘制第一阶段无效谱自身质量视图。"""
    plot_segments_without_bad_bands(axes[1], wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample standardized")
    axes[1].set_title("Preprocessed sample")
    axes[1].set_ylabel("Standardized intensity")
    axes[1].legend(loc="best")

    smooth = moving_average(record.z, INVALID_NOISE_SMOOTH_POINTS)
    detail = record.z - smooth
    plot_segments_without_bad_bands(axes[2], wn, record.z, cfg.bad_bands, color="0.70", linewidth=0.7, label="standardized")
    plot_segments_without_bad_bands(axes[2], wn, smooth, cfg.bad_bands, color="C1", linewidth=1.1, label=f"smooth {INVALID_NOISE_SMOOTH_POINTS}pt")
    plot_segments_without_bad_bands(axes[2], wn, detail, cfg.bad_bands, color="C4", linewidth=0.7, label="detail")
    axes[2].set_title("Self noise / structure view")
    axes[2].set_ylabel("Standardized value")
    axes[2].legend(loc="best")


def _record_summary_text(record):
    """生成候选谱图底部的文本摘要。"""
    lines = [
        f"Stage: {record.stage}",
        f"Decision: {record.decision}",
        f"Reasons: {'; '.join(record.reasons)}",
        f"Labels: {'; '.join(reason_labels(record.reasons))}",
    ]

    if record.stage == "invalid":
        lines.extend(
            [
                f"raw_wn={record.raw_wn_min:.1f}-{record.raw_wn_max:.1f}, coverage={record.coverage_ratio:.3f}",
                f"long_flat_points={record.long_flat_points}, flat_fraction={record.flat_fraction:.3f}, roughness={record.roughness:.4f}",
                f"smooth_range={record.smooth_range:.4f}, detail_noise={record.detail_noise:.4f}, structure_ratio={record.structure_ratio:.3f}",
            ]
        )
    elif record.stage == "similar":
        lines.extend(
            [
                f"ref_pool_size={record.ref_pool_size}",
                f"corr_ref={record.corr_ref:.3f}, rmse_to_ref={record.rmse_to_ref:.3f}",
                f"local_residual_count={record.local_residual_count}, local_residual_max={record.local_residual_max:.3f}, local_residual_area={record.local_residual_area:.3f}",
                f"local_residual_width_points={record.local_residual_width_points}, local_residual_center_point={record.local_residual_center_point}",
                f"folder_candidate_count={record.folder_candidate_count}, folder_candidate_fraction={record.folder_candidate_fraction:.3f}",
            ]
        )

    lines.append(f"risk_score={record.risk_score:.3f}")
    return "\n".join(lines)


def plot_stage_candidate(record, out_path, profile, cfg, prefix_stats, audit_cfg):
    """绘制单条候选谱的阶段候选图。"""
    if record.z is None:
        return

    wn = output_wn(cfg)[: record.z.size]
    if record.stage == "similar":
        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
        _plot_raw_panel(axes[0], record, profile, cfg)
        _plot_class_panel(axes[1], record, wn, cfg, prefix_stats)
        _plot_local_residual_panel(axes[2], record, wn, cfg, audit_cfg)
        text_ax = axes[3]
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
        _plot_raw_panel(axes[0], record, profile, cfg)
        _plot_invalid_panels(axes, record, wn, cfg, audit_cfg)
        text_ax = axes[3]

    text_ax.axis("off")
    text = _record_summary_text(record)
    text_ax.text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(record.rel_path)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _select_figure_targets(delete_records, max_figures):
    """按图数上限选择删除候选图。"""
    if max_figures < 0:
        return []
    if max_figures == 0:
        return delete_records

    return delete_records[:max_figures]


def write_stage_figures(out_dir, records, profile, cfg, prefix_stats, max_spectrum_figures, audit_cfg):
    """按图数量上限输出删除候选谱图。"""
    delete_records = sorted(
        [record for record in records if record.decision == "remove_candidate" and record.z is not None],
        key=lambda item: (-item.risk_score, item.rel_path),
    )
    selected = _select_figure_targets(delete_records, max_spectrum_figures)

    for record in selected:
        out_path = out_dir / "figures" / "delete" / record.genus / record.folder / f"{Path(record.file).stem}.png"
        plot_stage_candidate(record, out_path, profile, cfg, prefix_stats, audit_cfg)

    return {"spectrum": len(selected), "folder": 0, "total": len(selected)}


def write_stage_summary(out_dir, dataset_name, records, cfg, audit_cfg, dataset_dir, init_root, stage, figure_counts, moved):
    """写入阶段扫描 summary.md 和 summary.json。"""
    valid = [record for record in records if record.z is not None]
    skipped = [record for record in records if record.z is None]
    remove_records = [record for record in valid if record.decision == "remove_candidate"]
    reason_counts = {}
    for record in remove_records:
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
        f"- 单谱候选图：{figure_counts['spectrum']}",
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
        "reason_counts": reason_counts,
        "figures": figure_counts,
        "moved": moved,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stage_outputs(out_dir, records, profile, cfg, audit_cfg, prefix_stats, dataset_dir, init_root, stage, max_spectrum_figures, moved=False):
    """写入阶段扫描的 CSV、图和摘要。"""
    all_rows = [record_to_row(record) for record in records]
    fieldnames = list(all_rows[0].keys()) if all_rows else None
    delete_rows = sorted(
        [record_to_row(record) for record in records if record.decision == "remove_candidate"],
        key=lambda row: (row["decision"] != "remove_candidate", -float(row["risk_score"] or 0), row["rel_path"]),
    )

    delete_csv = out_dir / "delete_candidates.csv"
    write_csv(delete_csv, delete_rows, fieldnames)
    write_csv(out_dir / "all_spectra_scores.csv", all_rows, fieldnames)
    (out_dir / "delete_manifest.txt").write_text("".join(f"{row['rel_path']}\n" for row in delete_rows), encoding="utf-8")

    figure_counts = write_stage_figures(out_dir, records, profile, cfg, prefix_stats, max_spectrum_figures, audit_cfg)
    write_stage_summary(out_dir, profile.dataset_name, records, cfg, audit_cfg, dataset_dir, init_root, stage, figure_counts, moved)
    return {"delete_csv": delete_csv, "delete_rows": delete_rows, "figures": figure_counts}


def full_scan(
    dataset,
    stage="invalid",
    output_root=None,
    timestamp=None,
    max_spectrum_figures=100,
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
    print(f"- All scores: {out_dir / 'all_spectra_scores.csv'}")
    print(f"- Figures: {out_dir / 'figures'}")
    return out_dir


def build_parser():
    """构建 full 子命令参数解析器。"""
    parser = argparse.ArgumentParser(description="分阶段清洗 Raman audit")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("--stage", choices=("invalid", "similar"), default="invalid", help="本次执行的清洗阶段")
    parser.add_argument("--move", action="store_true", help="移动本阶段 delete_candidates 到对应 delete 分类目录")
    return parser


def main(argv=None):
    """执行 full 子命令。"""
    args = build_parser().parse_args(argv)
    full_scan(
        args.dataset,
        stage=args.stage,
        move=args.move,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
