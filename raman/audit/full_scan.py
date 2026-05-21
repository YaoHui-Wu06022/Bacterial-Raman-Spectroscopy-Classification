"""全库只读异常谱复查入口"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.audit.common import (
    PROJECT_ROOT,
    fill_between_segments_without_bad_bands,
    output_wn,
    plot_segments_without_bad_bands,
    preprocess_spectrum_for_audit,
    resolve_dataset,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.scoring import (
    SpectrumRecord,
    build_folder_records,
    cap_remove_candidates,
    classify_records,
    folder_to_row,
    reason_labels,
    record_to_row,
    score_groups,
    score_references,
    top_folder_candidates,
)
from raman.data.archive import iter_arc_dirs
from raman.data.build import DEFAULT_PIPELINE_CONFIG, _cosmic_ray_enabled
from raman.data.offline import remove_cosmic_rays
from raman.data.spectrum import read_arc_data


def write_csv(path, rows, fieldnames=None):
    """写 CSV，空结果也尽量保留表头"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else None)
    if fieldnames:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_records(profile, cfg, dataset_dir, init_root):
    """读取 init 下所有叶子目录并执行当前离线预处理"""
    records = []
    wn_ref = cfg.build_wn_ref()
    for root, arc_files in iter_arc_dirs(init_root):
        rel_group = root.relative_to(init_root)
        genus = rel_group.parts[0] if len(rel_group.parts) >= 1 else "."
        folder = rel_group.parts[1] if len(rel_group.parts) >= 2 else root.name
        group = rel_group.as_posix()
        print(f"[Preprocess] {group}: {len(arc_files)} files")

        for filename in arc_files:
            path = root / filename
            payload = preprocess_spectrum_for_audit(path, profile, cfg, wn_ref=wn_ref, include_raw=False)
            record = SpectrumRecord(
                path=path,
                rel_path=path.relative_to(init_root).as_posix(),
                group=group,
                genus=genus,
                folder=folder,
                file=filename,
                skip_reason=payload.get("skip_reason", ""),
            )
            if not record.skip_reason:
                stats = payload["cosmic_stats"]
                record.z = np.asarray(payload["z"], dtype=np.float32)
                record.sp = np.asarray(payload["sp"], dtype=np.float32)
                record.cosmic_total = int(stats)
                record.cosmic_narrow = int(getattr(stats, "narrow", 0))
                record.cosmic_peak = int(getattr(stats, "peak", 0))
                record.cosmic_residual = int(getattr(stats, "residual", 0))
            records.append(record)
    return records


def cosmic_clean_for_plot(wn, sp, profile, cfg):
    """仅用于复核图展示宇宙射线清理结果"""
    if not _cosmic_ray_enabled(profile, cfg):
        return np.asarray(sp, dtype=np.float32)
    cleaned, _ = remove_cosmic_rays(
        sp,
        window_cm=cfg.cosmic_ray_narrow_window_cm,
        threshold=cfg.cosmic_ray_threshold,
        max_iter=cfg.cosmic_ray_max_iter,
        valid_mask=None,
        wn=wn,
        peak_prominence_z=cfg.cosmic_ray_peak_prominence_z,
        peak_width_max_cm=cfg.cosmic_ray_peak_width_max_cm,
        peak_ratio_z_per_cm=cfg.cosmic_ray_peak_ratio_z_per_cm,
        peak_pad_cm=cfg.cosmic_ray_peak_pad_cm,
        peak_rel_height=cfg.cosmic_ray_peak_rel_height,
        residual_threshold_z=cfg.cosmic_ray_residual_threshold_z,
        residual_max_points_fraction=cfg.cosmic_ray_residual_max_points_fraction,
    )
    return cleaned


def plot_spectrum_candidate(record, out_path, profile, cfg, group_stats, ref_stats, audit_cfg):
    """输出单谱复核图"""
    raw_wn, raw_sp = read_arc_data(record.path)
    raw_cosmic = cosmic_clean_for_plot(raw_wn, raw_sp, profile, cfg)
    wn = output_wn(cfg)
    gstats = group_stats.get(record.group)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
    plot_segments_without_bad_bands(axes[0], raw_wn, raw_sp, cfg.bad_bands, color="0.60", linewidth=0.9, label="raw")
    plot_segments_without_bad_bands(axes[0], raw_wn, raw_cosmic, cfg.bad_bands, color="C0", linewidth=0.9, label="cosmic cleaned")
    axes[0].set_title("Raw / cosmic cleaned")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="best")

    fill_between_segments_without_bad_bands(axes[1], wn, gstats["q10"], gstats["q90"], cfg.bad_bands, color="C0", alpha=0.16, label="group q10-q90")
    plot_segments_without_bad_bands(axes[1], wn, gstats["center"], cfg.bad_bands, color="C0", linewidth=1.4, label="group median")
    if ref_stats is not None:
        plot_segments_without_bad_bands(axes[1], wn, ref_stats["ref_median"], cfg.bad_bands, color="C2", linewidth=1.1, label="ref median")
    plot_segments_without_bad_bands(axes[1], wn, record.z, cfg.bad_bands, color="C3", linewidth=1.0, label="sample SNV")
    axes[1].set_title("After full preprocessing")
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(loc="best")

    group_z = (record.z - gstats["center"]) / gstats["wave_scale"]
    plot_segments_without_bad_bands(axes[2], wn, group_z, cfg.bad_bands, color="C4", linewidth=0.9, label="group robust z")
    if ref_stats is not None:
        ref_z = (record.z - ref_stats["ref_median"]) / ref_stats["ref_scale"]
        plot_segments_without_bad_bands(axes[2], wn, ref_z, cfg.bad_bands, color="C1", linewidth=0.8, alpha=0.75, label="ref robust z")
    axes[2].axhline(audit_cfg.group_point_z_threshold, color="C3", linestyle="--", linewidth=0.8)
    axes[2].axhline(-audit_cfg.group_point_z_threshold, color="C3", linestyle="--", linewidth=0.8)
    for pos in record.step_positions:
        axes[2].axvline(pos, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_title("Robust z-score and detected steps")
    axes[2].set_ylabel("z")
    axes[2].legend(loc="best")

    axes[3].axis("off")
    text = (
        f"Decision: {record.decision}\n"
        f"Reasons: {'; '.join(record.reasons)}\n"
        f"Labels: {'; '.join(reason_labels(record.reasons))}\n"
        f"corr_group={record.corr_group:.3f}, bad_ratio_group={record.bad_ratio_group:.3f}\n"
        f"corr_ref={record.corr_ref:.3f}, nearest_ref_corr={record.nearest_ref_corr:.3f}, bad_ratio_z6={record.bad_ratio_z6:.3f}\n"
        f"step_count={record.step_count}, bad_band_edge_step={record.bad_band_edge_step_count}, residual_regions={record.residual_cosmic_regions}\n"
        f"cosmic_total={record.cosmic_total}"
    )
    axes[3].text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(record.rel_path)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_folder_candidate(folder_record, records, out_path, cfg, group_stats, ref_stats):
    """输出文件夹复核图"""
    valid = [record for record in records if record.group == folder_record.group and record.z is not None]
    if not valid:
        return
    wn = output_wn(cfg)
    gstats = group_stats.get(folder_record.group)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for record in valid:
        color = "C3" if record.decision == "remove_candidate" else ("C1" if record.decision == "review_candidate" else "0.75")
        alpha = 0.75 if record.decision != "keep" else 0.25
        plot_segments_without_bad_bands(axes[0], wn, record.z, cfg.bad_bands, color=color, alpha=alpha, linewidth=0.75)
    plot_segments_without_bad_bands(axes[0], wn, gstats["center"], cfg.bad_bands, color="C0", linewidth=1.8, label="group median")
    if ref_stats is not None:
        plot_segments_without_bad_bands(axes[0], wn, ref_stats["ref_median"], cfg.bad_bands, color="C2", linewidth=1.4, label="ref median")
    axes[0].set_ylabel("SNV intensity")
    axes[0].set_title("Folder spectra overview")
    axes[0].legend(loc="best")

    axes[1].axis("off")
    text = (
        f"Decision: {folder_record.decision}\n"
        f"Reasons: {'; '.join(folder_record.reasons)}\n"
        f"Files/valid/skipped: {folder_record.files}/{folder_record.valid}/{folder_record.skipped}\n"
        f"Remove/review candidates: {folder_record.remove_candidates}/{folder_record.review_candidates}\n"
        f"candidate_fraction={folder_record.candidate_fraction:.3f}, ref_remove_fraction={folder_record.ref_remove_fraction:.3f}\n"
        f"folder_corr_ref={folder_record.folder_corr_ref:.3f}, p95_cosmic={folder_record.p95_cosmic_total:.1f}, max_cosmic={folder_record.max_cosmic_total}"
    )
    axes[1].text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle(folder_record.group)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_figures(out_dir, records, folders, profile, cfg, group_stats, ref_stats_by_group, max_spectrum_figures, max_folder_figures, audit_cfg):
    """按数量上限输出复核图"""
    candidates = [record for record in records if record.decision in {"remove_candidate", "review_candidate"} and record.z is not None]
    candidates = sorted(candidates, key=lambda item: (item.decision != "remove_candidate", -item.risk_score, item.rel_path))
    if max_spectrum_figures < 0:
        spectrum_to_plot = []
    elif max_spectrum_figures == 0:
        spectrum_to_plot = candidates
    else:
        spectrum_to_plot = candidates[:max_spectrum_figures]
    for record in spectrum_to_plot:
        out_path = out_dir / "figures" / "spectra" / record.genus / record.folder / f"{Path(record.file).stem}.png"
        plot_spectrum_candidate(record, out_path, profile, cfg, group_stats, ref_stats_by_group.get(record.group), audit_cfg)

    folder_candidates = top_folder_candidates(folders, max_folder_figures)
    for folder in folder_candidates:
        out_path = out_dir / "figures" / "folders" / folder.genus / f"{folder.folder}.png"
        plot_folder_candidate(folder, records, out_path, cfg, group_stats, ref_stats_by_group.get(folder.group))
    return len(spectrum_to_plot), len(folder_candidates)


def parameter_advice(records, cfg):
    """根据候选分布给出保守调参建议"""
    valid = [record for record in records if record.z is not None]
    if not valid:
        return "没有有效光谱，无法评估宇宙射线参数"
    residual_records = [record for record in valid if "residual_cosmic_like" in record.reasons]
    high_cosmic = [record for record in valid if "excessive_cosmic_cleanup" in record.reasons]
    residual_ratio = len(residual_records) / len(valid)
    high_cosmic_ratio = len(high_cosmic) / len(valid)
    residual_groups = len({record.group for record in residual_records})
    if residual_ratio > 0.03 and residual_groups > 20:
        return "残留正向异常在多个文件夹中系统性出现，后续可考虑小幅调整宇宙射线峰宽或 residual 阈值；本轮不自动改参数"
    if high_cosmic_ratio > 0.05:
        return "过量宇宙射线替换样本比例偏高，建议先看候选图确认是否为噪声态或阶梯谱；当前不直接改参数"
    return "未看到必须调参的系统性证据，当前宇宙射线参数建议保持不变"


def write_summary(out_dir, dataset_name, records, folders, reported_folders, cfg, dataset_dir, init_root, fig_counts):
    """写中文报告和机器可读摘要"""
    valid = [record for record in records if record.z is not None]
    skipped = [record for record in records if record.z is None]
    remove_records = [record for record in valid if record.decision == "remove_candidate"]
    review_records = [record for record in valid if record.decision == "review_candidate"]
    folder_candidates = [folder for folder in reported_folders if folder.decision in {"remove_candidate", "review_candidate"}]
    cosmic_totals = np.asarray([record.cosmic_total for record in valid], dtype=np.float32)

    lines = [
        f"# dataset/{dataset_name} 全库异常谱复查报告",
        "",
        "## 总体",
        "",
        f"- 数据目录：`{init_root}`",
        f"- 输出目录：`{out_dir}`",
        f"- 小文件夹数：{len(folders)}",
        f"- 总光谱数：{len(records)}",
        f"- 有效光谱数：{len(valid)}",
        f"- 跳过光谱数：{len(skipped)}",
        f"- 建议移除候选谱：{len(remove_records)}",
        f"- 仅复核候选谱：{len(review_records)}",
        f"- 输出文件夹候选：{len(folder_candidates)}",
        f"- 已输出候选谱图：{fig_counts[0]}",
        f"- 已输出文件夹图：{fig_counts[1]}",
        "- 本报告只读生成，没有移动或删除任何 `.arc_data`",
        "",
        "## 宇宙射线统计",
        "",
        f"- 当前参数：`peak_width_max_cm={cfg.cosmic_ray_peak_width_max_cm}`, `peak_prominence_z={cfg.cosmic_ray_peak_prominence_z}`, `residual_threshold_z={cfg.cosmic_ray_residual_threshold_z}`",
    ]
    if cosmic_totals.size:
        lines.extend(
            [
                f"- cosmic_total：中位数 {np.median(cosmic_totals):.1f}，p90 {np.quantile(cosmic_totals, 0.90):.1f}，p95 {np.quantile(cosmic_totals, 0.95):.1f}，最大 {np.max(cosmic_totals):.0f}",
                f"- residual_cosmic_like 候选：{sum('residual_cosmic_like' in record.reasons for record in valid)}",
                f"- step_like / bad_band_edge_step 候选：{sum(('step_like_spectrum' in record.reasons or 'bad_band_edge_step' in record.reasons) for record in valid)}",
            ]
        )
    lines.extend(["", "## 参数建议", "", f"- {parameter_advice(records, cfg)}", ""])
    lines.extend(["## 建议移除候选谱", ""])
    if remove_records:
        for record in sorted(remove_records, key=lambda item: (-item.risk_score, item.rel_path))[:80]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(reason_labels(record.reasons))}（{'; '.join(record.reasons)}）")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 仅复核候选谱", ""])
    if review_records:
        for record in sorted(review_records, key=lambda item: (-item.risk_score, item.rel_path))[:80]:
            lines.append(f"- `{record.rel_path}`：{'; '.join(reason_labels(record.reasons))}（{'; '.join(record.reasons)}）")
    else:
        lines.append("- 暂无")

    lines.extend(["", "## 文件夹候选", ""])
    if folder_candidates:
        for folder in sorted(folder_candidates, key=lambda item: (item.decision != "remove_candidate", -item.candidate_fraction, item.group)):
            lines.append(
                f"- `{folder.group}`：{folder.decision}，{'; '.join(folder.reasons)}，"
                f"候选比例 {folder.candidate_fraction:.2%}，folder_corr_ref={folder.folder_corr_ref:.3f}"
            )
    else:
        lines.append("- 暂无")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "dataset_dir": str(dataset_dir),
        "init_root": str(init_root),
        "output_dir": str(out_dir),
        "pipeline_config": asdict(cfg),
        "audit_config": asdict(DEFAULT_AUDIT_CONFIG),
        "total_folders": len(folders),
        "total_spectra": len(records),
        "valid_spectra": len(valid),
        "skipped_spectra": len(skipped),
        "remove_candidates": len(remove_records),
        "review_candidates": len(review_records),
        "folder_candidates": len(folder_candidates),
        "figures": {"spectra": fig_counts[0], "folders": fig_counts[1]},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def full_scan(dataset, output_root=None, timestamp=None, max_remove_candidates=100, max_folder_candidates=5, max_spectrum_figures=0, max_folder_figures=0):
    """执行全库只读复查"""
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    cfg = DEFAULT_PIPELINE_CONFIG
    audit_cfg = DEFAULT_AUDIT_CONFIG
    init_root = dataset_dir / profile.root_init
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init folder: {init_root}")

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_root) if output_root else dataset_dir / "audit_full_scan"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    out_dir = output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_dir}")
    print(f"Input: {init_root}")
    print(f"Output: {out_dir}")

    records = load_records(profile, cfg, dataset_dir, init_root)
    group_stats = score_groups(records, cfg, audit_cfg)
    ref_stats_by_group = score_references(records, audit_cfg)
    classify_records(records, ref_stats_by_group, audit_cfg)
    cap_remove_candidates(records, max_remove_candidates)
    folder_records = build_folder_records(records, group_stats, ref_stats_by_group, audit_cfg)
    reported_folder_records = top_folder_candidates(folder_records, max_folder_candidates)

    candidate_rows = [
        record_to_row(record)
        for record in records
        if record.decision in {"remove_candidate", "review_candidate", "skip"}
    ]
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["decision"] != "remove_candidate", row["group"], row["file"]))
    delete_rows = [row for row in candidate_rows if row["decision"] == "remove_candidate"]
    folder_rows = [folder_to_row(record) for record in reported_folder_records]
    all_rows = [record_to_row(record) for record in records]

    spectrum_fieldnames = list(all_rows[0].keys()) if all_rows else None
    folder_fieldnames = list(folder_rows[0].keys()) if folder_rows else [
        "decision",
        "reasons",
        "group",
        "files",
        "remove_candidates",
        "review_candidates",
        "candidate_fraction",
        "folder_corr_ref",
        "step_spectra",
    ]
    write_csv(out_dir / "spectrum_candidates.csv", candidate_rows, spectrum_fieldnames)
    write_csv(out_dir / "delete_candidates.csv", delete_rows, spectrum_fieldnames)
    write_csv(out_dir / "folder_candidates.csv", folder_rows, folder_fieldnames)
    write_csv(out_dir / "all_spectra_scores.csv", all_rows, spectrum_fieldnames)
    (out_dir / "delete_manifest.txt").write_text(
        "".join(f"{row['rel_path']}\n" for row in delete_rows),
        encoding="utf-8",
    )

    fig_counts = write_figures(
        out_dir,
        records,
        reported_folder_records,
        profile,
        cfg,
        group_stats,
        ref_stats_by_group,
        max_spectrum_figures,
        max_folder_figures,
        audit_cfg,
    )
    write_summary(out_dir, profile.dataset_name, records, folder_records, reported_folder_records, cfg, dataset_dir, init_root, fig_counts)

    print("\nFull scan finished:")
    print(f"- Summary: {out_dir / 'summary.md'}")
    print(f"- Spectrum candidates: {out_dir / 'spectrum_candidates.csv'}")
    print(f"- Delete candidates: {out_dir / 'delete_candidates.csv'}")
    print(f"- Folder candidates: {out_dir / 'folder_candidates.csv'}")
    print(f"- All scores: {out_dir / 'all_spectra_scores.csv'}")
    print(f"- Figures: {out_dir / 'figures'}")
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description="全库只读异常谱复查")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("--output-root", default=None, help="覆盖输出根目录")
    parser.add_argument("--timestamp", default=None, help="覆盖时间戳目录名")
    parser.add_argument("--max-remove-candidates", type=int, default=100, help="最多保留多少个高置信移除候选")
    parser.add_argument("--max-folder-candidates", type=int, default=5, help="最多输出多少个文件夹候选")
    parser.add_argument("--max-spectrum-figures", type=int, default=0, help="最多输出多少张单谱复核图，0 表示全部，负数表示不输出")
    parser.add_argument("--max-folder-figures", type=int, default=0, help="最多输出多少张文件夹复核图")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
