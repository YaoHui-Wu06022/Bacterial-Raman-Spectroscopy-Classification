"""下凹坏段扫描工具。

该脚本用于发现采集过程里可能出现的系统性下凹波段。它只读扫描数据，
输出候选波段、类别/文件夹覆盖率和复核图，不修改原始光谱。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from raman.audit.common import (
    contiguous_regions,
    moving_average,
    preprocess_spectrum_for_audit,
    region_width_cm,
    resolve_audit_input,
    resolve_dataset,
)
from raman.data.build import DEFAULT_PIPELINE_CONFIG


@dataclass(frozen=True)
class BadBandScanConfig:
    """下凹坏段扫描参数。"""

    smooth_points: int = 9
    side_cm: float = 18.0
    min_width_cm: float = 20.0
    max_width_cm: float = 80.0
    width_step_cm: float = 10.0
    stride_cm: float = 2.0
    sample_depth_threshold: float = 0.35
    min_fraction: float = 0.35
    min_median_depth: float = 0.20
    min_q80_depth: float = 0.45
    overlap_ratio: float = 0.50
    max_candidates: int = 12


@dataclass(frozen=True)
class BandCandidate:
    """一个候选坏段窗口的摘要。"""

    score: float
    start_idx: int
    end_idx: int
    band_min: float
    band_max: float
    width_cm: float
    sample_fraction: float
    median_depth: float
    q80_depth: float
    mean_depth: float


def _step_cm(wn):
    diffs = np.abs(np.diff(np.asarray(wn, dtype=np.float32)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-8)]
    return float(np.median(diffs)) if diffs.size else 1.0


def _points_for_cm(wn, cm, minimum):
    return max(int(minimum), int(round(float(cm) / _step_cm(wn))))


def _sample_files(files, max_files, seed):
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return files
    rng = np.random.RandomState(int(seed))
    indices = rng.choice(len(files), size=int(max_files), replace=False)
    return [files[int(i)] for i in sorted(indices)]


def _preprocess_files(files, profile, cfg, input_root, progress_every=500):
    wn_ref = cfg.build_wn_ref()
    spectra = []
    rel_paths = []
    skipped = []

    for idx, path in enumerate(files, start=1):
        if progress_every > 0 and idx % progress_every == 0:
            print(f"[bad-band] preprocessed {idx}/{len(files)}")

        payload = preprocess_spectrum_for_audit(
            path,
            profile,
            cfg=cfg,
            wn_ref=wn_ref,
        )
        reason = payload.get("skip_reason", "")
        if reason:
            skipped.append((path, reason))
            continue

        spectra.append(np.asarray(payload["z"], dtype=np.float32))
        rel_paths.append(Path(path).resolve().relative_to(input_root))

    if not spectra:
        return None, np.empty((0, 0), dtype=np.float32), rel_paths, skipped

    return payload["wn"], np.asarray(spectra, dtype=np.float32), rel_paths, skipped


def _smooth_spectra(spectra, smooth_points):
    return np.asarray(
        [moving_average(row, smooth_points) for row in spectra],
        dtype=np.float32,
    )


def _window_depths(spectra, start, end, side_points):
    left = spectra[:, start - side_points : start].mean(axis=1)
    right = spectra[:, end : end + side_points].mean(axis=1)
    center = spectra[:, start:end].mean(axis=1)
    # 两侧都高于窗口内部时，才视作局部下凹；这样可减少单侧斜坡误判。
    return np.minimum(left - center, right - center)


def _build_candidate(wn, start, end, depths, cfg):
    flagged = depths > cfg.sample_depth_threshold
    fraction = float(flagged.mean())
    median_depth = float(np.quantile(depths, 0.50))
    q80_depth = float(np.quantile(depths, 0.80))
    mean_depth = float(np.mean(depths))
    score = fraction * max(median_depth, 0.0) * max(q80_depth, 0.0)
    return BandCandidate(
        score=float(score),
        start_idx=int(start),
        end_idx=int(end),
        band_min=float(wn[start]),
        band_max=float(wn[end - 1]),
        width_cm=region_width_cm(wn, start, end),
        sample_fraction=fraction,
        median_depth=median_depth,
        q80_depth=q80_depth,
        mean_depth=mean_depth,
    )


def _scan_candidates(wn, spectra, cfg):
    side_points = _points_for_cm(wn, cfg.side_cm, minimum=5)
    stride_points = _points_for_cm(wn, cfg.stride_cm, minimum=1)
    max_step = _step_cm(wn) * 1.8
    candidates = []

    width_values = np.arange(
        cfg.min_width_cm,
        cfg.max_width_cm + cfg.width_step_cm * 0.5,
        cfg.width_step_cm,
    )
    for width_cm in width_values:
        width_points = _points_for_cm(wn, width_cm, minimum=5)
        last_start = spectra.shape[1] - width_points - side_points
        for start in range(side_points, max(side_points, last_start), stride_points):
            end = start + width_points
            window_wn = wn[start - side_points : end + side_points]
            if window_wn.size >= 2 and np.max(np.abs(np.diff(window_wn))) > max_step:
                continue
            depths = _window_depths(spectra, start, end, side_points)
            candidate = _build_candidate(wn, start, end, depths, cfg)
            if (
                candidate.sample_fraction >= cfg.min_fraction
                and candidate.median_depth >= cfg.min_median_depth
                and candidate.q80_depth >= cfg.min_q80_depth
            ):
                candidates.append(candidate)

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _overlap_ratio(a, b):
    overlap = max(0, min(a.end_idx, b.end_idx) - max(a.start_idx, b.start_idx))
    if overlap <= 0:
        return 0.0
    shorter = min(a.end_idx - a.start_idx, b.end_idx - b.start_idx)
    return float(overlap) / float(max(shorter, 1))


def _select_candidates(candidates, cfg):
    selected = []
    for candidate in candidates:
        if any(_overlap_ratio(candidate, old) >= cfg.overlap_ratio for old in selected):
            continue
        selected.append(candidate)
        if len(selected) >= cfg.max_candidates:
            break
    return selected


def _group_key(rel_path):
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return str(Path(parts[0]) / parts[1])
    if len(parts) == 1:
        return "."
    return "."


def _class_key(rel_path):
    parts = Path(rel_path).parts
    return parts[0] if parts else "."


def _fraction_text(values):
    pieces = []
    for name in sorted(values):
        total, flagged = values[name]
        frac = flagged / total if total else 0.0
        pieces.append(f"{name}:{flagged}/{total}={frac:.3f}")
    return "; ".join(pieces)


def _class_fractions(rel_paths, flags):
    by_class = {}
    for rel, flag in zip(rel_paths, flags):
        key = _class_key(rel)
        total, flagged = by_class.get(key, (0, 0))
        by_class[key] = (total + 1, flagged + int(bool(flag)))
    return by_class


def _folder_rows(rank, candidate, depths, rel_paths, cfg):
    flags = depths > cfg.sample_depth_threshold
    stats = {}
    for rel, flag, depth in zip(rel_paths, flags, depths):
        key = _group_key(rel)
        item = stats.setdefault(key, {"total": 0, "flagged": 0, "depths": []})
        item["total"] += 1
        item["flagged"] += int(bool(flag))
        item["depths"].append(float(depth))

    rows = []
    for folder, item in stats.items():
        total = item["total"]
        flagged = item["flagged"]
        if flagged <= 0:
            continue
        rows.append(
            {
                "rank": rank,
                "band_min": f"{candidate.band_min:.3f}",
                "band_max": f"{candidate.band_max:.3f}",
                "folder": folder,
                "flagged": flagged,
                "total": total,
                "fraction": f"{flagged / total:.6f}",
                "mean_depth": f"{np.mean(item['depths']):.6f}",
            }
        )
    rows.sort(key=lambda row: (float(row["fraction"]), int(row["flagged"])), reverse=True)
    return rows


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(out_path, wn, spectra, selected, cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    median = np.median(spectra, axis=0)
    q10 = np.quantile(spectra, 0.10, axis=0)
    q90 = np.quantile(spectra, 0.90, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    ax = axes[0]
    ax.fill_between(wn, q10, q90, color="#9ecae1", alpha=0.35, label="q10-q90")
    ax.plot(wn, median, color="#08519c", linewidth=1.2, label="median SNV")
    for rank, item in enumerate(selected, start=1):
        ax.axvspan(item.band_min, item.band_max, alpha=0.18, label=f"#{rank}")
    ax.set_ylabel("SNV")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for rank, item in enumerate(selected, start=1):
        depths = _window_depths(
            spectra,
            item.start_idx,
            item.end_idx,
            _points_for_cm(wn, cfg.side_cm, minimum=5),
        )
        flags = depths > cfg.sample_depth_threshold
        ax.bar(
            [rank],
            [item.sample_fraction],
            color="#31a354" if rank == 1 else "#756bb1",
        )
        ax.text(
            rank,
            item.sample_fraction + 0.02,
            f"{item.band_min:.0f}-{item.band_max:.0f}\n{flags.sum()}/{flags.size}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("flagged fraction")
    ax.set_xlabel("candidate rank")
    ax.set_xticks(range(1, len(selected) + 1))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(path, dataset, input_root, out_dir, total_files, valid_count, skipped, selected, cfg, use_config_bad_bands):
    lines = [
        f"# {dataset} 下凹坏段扫描报告",
        "",
        "## 总体",
        "",
        f"- 输入目录：`{input_root}`",
        f"- 输出目录：`{out_dir}`",
        f"- 总光谱数：{total_files}",
        f"- 有效光谱数：{valid_count}",
        f"- 跳过光谱数：{len(skipped)}",
        f"- 预处理是否沿用当前 bad_bands 遮罩：{bool(use_config_bad_bands)}",
        f"- 单谱下凹阈值：{cfg.sample_depth_threshold:.3f}",
        f"- 候选筛选：覆盖率 >= {cfg.min_fraction:.2f}, 中位深度 >= {cfg.min_median_depth:.2f}, q80 >= {cfg.min_q80_depth:.2f}",
        "",
        "## 候选波段",
        "",
    ]
    if not selected:
        lines.append("- 未发现满足阈值的系统性下凹候选波段")
    for rank, item in enumerate(selected, start=1):
        lines.append(
            f"- #{rank}: `{item.band_min:.1f}-{item.band_max:.1f} cm^-1`, "
            f"score={item.score:.3f}, 覆盖率={item.sample_fraction:.3f}, "
            f"中位深度={item.median_depth:.3f}, q80={item.q80_depth:.3f}"
        )
    lines.extend(
        [
            "",
            "## 判读说明",
            "",
            "- 分数只用于排序，最终是否加入离线 `bad_bands` 仍建议结合复核图确认。",
            "- 默认扫描时不遮罩已有坏段，目的是允许脚本重新发现已经怀疑的区域。",
            "- 下凹深度在 SNV 后计算，表示候选窗口相对左右肩部的整体负向偏离。",
        ]
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_bad_band_scan(
    dataset,
    *,
    subdir=None,
    folder=None,
    output_root=None,
    timestamp=None,
    max_files=0,
    seed=42,
    no_plot=False,
    use_config_bad_bands=False,
    scan_config=None,
):
    profile, dataset_dir = resolve_dataset(dataset)
    input_root, rel_base = resolve_audit_input(dataset_dir, profile, subdir=subdir, folder=folder)
    input_root = input_root.resolve()

    files = sorted(input_root.rglob("*.arc_data"))
    total_files = len(files)
    files = _sample_files(files, max_files=max_files, seed=seed)
    if not files:
        raise FileNotFoundError(f"No .arc_data files found under {input_root}")

    cfg = DEFAULT_PIPELINE_CONFIG if use_config_bad_bands else replace(DEFAULT_PIPELINE_CONFIG, bad_bands=())
    scan_cfg = scan_config or BadBandScanConfig()

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_root) if output_root is not None else dataset_dir / "audit_bad_band"
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bad-band] input={input_root}")
    print(f"[bad-band] files={len(files)} sampled_from={total_files}")
    wn, spectra, rel_paths, skipped = _preprocess_files(files, profile, cfg, input_root)
    if spectra.size == 0:
        raise RuntimeError("No valid spectra after preprocessing")

    smoothed = _smooth_spectra(spectra, scan_cfg.smooth_points)
    candidates = _scan_candidates(wn, smoothed, scan_cfg)
    selected = _select_candidates(candidates, scan_cfg)

    band_rows = []
    folder_rows = []
    side_points = _points_for_cm(wn, scan_cfg.side_cm, minimum=5)
    for rank, item in enumerate(selected, start=1):
        depths = _window_depths(smoothed, item.start_idx, item.end_idx, side_points)
        flags = depths > scan_cfg.sample_depth_threshold
        class_fractions = _class_fractions(rel_paths, flags)
        band_rows.append(
            {
                "rank": rank,
                "band_min": f"{item.band_min:.3f}",
                "band_max": f"{item.band_max:.3f}",
                "width_cm": f"{item.width_cm:.3f}",
                "score": f"{item.score:.6f}",
                "sample_fraction": f"{item.sample_fraction:.6f}",
                "median_depth": f"{item.median_depth:.6f}",
                "q80_depth": f"{item.q80_depth:.6f}",
                "mean_depth": f"{item.mean_depth:.6f}",
                "flagged": int(flags.sum()),
                "total": int(flags.size),
                "class_fraction": _fraction_text(class_fractions),
            }
        )
        folder_rows.extend(_folder_rows(rank, item, depths, rel_paths, scan_cfg))

    _write_csv(
        out_dir / "candidate_bands.csv",
        band_rows,
        [
            "rank",
            "band_min",
            "band_max",
            "width_cm",
            "score",
            "sample_fraction",
            "median_depth",
            "q80_depth",
            "mean_depth",
            "flagged",
            "total",
            "class_fraction",
        ],
    )
    _write_csv(
        out_dir / "candidate_folder_summary.csv",
        folder_rows,
        ["rank", "band_min", "band_max", "folder", "flagged", "total", "fraction", "mean_depth"],
    )
    with open(out_dir / "scan_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": dataset,
                "input_root": str(input_root),
                "rel_base": str(rel_base),
                "total_files": total_files,
                "scanned_files": len(files),
                "valid_files": int(spectra.shape[0]),
                "skipped_files": len(skipped),
                "use_config_bad_bands": bool(use_config_bad_bands),
                "scan_config": asdict(scan_cfg),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    _write_summary(
        out_dir / "summary.md",
        dataset,
        input_root,
        out_dir,
        total_files,
        int(spectra.shape[0]),
        skipped,
        selected,
        scan_cfg,
        use_config_bad_bands,
    )
    if not no_plot and selected:
        _write_plot(out_dir / "dip_candidates.png", wn, smoothed, selected, scan_cfg)

    if selected:
        best = selected[0]
        print(
            "[bad-band] top candidate: "
            f"{best.band_min:.1f}-{best.band_max:.1f} cm^-1 "
            f"fraction={best.sample_fraction:.3f} score={best.score:.3f}"
        )
    else:
        print("[bad-band] no candidate bands passed thresholds")
    print(f"[bad-band] output={out_dir}")
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description="扫描系统性下凹坏段")
    parser.add_argument("dataset", help="数据集名称或 profile id，例如 MN_IgA / 细菌")
    parser.add_argument("--subdir", default=None, help="输入阶段目录，默认 init")
    parser.add_argument("--folder", default=None, help="只扫描某个文件夹，支持 属名/文件夹名")
    parser.add_argument("--output-root", default=None, help="输出根目录，默认 dataset/<数据集>/audit_bad_band")
    parser.add_argument("--timestamp", default=None, help="固定输出目录名")
    parser.add_argument("--max-files", type=int, default=0, help="抽样扫描数量；0 表示全量")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--no-plot", action="store_true", help="不输出复核图")
    parser.add_argument(
        "--use-config-bad-bands",
        action="store_true",
        help="沿用当前离线配置中的 bad_bands 遮罩；默认不遮罩，便于发现坏段",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_bad_band_scan(
        args.dataset,
        subdir=args.subdir,
        folder=args.folder,
        output_root=args.output_root,
        timestamp=args.timestamp,
        max_files=args.max_files,
        seed=args.seed,
        no_plot=args.no_plot,
        use_config_bad_bands=args.use_config_bad_bands,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
