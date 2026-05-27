"""给定区间内扫描系统性下凹坏段"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from raman.audit.common import (
    preprocess_spectrum_for_audit,
    write_csv,
)
from raman.data.build import DEFAULT_PIPELINE_CONFIG
from raman.tool.array import moving_average, robust_mad_scale
from raman.tool.dataset import resolve_dataset
from raman.tool.path import PROJECT_ROOT
from raman.tool.spectrum import median_step_cm


@dataclass(frozen=True)
class BadBandScanConfig:
    """区间下凹扫描参数"""

    scan_min: float = 850.0
    scan_max: float = 1000.0
    smooth_points: int = 9
    side_points: int = 15
    min_width_points: int = 8
    max_width_points: int = 60
    width_step_points: int = 4
    stride_points: int = 2
    depth_threshold: float = 0.35


BAD_BAND_MAX_FILES = 0
BAD_BAND_SAMPLE_SEED = 42
BAD_BAND_OUTPUT_DIR = "audit_bad_band"


def _sample_files(files, max_files, seed):
    """按固定随机种子抽样文件"""
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return files
    rng = np.random.RandomState(int(seed))
    indices = rng.choice(len(files), size=int(max_files), replace=False)
    return [files[int(i)] for i in sorted(indices)]


def _resolve_scan_target(target):
    """解析 profile 名、数据集名或 dataset 下的扫描目录"""
    target_text = str(target).strip().strip('"').strip("'")
    try:
        profile, dataset_dir = resolve_dataset(target_text)
        return profile, dataset_dir, (dataset_dir / profile.root_init).resolve()
    except KeyError:
        pass

    path = Path(target_text)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Bad-band scan target not found: {target}")

    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    try:
        rel = path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Bad-band scan folder must be under dataset/: {path}") from exc
    if not rel.parts:
        raise ValueError(f"Bad-band scan folder must include dataset name: {path}")

    profile, dataset_dir = resolve_dataset(rel.parts[0])
    return profile, dataset_dir, path


def _load_spectra(target, max_files=BAD_BAND_MAX_FILES, seed=BAD_BAND_SAMPLE_SEED):
    """读取并预处理待扫描光谱，强制不使用 bad_bands mask"""
    profile, dataset_dir, input_root = _resolve_scan_target(target)
    files_all = sorted(input_root.rglob("*.arc_data"))
    files = _sample_files(files_all, max_files, seed)
    if not files:
        raise FileNotFoundError(f"No .arc_data files found under {input_root}")

    cfg = replace(DEFAULT_PIPELINE_CONFIG, bad_bands=())
    wn_ref = cfg.build_wn_ref()
    spectra = []
    skipped_count = 0
    for idx, path in enumerate(files, start=1):
        if idx % 500 == 0:
            print(f"[bad-band] preprocessed {idx}/{len(files)}")
        payload = preprocess_spectrum_for_audit(path, profile, cfg=cfg, wn_ref=wn_ref)
        reason = payload.get("skip_reason", "")
        if reason:
            skipped_count += 1
            continue
        spectra.append(np.asarray(payload["z"], dtype=np.float32))

    if not spectra:
        raise RuntimeError("No valid spectra after preprocessing")

    return {
        "profile": profile,
        "dataset_dir": dataset_dir,
        "input_root": input_root,
        "total_files": len(files_all),
        "scanned_files": len(files),
        "wn": np.asarray(payload["wn"], dtype=np.float32),
        "spectra": np.asarray(spectra, dtype=np.float32),
        "skipped_count": skipped_count,
    }


def _window_depths(spectra, start, end, side_points):
    """计算窗口相对左右肩部的下凹深度"""
    left = spectra[:, start - side_points : start].mean(axis=1)
    right = spectra[:, end : end + side_points].mean(axis=1)
    center = spectra[:, start:end].mean(axis=1)
    return np.minimum(left, right) - center


def _window_row(wn, start, end, depths, cfg):
    """把一个候选窗口整理成结果行"""
    flags = depths >= cfg.depth_threshold
    width_cm = abs(float(wn[end - 1]) - float(wn[start])) + median_step_cm(wn)
    median_depth = float(np.median(depths))
    q80_depth = float(np.quantile(depths, 0.80))
    fraction = float(np.mean(flags))
    score = fraction * max(median_depth, 0.0) * max(q80_depth, 0.0)
    return {
        "band_min": f"{float(wn[start]):.3f}",
        "band_max": f"{float(wn[end - 1]):.3f}",
        "width_points": int(end - start),
        "width_cm": f"{width_cm:.3f}",
        "score": f"{score:.6f}",
        "flagged": int(np.sum(flags)),
        "total": int(depths.size),
        "fraction": f"{fraction:.6f}",
        "median_depth": f"{median_depth:.6f}",
        "q80_depth": f"{q80_depth:.6f}",
        "mean_depth": f"{float(np.mean(depths)):.6f}",
    }


def _scan_windows(wn, spectra, cfg):
    """在给定区间内扫描下凹核心窗口"""
    scan_idx = np.where((wn >= cfg.scan_min) & (wn <= cfg.scan_max))[0]
    if scan_idx.size < cfg.min_width_points:
        raise ValueError("Scan range has too few points")

    best = None
    first = int(scan_idx[0])
    last = int(scan_idx[-1]) + 1
    min_width = max(2, int(cfg.min_width_points))
    max_width = max(min_width, int(cfg.max_width_points))
    width_step = max(1, int(cfg.width_step_points))
    stride = max(1, int(cfg.stride_points))
    side = max(1, int(cfg.side_points))

    for width in range(min_width, max_width + 1, width_step):
        for start in range(first, last - width + 1, stride):
            end = start + width
            if start - side < 0 or end + side > spectra.shape[1]:
                continue
            depths = _window_depths(spectra, start, end, side)
            row = _window_row(wn, start, end, depths, cfg)
            if best is None or float(row["score"]) > float(best["score"]):
                best = row

    if best is None:
        raise RuntimeError("No candidate windows were generated")
    return best


def _expand_to_fast_edges(wn, spectra, core_row, cfg):
    """从谷底核心扩展到两侧快速下跌和回升边界"""
    scan_idx = np.where((wn >= cfg.scan_min) & (wn <= cfg.scan_max))[0]
    first = int(scan_idx[0])
    last = int(scan_idx[-1]) + 1
    core_start = int(np.argmin(np.abs(wn - float(core_row["band_min"]))))
    core_end = int(np.argmin(np.abs(wn - float(core_row["band_max"])))) + 1
    trace = moving_average(np.median(spectra, axis=0), max(3, int(cfg.smooth_points)))
    valley = core_start + int(np.argmin(trace[core_start:core_end]))
    search = max(int(cfg.side_points) * 2, int(cfg.max_width_points) // 2)
    left_lo = max(first, valley - search)
    right_hi = min(last, valley + search + 1)

    diff = np.diff(trace, prepend=trace[0])
    local_diff = diff[left_lo:right_hi]
    threshold = max(robust_mad_scale(local_diff) * 1.2, float(np.quantile(np.abs(local_diff), 0.60)), 1e-6)

    left = None
    start = None
    for idx in range(left_lo + 1, valley + 1):
        if diff[idx] <= -threshold:
            if start is None:
                start = idx
        elif start is not None:
            left = max(first, start - 1)
            start = None
    if start is not None:
        left = max(first, start - 1)
    if left is None or left >= core_start:
        left = left_lo + int(np.argmax(trace[left_lo : core_start + 1]))

    right = None
    start = None
    for idx in range(valley + 1, right_hi):
        if diff[idx] >= threshold:
            if start is None:
                start = idx
        elif start is not None:
            right = min(last, idx + 1)
            break
    if start is not None and right is None:
        right = min(last, right_hi)
    if right is None or right <= core_end:
        right = core_end - 1 + int(np.argmax(trace[core_end - 1 : right_hi])) + 1

    depths = _window_depths(spectra, left, right, cfg.side_points)
    return _window_row(wn, left, right, depths, cfg)


def _plot_overview(out_path, wn, spectra, cfg, best_row):
    """绘制坏段扫描示意图"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_min = float(best_row["band_min"])
    band_max = float(best_row["band_max"])
    plot_mask = (wn >= cfg.scan_min - 60.0) & (wn <= cfg.scan_max + 60.0)
    median = np.median(spectra, axis=0)
    q10 = np.quantile(spectra, 0.10, axis=0)
    q90 = np.quantile(spectra, 0.90, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].fill_between(wn[plot_mask], q10[plot_mask], q90[plot_mask], color="#9ecae1", alpha=0.35, label="q10-q90")
    axes[0].plot(wn[plot_mask], median[plot_mask], color="#08519c", linewidth=1.2, label="median standardized")
    axes[0].axvspan(cfg.scan_min, cfg.scan_max, color="#fdd0a2", alpha=0.16, label="scan range")
    axes[0].axvspan(band_min, band_max, color="#cb181d", alpha=0.22, label="best bad band")
    axes[0].set_ylabel("Standardized value")
    axes[0].legend(loc="best", fontsize=8)

    start = int(np.argmin(np.abs(wn - band_min)))
    end = int(np.argmin(np.abs(wn - band_max))) + 1
    depths = _window_depths(spectra, start, end, cfg.side_points)
    axes[1].hist(depths, bins=40, color="#756bb1", alpha=0.85)
    axes[1].axvline(cfg.depth_threshold, color="#cb181d", linestyle="--", linewidth=1.2, label="depth threshold")
    axes[1].set_xlabel("dip depth")
    axes[1].set_ylabel("spectra count")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_title(f"flagged={best_row['flagged']}/{best_row['total']} fraction={best_row['fraction']}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary(path, prepared, cfg, best_row):
    """写入最优坏段摘要"""
    lines = [
        "# 坏段区间扫描报告",
        "",
        f"- 输入目录：`{prepared['input_root']}`",
        f"- 总光谱数：{prepared['total_files']}",
        f"- 扫描光谱数：{prepared['scanned_files']}",
        f"- 有效光谱数：{prepared['spectra'].shape[0]}",
        f"- 跳过光谱数：{prepared['skipped_count']}",
        f"- 扫描区间：`{cfg.scan_min:.1f}-{cfg.scan_max:.1f} cm^-1`",
        f"- 离线 bad_bands mask：未使用",
        "",
        "## 最合适坏段",
        "",
        f"- 区间：`{float(best_row['band_min']):.1f}-{float(best_row['band_max']):.1f} cm^-1`",
        f"- 宽度：{best_row['width_points']} 点，{float(best_row['width_cm']):.1f} cm^-1",
        f"- 命中：{best_row['flagged']}/{best_row['total']}，比例 {float(best_row['fraction']):.3f}",
        f"- 中位下凹深度：{float(best_row['median_depth']):.3f}",
        f"- q80 下凹深度：{float(best_row['q80_depth']):.3f}",
        f"- score：{float(best_row['score']):.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bad_band_scan(
    target,
    *,
    no_plot=False,
    scan_config=None,
):
    """运行给定区间的坏段扫描"""
    cfg = scan_config or BadBandScanConfig()
    prepared = _load_spectra(target)
    spectra = np.asarray([moving_average(row, cfg.smooth_points) for row in prepared["spectra"]], dtype=np.float32)
    core = _scan_windows(prepared["wn"], spectra, cfg)
    best = _expand_to_fast_edges(prepared["wn"], spectra, core, cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = prepared["dataset_dir"] / BAD_BAND_OUTPUT_DIR
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "best_bad_band.csv", [best], list(best.keys()))
    _write_summary(out_dir / "summary.md", prepared, cfg, best)
    if not no_plot:
        _plot_overview(out_dir / "bad_band_overview.png", prepared["wn"], spectra, cfg, best)

    print(f"[bad-band] input={prepared['input_root']}")
    print(f"[bad-band] valid={prepared['spectra'].shape[0]} skipped={prepared['skipped_count']}")
    print(
        "[bad-band] best: "
        f"{float(best['band_min']):.1f}-{float(best['band_max']):.1f} cm^-1 "
        f"flagged={best['flagged']}/{best['total']} fraction={float(best['fraction']):.3f}"
    )
    print(f"[bad-band] output={out_dir}")
    return out_dir


def build_parser():
    """构建 bad-band 子命令参数解析器"""
    parser = argparse.ArgumentParser(description="扫描给定区间内最可能的系统性下凹坏段")
    parser.add_argument("target", help="扫描目标：profile id、数据集名，或 dataset/<数据集>/... 下的文件夹")
    parser.add_argument("--no-plot", action="store_true", help="不输出示意图")
    return parser


def main(argv=None):
    """执行 bad-band 命令入口"""
    args = build_parser().parse_args(argv)
    run_bad_band_scan(
        args.target,
        no_plot=args.no_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
