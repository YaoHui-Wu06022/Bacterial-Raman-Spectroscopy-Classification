from pathlib import Path
import random
import sys

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _find_project_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "raman").is_dir() and (path / "dataset").is_dir():
            return path
    return current


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman.data.build import COMMON_BAD_BANDS, DEFAULT_PIPELINE_CONFIG
from raman.data.offline import (
    _median_filter_1d,
    _odd_window_points,
    _residual_z_score,
    estimate_baseline,
    remove_cosmic_rays,
)
from raman.data.profiles import get_dataset_dir, get_profile
from raman.data.spectrum import build_valid_mask, read_arc_data


CHINESE_FONT_FILES = (
    ("Microsoft YaHei", Path("C:/Windows/Fonts/msyh.ttc")),
    ("SimHei", Path("C:/Windows/Fonts/simhei.ttf")),
    ("SimSun", Path("C:/Windows/Fonts/simsun.ttc")),
    ("DengXian", Path("C:/Windows/Fonts/Deng.ttf")),
)


def _configure_plot_style():
    """配置绘图风格并自动注册本机中文字体"""
    for _, font_path in CHINESE_FONT_FILES:
        if font_path.is_file():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except RuntimeError:
                pass

    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    chinese_font = next((name for name, _ in CHINESE_FONT_FILES if name in available_fonts), None)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (11, 5)
    if chinese_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chinese_font, "DejaVu Sans"]
    else:
        print("未找到可用中文字体，建议安装 Microsoft YaHei 或 SimHei")


_configure_plot_style()


def _resolve_sample_path(project_root, dataset_dir, profile, sample_path, sample_folder, sample_seed):
    if sample_path:
        path = Path(sample_path)
        return path if path.is_absolute() else project_root / path

    root = Path(sample_folder) if sample_folder else Path(profile.root_init)
    if not root.is_absolute():
        project_path = project_root / root
        root = project_path if project_path.exists() or root.parts[:1] == ("dataset",) else dataset_dir / root

    matches = sorted(root.rglob("*.arc_data"))
    if not matches:
        raise FileNotFoundError(f"未找到可展示的 .arc_data 文件：{root}")
    return random.Random(sample_seed).choice(matches)


def _add_bad_band_spans(ax, bad_bands):
    for idx, (band_min, band_max) in enumerate(bad_bands):
        ax.axvspan(
            band_min,
            band_max,
            color="gray",
            alpha=0.15,
            label="bad bands" if idx == 0 else None,
        )


def _plot_without_bad_bands(ax, wn, y, bad_bands, **kwargs):
    wn = np.asarray(wn, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    keep_mask = build_valid_mask(wn, bad_bands)
    if keep_mask is None:
        keep_mask = np.ones_like(wn, dtype=bool)
    if "color" not in kwargs:
        kwargs["color"] = ax._get_lines.get_next_color()

    start = None
    for idx, keep in enumerate(keep_mask):
        if keep and start is None:
            start = idx
        elif (not keep) and start is not None:
            ax.plot(wn[start:idx], y[start:idx], **kwargs)
            kwargs.pop("label", None)
            start = None
    if start is not None:
        ax.plot(wn[start:], y[start:], **kwargs)


def _style_axis(ax, title, bad_bands, display_min=None, display_max=None, ylabel="Intensity"):
    _add_bad_band_spans(ax, bad_bands)
    if display_min is not None and display_max is not None:
        ax.set_xlim(display_min, display_max)
    ax.set_title(title)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def _run_cosmic_debug(raw, wn, cfg, bad_bands):
    valid_mask = np.ones_like(wn, dtype=bool)
    narrow_clean = np.asarray(raw, dtype=np.float32).copy()
    narrow_mask = np.zeros_like(narrow_clean, dtype=bool)
    window = _odd_window_points(cfg.cosmic_ray_narrow_window_points)

    for _ in range(int(cfg.cosmic_ray_max_iter)):
        local_median = _median_filter_1d(narrow_clean, window)
        residual = narrow_clean - local_median
        z_score = _residual_z_score(residual, valid_mask)
        if z_score is None:
            break
        spike_mask = valid_mask & (z_score > float(cfg.cosmic_ray_threshold))
        if not spike_mask.any():
            break
        narrow_mask |= spike_mask
        narrow_clean[spike_mask] = local_median[spike_mask]

    final_clean, stats = remove_cosmic_rays(
        raw,
        window_points=cfg.cosmic_ray_narrow_window_points,
        threshold=cfg.cosmic_ray_threshold,
        max_iter=cfg.cosmic_ray_max_iter,
        valid_mask=valid_mask,
        peak_prominence_z=cfg.cosmic_ray_peak_prominence_z,
        peak_window_points=cfg.cosmic_ray_peak_window_points,
        peak_expand_z=cfg.cosmic_ray_peak_expand_z,
        peak_expand_gap_points=cfg.cosmic_ray_peak_expand_gap_points,
        peak_width_max_points=cfg.cosmic_ray_peak_width_max_points,
        peak_mean_z_min=cfg.cosmic_ray_peak_mean_z_min,
        peak_pad_points=cfg.cosmic_ray_peak_pad_points,
    )

    peak_mask = np.abs(final_clean - narrow_clean) > 1e-6
    peak_window = _odd_window_points(cfg.cosmic_ray_peak_window_points)
    peak_local_median = _median_filter_1d(narrow_clean, peak_window)
    peak_residual = narrow_clean - peak_local_median
    peak_residual_z = _residual_z_score(peak_residual, valid_mask)
    if peak_residual_z is None:
        peak_residual_z = np.full(raw.shape, np.nan, dtype=np.float32)

    keep_mask = build_valid_mask(wn, bad_bands)
    if keep_mask is None:
        keep_mask = np.ones_like(wn, dtype=bool)

    segments = []
    visible = np.flatnonzero(peak_mask & valid_mask & keep_mask)
    if visible.size:
        for part in np.split(visible, np.where(np.diff(visible) > 1)[0] + 1):
            start = int(part[0])
            end = int(part[-1]) + 1
            z_segment = peak_residual_z[start:end]
            before = narrow_clean[start:end]
            after = final_clean[start:end]
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "wn_min": float(wn[start]),
                    "wn_max": float(wn[end - 1]),
                    "width_points": int(end - start),
                    "max_z": float(np.nanmax(z_segment)) if z_segment.size else np.nan,
                    "mean_z": float(np.nanmean(np.maximum(z_segment, 0.0))) if z_segment.size else np.nan,
                    "delta_mean": float(np.mean(before - after)) if before.size else 0.0,
                }
            )

    return {
        "raw": np.asarray(raw, dtype=np.float32),
        "narrow_clean": narrow_clean,
        "narrow_mask": narrow_mask,
        "final_clean": final_clean,
        "final_stats": stats,
        "peak_mask": peak_mask,
        "peak_residual_z": peak_residual_z,
        "peak_segments": segments,
    }


def show_preprocess(
    sample_path=None,
    *,
    dataset_name="细菌",
    sample_folder=None,
    sample_seed=None,
    baseline_method=None,
    compare_baselines=True,
    baseline_compare_methods=("airpls", "asls"),
    display_min=None,
    display_max=None,
    max_peak_segments=6,
    bad_bands=COMMON_BAD_BANDS,
):
    cfg = DEFAULT_PIPELINE_CONFIG
    profile = get_profile(dataset_name)
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    method = (baseline_method or cfg.baseline_method).lower()

    resolved_path = _resolve_sample_path(
        PROJECT_ROOT,
        dataset_dir,
        profile,
        sample_path,
        sample_folder,
        sample_seed,
    )
    wn, sp = read_arc_data(resolved_path)
    if wn.size == 0 or sp.size == 0:
        raise ValueError(f"读取失败：{resolved_path}")

    bad_keep_mask = build_valid_mask(wn, bad_bands)
    if bad_keep_mask is None:
        bad_keep_mask = np.ones_like(wn, dtype=bool)

    cosmic = _run_cosmic_debug(sp, wn, cfg, bad_bands)
    baseline_fit_mask = (wn >= cfg.baseline_fit_min) & (wn <= cfg.baseline_fit_max)
    wn_fit = wn[baseline_fit_mask]
    sp_cosmic_fit = cosmic["final_clean"][baseline_fit_mask]
    valid_fit_mask = build_valid_mask(wn_fit, bad_bands)
    if valid_fit_mask is None:
        valid_fit_mask = np.ones_like(wn_fit, dtype=bool)
    if wn_fit.size < 10:
        raise ValueError("baseline 拟合窗口内有效点过少")

    baseline = estimate_baseline(
        sp_cosmic_fit,
        method=method,
        lam=cfg.baseline_lam,
        p=cfg.baseline_asls_p,
        niter=cfg.baseline_max_iter,
        valid_mask=valid_fit_mask,
    )
    sp_corrected = sp_cosmic_fit - baseline

    print(f"sample_path = {resolved_path}")
    print(
        "cosmic replaced: "
        f"narrow={cosmic['final_stats'].narrow}, "
        f"peak={cosmic['final_stats'].peak}, "
        f"total={cosmic['final_stats'].total}"
    )
    print(f"baseline method = {method}")

    raw_plot = cosmic["raw"].copy()
    narrow_plot = cosmic["narrow_clean"].copy()
    final_plot = cosmic["final_clean"].copy()
    raw_plot[~bad_keep_mask] = np.nan
    narrow_plot[~bad_keep_mask] = np.nan
    final_plot[~bad_keep_mask] = np.nan
    visible_peak_mask = cosmic["peak_mask"] & bad_keep_mask

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(wn, raw_plot, label="原始谱", alpha=0.45, linewidth=1.0)
    axes[0].plot(wn, narrow_plot, label="narrow 清理后", alpha=0.95, linewidth=1.2)
    _style_axis(
        axes[0],
        f"1. 原始谱 / narrow 清理后 | narrow={int(cosmic['final_stats'].narrow)} 点",
        bad_bands,
        display_min,
        display_max,
    )

    axes[1].plot(wn, narrow_plot, label="narrow 清理后", alpha=0.65, linewidth=1.0)
    axes[1].plot(wn, final_plot, label="narrow + peak 清理后", alpha=0.95, linewidth=1.2)
    if visible_peak_mask.any():
        axes[1].scatter(
            wn[visible_peak_mask],
            cosmic["narrow_clean"][visible_peak_mask],
            s=24,
            color="darkorange",
            label="peak 替换前点位",
            zorder=5,
        )
    else:
        axes[1].text(0.02, 0.92, "当前样本没有可见的 peak 替换点", transform=axes[1].transAxes)
    _style_axis(
        axes[1],
        f"2. narrow 清理后 / narrow + peak 清理后 | peak={int(cosmic['peak_mask'].sum())} 点",
        bad_bands,
        display_min,
        display_max,
    )
    fig.tight_layout()
    plt.show()

    _plot_peak_debug(wn, cosmic, bad_keep_mask, cfg, bad_bands, display_min, display_max, max_peak_segments)
    _plot_baseline(
        wn_fit,
        sp_cosmic_fit,
        baseline,
        sp_corrected,
        method,
        bad_bands,
        display_min,
        display_max,
        compare_baselines,
        baseline_compare_methods,
        cfg,
        valid_fit_mask,
    )


def _plot_peak_debug(wn, cosmic, bad_keep_mask, cfg, bad_bands, display_min, display_max, max_peak_segments):
    z_plot = cosmic["peak_residual_z"].copy()
    z_plot[~bad_keep_mask] = np.nan
    visible_peak_mask = cosmic["peak_mask"] & bad_keep_mask

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(wn, z_plot, label="peak 残差 z", color="tab:blue", linewidth=1.1)
    ax.axhline(
        cfg.cosmic_ray_peak_prominence_z,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
        label="高阈值核心 z",
    )
    ax.axhline(
        cfg.cosmic_ray_peak_expand_z,
        color="crimson",
        linestyle=":",
        linewidth=1.2,
        label="低阈值扩展 z",
    )
    if visible_peak_mask.any():
        ax.scatter(
            wn[visible_peak_mask],
            cosmic["peak_residual_z"][visible_peak_mask],
            s=24,
            color="darkorange",
            label="peak 替换点",
            zorder=5,
        )
    else:
        ax.text(0.02, 0.92, "当前样本没有可见的 peak 替换点", transform=ax.transAxes)
    _style_axis(ax, "3. peak 局部 median 正残差 z-score", bad_bands, display_min, display_max, "残差 z-score")
    plt.show()

    segments = cosmic["peak_segments"][:max_peak_segments]
    if not segments:
        print("当前样本没有可见的 peak 阶段替换区域可放大")
        return

    fig, axes = plt.subplots(len(segments), 1, figsize=(13, 3.2 * len(segments)), squeeze=False)
    axes = axes[:, 0]
    for idx, (ax, segment) in enumerate(zip(axes, segments), start=1):
        start = segment["start"]
        end = segment["end"]
        wn_min = segment["wn_min"]
        wn_max = segment["wn_max"]
        pad = max((wn_max - wn_min) * 4, 20.0)
        local = (wn >= wn_min - pad) & (wn <= wn_max + pad)
        local_keep = build_valid_mask(wn[local], bad_bands)
        if local_keep is None:
            local_keep = np.ones_like(wn[local], dtype=bool)
        before_plot = cosmic["narrow_clean"][local].copy()
        after_plot = cosmic["final_clean"][local].copy()
        before_plot[~local_keep] = np.nan
        after_plot[~local_keep] = np.nan

        ax.plot(wn[local], before_plot, label="peak 前（narrow 后）", alpha=0.75, linewidth=1.1)
        ax.plot(wn[local], after_plot, label="peak 后", alpha=0.95, linewidth=1.2)
        ax.scatter(
            wn[start:end],
            cosmic["narrow_clean"][start:end],
            s=24,
            color="darkorange",
            label="peak 替换前点位",
            zorder=5,
        )
        ax.axvspan(wn_min, wn_max, color="darkorange", alpha=0.16)
        _add_bad_band_spans(ax, bad_bands)
        ax.set_title(
            f"peak {idx}: {wn_min:.2f}-{wn_max:.2f} cm$^{{-1}}$ | "
            f"点数={segment['width_points']}, max_z={segment['max_z']:.2f}, mean_z={segment['mean_z']:.2f}"
        )
        ax.set_ylabel("Intensity")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    plt.show()


def _plot_baseline(
    wn_fit,
    sp_cosmic_fit,
    baseline,
    sp_corrected,
    method,
    bad_bands,
    display_min,
    display_max,
    compare_baselines,
    baseline_compare_methods,
    cfg,
    valid_fit_mask,
):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    _plot_without_bad_bands(axes[0], wn_fit, sp_cosmic_fit, bad_bands, label="after cosmic removal", alpha=0.85, linewidth=1.1)
    _plot_without_bad_bands(axes[0], wn_fit, baseline, bad_bands, label=f"{method} baseline", alpha=0.95, linewidth=2.0)
    _style_axis(axes[0], f"4. {method} 基线估计", bad_bands, display_min, display_max)

    _plot_without_bad_bands(axes[1], wn_fit, sp_corrected, bad_bands, label="baseline corrected", alpha=0.95, linewidth=1.1)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    _style_axis(axes[1], "5. 基线扣除后", bad_bands, display_min, display_max, "Corrected intensity")
    fig.tight_layout()
    plt.show()

    if not compare_baselines:
        print("compare_baselines=False，跳过方法对比")
        return

    baselines = {}
    corrected = {}
    for item in baseline_compare_methods:
        baselines[item] = estimate_baseline(
            sp_cosmic_fit,
            method=item,
            lam=cfg.baseline_lam,
            p=cfg.baseline_asls_p,
            niter=cfg.baseline_max_iter,
            valid_mask=valid_fit_mask,
        )
        corrected[item] = sp_cosmic_fit - baselines[item]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    _plot_without_bad_bands(axes[0], wn_fit, sp_cosmic_fit, bad_bands, label="after cosmic removal", color="0.65", linewidth=1.0)
    for item in baseline_compare_methods:
        _plot_without_bad_bands(axes[0], wn_fit, baselines[item], bad_bands, label=f"{item} baseline", linewidth=1.8)
    _style_axis(axes[0], "6. 不同 baseline 方法的基线形态", bad_bands, display_min, display_max)

    for item in baseline_compare_methods:
        _plot_without_bad_bands(axes[1], wn_fit, corrected[item], bad_bands, label=f"{item} corrected", linewidth=1.0)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    _style_axis(axes[1], "7. 不同 baseline 方法的扣除结果", bad_bands, display_min, display_max, "Corrected intensity")
    fig.tight_layout()
    plt.show()
