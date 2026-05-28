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

from raman.data.build import COMMON_BAD_BANDS, DEFAULT_PIPELINE_CONFIG, _cosmic_ray_kwargs
from raman.data.preprocess import CosmicRayStats, estimate_baseline, remove_cosmic_rays
from raman.data.profiles import get_dataset_dir, get_profile
from raman.data.io import read_arc_data
from raman.tool.spectrum import build_valid_mask


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


def _run_cosmic_debug(raw, options):
    if not bool(options.get("cosmic_ray_remove")):
        raw = np.asarray(raw, dtype=np.float32)
        return {
            "enabled": False,
            "raw": raw,
            "cosmic_ray_mask": np.zeros_like(raw, dtype=bool),
            "final_clean": raw.copy(),
            "final_stats": CosmicRayStats(),
        }

    final_clean, stats = remove_cosmic_rays(
        raw,
        window_points=options["cosmic_ray_window_points"],
        threshold=options["cosmic_ray_threshold"],
        max_iter=options["cosmic_ray_max_iter"],
        valid_mask=None,
    )

    return {
        "enabled": True,
        "raw": np.asarray(raw, dtype=np.float32),
        "cosmic_ray_mask": np.abs(final_clean - raw) > 1e-6,
        "final_clean": final_clean,
        "final_stats": stats,
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

    label_display = resolved_path.relative_to(dataset_dir).as_posix() if resolved_path.is_relative_to(dataset_dir) else None
    cosmic_options = _cosmic_ray_kwargs(profile, cfg, label_display)
    cosmic = _run_cosmic_debug(sp, cosmic_options)
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
    if cosmic["enabled"]:
        print(f"cosmic_ray replaced: {int(cosmic['final_stats'])}")
    else:
        print("cosmic_ray removal disabled")
    print(f"baseline method = {method}")

    if cosmic["enabled"]:
        raw_plot = cosmic["raw"].copy()
        clean_plot = cosmic["final_clean"].copy()
        delta_plot = cosmic["raw"] - cosmic["final_clean"]
        raw_plot[~bad_keep_mask] = np.nan
        clean_plot[~bad_keep_mask] = np.nan
        delta_plot[~bad_keep_mask] = np.nan
        visible_cosmic_ray_mask = cosmic["cosmic_ray_mask"] & bad_keep_mask

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
        axes[0].plot(wn, raw_plot, label="原始谱", alpha=0.45, linewidth=1.0)
        axes[0].plot(wn, clean_plot, label="宇宙射线清理后", alpha=0.95, linewidth=1.2)
        _style_axis(
            axes[0],
            f"原始谱 / 宇宙射线清理后 | 替换={int(cosmic['final_stats'])} 点",
            bad_bands,
            display_min,
            display_max,
        )

        axes[1].plot(wn, delta_plot, label="宇宙射线替换差值", alpha=0.95, linewidth=1.1)
        if visible_cosmic_ray_mask.any():
            axes[1].scatter(
                wn[visible_cosmic_ray_mask],
                delta_plot[visible_cosmic_ray_mask],
                s=24,
                color="darkorange",
                label="宇宙射线替换点",
                zorder=5,
            )
        else:
            axes[1].text(0.02, 0.92, "当前样本没有宇宙射线替换点", transform=axes[1].transAxes)
        _style_axis(
            axes[1],
            "宇宙射线替换差值",
            bad_bands,
            display_min,
            display_max,
            "Delta",
        )
        fig.tight_layout()
        plt.show()

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
    _style_axis(axes[0], f"{method} 基线估计", bad_bands, display_min, display_max)

    _plot_without_bad_bands(axes[1], wn_fit, sp_corrected, bad_bands, label="baseline corrected", alpha=0.95, linewidth=1.1)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    _style_axis(axes[1], "基线扣除后", bad_bands, display_min, display_max, "Corrected intensity")
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
    _style_axis(axes[0], "不同 baseline 方法的基线形态", bad_bands, display_min, display_max)

    for item in baseline_compare_methods:
        _plot_without_bad_bands(axes[1], wn_fit, corrected[item], bad_bands, label=f"{item} corrected", linewidth=1.0)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    _style_axis(axes[1], "不同 baseline 方法的扣除结果", bad_bands, display_min, display_max, "Corrected intensity")
    fig.tight_layout()
    plt.show()
