from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.config import config
from raman.data.build import DEFAULT_PIPELINE_CONFIG
from raman.data.input import normalize_spectrum
from raman.data.io import read_arc_data
from raman.data.preprocess import estimate_baseline
from raman.data.profiles import get_dataset_dir, get_profile
from raman.tool.naming import extract_letters_prefix
from raman.tool.path import PROJECT_ROOT
from raman.tool.plotting import GLASBEY_DARK_COLORS, plot_segments_without_bad_bands
from raman.tool.spectrum import build_valid_mask, get_config_bad_bands


TARGET_CM = 1002.0  # 峰位审查图中的参考波数
GRID_STEP = 0.2  # 绘图统一插值轴的步长
PREVIEW_FIGURE_WIDTH = 12  # preview 和 plot-shift 图宽
SNV_OFFSET_SCALE = 0.85  # SNV 曲线间距相对整体波动范围的倍率
SNV_OFFSET_MIN = 2.0  # SNV 曲线最小间距
MINMAX_OFFSET_SCALE = 0.35  # minmax 曲线间距相对整体波动范围的倍率
MINMAX_OFFSET_MIN = 0.38  # minmax 曲线最小间距
DELTA_NAME = "delta.txt"  # 当前训练文件夹累计平移量
DELTA_LOG_NAME = "delta_log.txt"  # 每次训练文件夹平移动作日志
DELTA_CS_NAME = "delta_cs.txt"  # 从测试菌迁入的 t 文件夹平移快照
PREFIX_PLOT_STATE_NAME = "prefix_plot_state.csv"  # 上次 prefix 总览图使用的平移状态
DELTA_FIELDS = ("genus", "folder", "prefix", "delta")  # delta.txt 字段
DELTA_CS_FIELDS = ("source_folder", "target_genus", "target_folder", "delta")  # delta_cs.txt 字段
DELTA_LOG_FIELDS = ("time", "genus", "folder", "prefix", "step_delta", "cumulative_delta", "files_changed", "note")  # delta_log.txt 字段
LEGACY_PLOT_STATE_FIELDS = (*DELTA_FIELDS, "plot_version")  # 兼容旧 prefix 状态文件
TRANSFERRED_FOLDER_SUFFIX = "t"  # 测试菌迁入训练集后的文件夹后缀


@dataclass(frozen=True)
class DatasetPaths:
    """shift 工具需要的一组数据集路径"""

    dataset_dir: Path
    init_dir: Path
    output_dir: Path
    delta_path: Path
    delta_log_path: Path


def resolve_dataset(dataset: str) -> DatasetPaths:
    """优先按 profile 解析数据集，失败后按 dataset/<输入> 解析"""
    try:
        profile = get_profile(dataset)
        dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
        init_dir = dataset_dir / profile.root_init
    except KeyError:
        dataset_dir = (PROJECT_ROOT / "dataset" / dataset).resolve()
        init_dir = dataset_dir / "init"

    output_dir = dataset_dir / "fig_init"
    return DatasetPaths(
        dataset_dir=dataset_dir,
        init_dir=init_dir,
        output_dir=output_dir,
        delta_path=output_dir / DELTA_NAME,
        delta_log_path=output_dir / DELTA_LOG_NAME,
    )


def build_plot_grid() -> np.ndarray:
    """按当前裁剪配置生成绘图波数轴"""
    cut_min = float(getattr(config, "cut_min", 600.0))
    cut_max = float(getattr(config, "cut_max", 1800.0))
    return np.arange(cut_min, cut_max + GRID_STEP * 0.5, GRID_STEP, dtype=np.float32)


def shift_folder_prefix(folder: Path) -> str:
    """提取 shift 分组使用的文件夹前缀，保留紧随字母后的正负号"""
    return extract_letters_prefix(folder.name, keep_sign=True, fallback=folder.name)


def is_transferred_folder(folder: Path) -> bool:
    """判断是否为插入训练集的独立测试来源目录，如 KP06t"""
    return Path(folder).name.lower().endswith(TRANSFERRED_FOLDER_SUFFIX)


def iter_init_folders(init_dir: Path, include_transferred: bool = False) -> list[Path]:
    """列出 init/属/种文件夹 两层结构中的全部种文件夹"""
    if not init_dir.is_dir():
        raise FileNotFoundError(f"Missing init folder: {init_dir}")
    folders: list[Path] = []
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        for folder_dir in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            if not include_transferred and is_transferred_folder(folder_dir):
                continue
            folders.append(folder_dir)
    return folders


def folder_raw_median_curve(
    folder: Path,
    wn_ref: np.ndarray,
    wn_offset: float = 0.0,
) -> tuple[np.ndarray | None, int]:
    """读取小文件夹 raw 强度谱并返回统一轴中位谱"""
    curves: list[np.ndarray] = []
    for path in sorted(folder.glob("*.arc_data")):
        wn, sp = read_arc_data(path)
        if wn.size == 0 or sp.size == 0:
            continue
        order = np.argsort(wn)
        wn = wn[order] + float(wn_offset)
        sp = sp[order]

        curve = np.full_like(wn_ref, np.nan, dtype=np.float32)
        inside = (wn_ref >= wn[0]) & (wn_ref <= wn[-1])
        if np.any(inside):
            curve[inside] = np.interp(wn_ref[inside], wn, sp).astype(np.float32)
        curves.append(curve)

    if not curves:
        return None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.vstack(curves), axis=0), len(curves)


def normalize_preview_curve(corrected: np.ndarray, norm_method: str) -> np.ndarray:
    """按指定方法归一化已扣基线的预览曲线"""
    return normalize_spectrum(corrected, norm_method, preserve_nan=True)


def baseline_correct_preview_curve(curve: np.ndarray, wn_ref: np.ndarray) -> np.ndarray:
    """按离线默认参数给 preview 下图扣基线"""
    curve = np.asarray(curve, dtype=np.float32)
    output = np.full(curve.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(curve)
    if finite.sum() < 10:
        return output

    cfg = DEFAULT_PIPELINE_CONFIG
    fit_mask = finite & (wn_ref >= float(cfg.baseline_fit_min)) & (wn_ref <= float(cfg.baseline_fit_max))
    if fit_mask.sum() < 10:
        fit_mask = finite

    valid_fit_mask = build_valid_mask(wn_ref[fit_mask], cfg.bad_bands)
    baseline = estimate_baseline(
        curve[fit_mask],
        method=cfg.baseline_method,
        lam=cfg.baseline_lam,
        p=cfg.baseline_asls_p,
        niter=cfg.baseline_max_iter,
        valid_mask=valid_fit_mask,
    )
    output[fit_mask] = curve[fit_mask] - baseline
    return output


def folder_median_curves(
    folder: Path,
    wn_ref: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """返回小文件夹 raw、SNV 和 minmax 中位谱"""
    raw_curves: list[np.ndarray] = []
    snv_curves: list[np.ndarray] = []
    minmax_curves: list[np.ndarray] = []
    for path in sorted(folder.glob("*.arc_data")):
        wn, sp = read_arc_data(path)
        if wn.size == 0 or sp.size == 0:
            continue
        order = np.argsort(wn)
        wn = wn[order]
        sp = sp[order]

        curve = np.full_like(wn_ref, np.nan, dtype=np.float32)
        inside = (wn_ref >= wn[0]) & (wn_ref <= wn[-1])
        if not np.any(inside):
            continue
        curve[inside] = np.interp(wn_ref[inside], wn, sp).astype(np.float32)
        raw_curves.append(curve)
        corrected = baseline_correct_preview_curve(curve, wn_ref)
        snv_curves.append(normalize_preview_curve(corrected, "snv"))
        minmax_curves.append(normalize_preview_curve(corrected, "minmax"))

    if not raw_curves:
        return None, None, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_median = np.nanmedian(np.vstack(raw_curves), axis=0)
        snv_median = np.nanmedian(np.vstack(snv_curves), axis=0)
        minmax_median = np.nanmedian(np.vstack(minmax_curves), axis=0)
    return raw_median, snv_median, minmax_median, len(raw_curves)


def parse_delta(value: str | None) -> float:
    """解析累计平移量"""
    if value in {"", None}:
        return 0.0
    return float(str(value).strip())


def format_delta(value: float) -> str:
    """格式化累计平移量"""
    return f"{value:+g}" if abs(value) >= 1e-9 else ""


def read_delta_rows(path: Path) -> list[dict[str, str]]:
    """读取 delta.txt"""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != list(DELTA_FIELDS):
            return []
        return list(reader)


def write_delta_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """写出 delta.txt"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["genus"], row["folder"]))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def append_delta_log(
    path: Path,
    folder: Path,
    step_delta: float,
    cumulative_delta: float,
    files_changed: int,
    note: str = "",
) -> None:
    """追加记录单次平移动作"""
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_LOG_FIELDS, delimiter="\t")
        if need_header:
            writer.writeheader()
        writer.writerow(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "genus": folder.parent.name,
                "folder": folder.name,
                "prefix": shift_folder_prefix(folder),
                "step_delta": format_delta(step_delta),
                "cumulative_delta": format_delta(cumulative_delta),
                "files_changed": str(files_changed),
                "note": note,
            }
        )


def ensure_delta_rows(paths: DatasetPaths) -> list[dict[str, str]]:
    """读取 delta.txt，不存在时创建空记录"""
    rows = read_delta_rows(paths.delta_path)
    if rows or paths.delta_path.is_file():
        return rows
    write_delta_rows(paths.delta_path, [])
    return []


def delta_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    """构建小文件夹到累计平移量的映射"""
    return {(row["genus"], row["folder"]): parse_delta(row.get("delta")) for row in rows}


def upsert_delta(rows: list[dict[str, str]], folder: Path, cumulative_delta: float) -> list[dict[str, str]]:
    """更新单个小文件夹的累计平移量"""
    key = (folder.parent.name, folder.name)
    kept = [row for row in rows if (row["genus"], row["folder"]) != key]
    delta = format_delta(cumulative_delta)
    if delta:
        kept.append(
            {
                "genus": folder.parent.name,
                "folder": folder.name,
                "prefix": shift_folder_prefix(folder),
                "delta": delta,
            }
        )
    return kept


def resolve_folder(init_dir: Path, folder_arg: str) -> Path:
    """解析 属/种文件夹 或唯一的种文件夹名"""
    folder_arg = folder_arg.strip().strip('"').strip("'").replace("\\", "/")
    direct = (init_dir / folder_arg).resolve()
    init_resolved = init_dir.resolve()
    try:
        direct.relative_to(init_resolved)
    except ValueError as exc:
        raise ValueError(f"Folder escapes init root: {folder_arg}") from exc
    if direct.is_dir():
        return direct

    matches = [path.resolve() for path in init_dir.glob(f"*/{folder_arg}") if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Cannot find folder under init: {folder_arg}")
    choices = "\n".join(f"- {path.relative_to(init_resolved).as_posix()}" for path in matches)
    raise ValueError(f"Folder name is ambiguous, use Genus/Folder:\n{choices}")


def plot_bad_bands(ax, wn_ref: np.ndarray) -> None:
    """在图上标出当前配置坏段"""
    valid_mask = build_valid_mask(wn_ref, get_config_bad_bands(config))
    if valid_mask is None:
        return
    bad_mask = ~valid_mask
    if not np.any(bad_mask):
        return

    indices = np.flatnonzero(bad_mask)
    splits = np.where(np.diff(indices) > 1)[0] + 1
    for region in np.split(indices, splits):
        ax.axvspan(float(wn_ref[region[0]]), float(wn_ref[region[-1]]), color="gray", alpha=0.35)


def plot_prefix_group(
    raw_curves: dict[str, np.ndarray],
    snv_curves: dict[str, np.ndarray],
    minmax_curves: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    wn_ref: np.ndarray,
) -> None:
    """绘制同属同前缀 raw、SNV 和 minmax 中位谱总览图"""
    fig, axes = plt.subplots(3, 1, figsize=(PREVIEW_FIGURE_WIDTH, 12), sharex=True)
    for idx, (name, curve) in enumerate(raw_curves.items()):
        color = GLASBEY_DARK_COLORS[idx % len(GLASBEY_DARK_COLORS)]
        axes[0].plot(wn_ref, curve, lw=1.3, color=color, label=name)
    plot_bad_bands(axes[0], wn_ref)

    bad_bands = get_config_bad_bands(config)
    for ax, curves, scale, minimum in (
        (axes[1], snv_curves, SNV_OFFSET_SCALE, SNV_OFFSET_MIN),
        (axes[2], minmax_curves, MINMAX_OFFSET_SCALE, MINMAX_OFFSET_MIN),
    ):
        finite_chunks = [curve[np.isfinite(curve)] for curve in curves.values() if np.isfinite(curve).any()]
        finite_values = np.concatenate(finite_chunks) if finite_chunks else np.asarray([], dtype=np.float32)
        span = float(np.percentile(finite_values, 95) - np.percentile(finite_values, 5)) if finite_values.size else 1.0
        offset_step = max(span * scale, minimum)
        for idx, (name, curve) in enumerate(curves.items()):
            offset = idx * offset_step
            color = GLASBEY_DARK_COLORS[idx % len(GLASBEY_DARK_COLORS)]
            ax.axhline(offset, color="0.88", lw=0.7, zorder=0)
            plot_segments_without_bad_bands(
                ax,
                wn_ref,
                curve + offset,
                bad_bands,
                show_bad_bands=False,
                color=color,
                lw=1.3,
                label=name,
            )
        plot_bad_bands(ax, wn_ref)

    for ax, ylabel, subtitle in (
        (axes[0], "Raw intensity", "raw median"),
        (axes[1], "SNV intensity", "SNV median with vertical offsets"),
        (axes[2], "minmax intensity", "minmax median with vertical offsets"),
    ):
        ax.axvline(TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.35)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle(title)
    axes[2].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_shift_compare(
    before_curve: np.ndarray,
    after_curve: np.ndarray,
    reference_curves: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    wn_ref: np.ndarray,
    target_name: str,
) -> None:
    """绘制平移前后 raw 中位谱及同前缀参考曲线"""
    fig, axes = plt.subplots(2, 1, figsize=(PREVIEW_FIGURE_WIDTH, 8), sharex=True)
    target_color = GLASBEY_DARK_COLORS[len(reference_curves) % len(GLASBEY_DARK_COLORS)]
    for ax, curve, label, subtitle in (
        (axes[0], before_curve, f"{target_name} before shift", "before shift"),
        (axes[1], after_curve, f"{target_name} after shift", "after shift"),
    ):
        for idx, (name, reference) in enumerate(reference_curves.items()):
            ax.plot(
                wn_ref,
                reference,
                lw=1.0,
                alpha=0.8,
                color=GLASBEY_DARK_COLORS[idx % len(GLASBEY_DARK_COLORS)],
                label=name,
            )
        ax.plot(wn_ref, curve, color=target_color, lw=1.0, label=label)
        plot_bad_bands(ax, wn_ref)
        ax.axvline(TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.35)
        ax.set_ylabel("Raw intensity")
        ax.set_title(subtitle)
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.grid(alpha=0.15)
    fig.suptitle(title)
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def read_plot_state(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """读取上次同前缀审图状态"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames not in (list(DELTA_FIELDS), list(LEGACY_PLOT_STATE_FIELDS)):
            return {}
        return {(row["genus"], row["folder"]): row for row in reader}


def write_plot_state(path: Path, rows: list[dict[str, str]]) -> None:
    """写出本次同前缀审图状态"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["genus"], row["folder"]))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_transferred_delta_map(path: Path) -> dict[tuple[str, str], str]:
    """读取迁移 CS 文件夹对应的累计平移快照"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != list(DELTA_CS_FIELDS):
            return {}
        return {
            (row["target_genus"], row["target_folder"]): format_delta(parse_delta(row.get("delta")))
            for row in reader
            if row.get("target_genus") and row.get("target_folder")
        }


def current_delta_state(
    paths: DatasetPaths,
    rows: list[dict[str, str]],
    include_transferred: bool = False,
) -> list[dict[str, str]]:
    """生成当前全部小文件夹 delta 状态"""
    deltas = delta_map(rows)
    transferred_deltas = (
        read_transferred_delta_map(paths.output_dir / DELTA_CS_NAME)
        if include_transferred
        else {}
    )
    state: list[dict[str, str]] = []
    for folder in iter_init_folders(paths.init_dir, include_transferred=include_transferred):
        key = (folder.parent.name, folder.name)
        delta = format_delta(deltas.get(key, 0.0))
        if is_transferred_folder(folder):
            delta = transferred_deltas.get(key, delta)
        state.append(
            {
                "genus": folder.parent.name,
                "folder": folder.name,
                "prefix": shift_folder_prefix(folder),
                "delta": delta,
            }
        )
    return state


def plot_prefix_dataset(dataset: str, include_transferred: bool = False) -> list[Path]:
    """按 delta.txt 变化增量输出同前缀 raw、SNV 和 minmax 中位谱总览图"""
    paths = resolve_dataset(dataset)
    wn_ref = build_plot_grid()
    rows = ensure_delta_rows(paths)
    state = current_delta_state(paths, rows, include_transferred=include_transferred)
    old_state = read_plot_state(paths.output_dir / PREFIX_PLOT_STATE_NAME)
    changed_groups: set[tuple[str, str]] = set()

    for row in state:
        key = (row["genus"], row["folder"])
        if old_state.get(key) != row:
            changed_groups.add((row["genus"], row["prefix"]))
    current_keys = {(row["genus"], row["folder"]) for row in state}
    for key, row in old_state.items():
        if key not in current_keys:
            changed_groups.add((row.get("genus", ""), row.get("prefix", "")))

    outputs: list[Path] = []
    folders_by_group: dict[tuple[str, str], list[Path]] = {}
    for folder in iter_init_folders(paths.init_dir, include_transferred=include_transferred):
        prefix = shift_folder_prefix(folder)
        folders_by_group.setdefault((folder.parent.name, prefix), []).append(folder)

    for (genus, prefix), folders in sorted(folders_by_group.items()):
        out_path = paths.output_dir / genus / f"{prefix}.png"
        if out_path.is_file() and (genus, prefix) not in changed_groups:
            continue

        raw_curves: dict[str, np.ndarray] = {}
        snv_curves: dict[str, np.ndarray] = {}
        minmax_curves: dict[str, np.ndarray] = {}
        for folder in folders:
            raw_curve, snv_curve, minmax_curve, _ = folder_median_curves(folder, wn_ref)
            if raw_curve is not None and snv_curve is not None and minmax_curve is not None:
                raw_curves[folder.name] = raw_curve
                snv_curves[folder.name] = snv_curve
                minmax_curves[folder.name] = minmax_curve
        if not raw_curves:
            continue

        plot_prefix_group(
            raw_curves,
            snv_curves,
            minmax_curves,
            out_path,
            f"{genus} {prefix} median spectra",
            wn_ref,
        )
        outputs.append(out_path)

    write_plot_state(paths.output_dir / PREFIX_PLOT_STATE_NAME, state)
    return outputs


def plot_shift_folder(dataset: str, folder_arg: str) -> Path:
    """按 delta.txt 累计平移量输出单文件夹平移前后和同前缀参考图"""
    paths = resolve_dataset(dataset)
    wn_ref = build_plot_grid()
    folder = resolve_folder(paths.init_dir, folder_arg)
    cumulative_delta = delta_map(ensure_delta_rows(paths)).get((folder.parent.name, folder.name), 0.0)
    if abs(cumulative_delta) < 1e-9:
        raise ValueError(f"Folder has no delta in delta.txt, run apply first: {folder.parent.name}/{folder.name}")

    prefix = shift_folder_prefix(folder)
    before_raw, _ = folder_raw_median_curve(folder, wn_ref, wn_offset=-cumulative_delta)
    after_raw, _ = folder_raw_median_curve(folder, wn_ref)
    if before_raw is None or after_raw is None:
        raise RuntimeError(f"Cannot compute raw median curve: {folder}")

    reference_curves: dict[str, np.ndarray] = {}
    for reference_folder in sorted(path for path in folder.parent.iterdir() if path.is_dir()):
        if reference_folder == folder or is_transferred_folder(reference_folder):
            continue
        if shift_folder_prefix(reference_folder) != prefix:
            continue
        reference_raw, _ = folder_raw_median_curve(reference_folder, wn_ref)
        if reference_raw is not None:
            reference_curves[reference_folder.name] = reference_raw

    out_path = paths.output_dir / "figure" / prefix / f"{folder.name}_shift_compare.png"
    plot_shift_compare(
        before_raw,
        after_raw,
        reference_curves,
        out_path,
        f"{folder.parent.name} {prefix}: {folder.name} cumulative shift {cumulative_delta:+g} cm$^{{-1}}$",
        wn_ref,
        folder.name,
    )
    return out_path


def shift_folder(folder: Path, delta: float) -> int:
    """按增量 delta 修改目标小文件夹的波数列"""
    changed = 0
    for path in sorted(folder.glob("*.arc_data")):
        wn, sp = read_arc_data(path)
        if wn.size == 0:
            continue
        np.savetxt(path, np.column_stack([wn + delta, sp]), fmt=["%.3f", "%.6f"], delimiter="\t")
        changed += 1
    return changed


def apply_shift(dataset: str, folder_arg: str, delta: float, note: str = "") -> tuple[dict[str, str], int]:
    """执行单个小文件夹平移并更新 delta.txt"""
    paths = resolve_dataset(dataset)
    folder = resolve_folder(paths.init_dir, folder_arg)
    rows = ensure_delta_rows(paths)
    existing = delta_map(rows).get((folder.parent.name, folder.name), 0.0)
    cumulative_delta = existing + float(delta)

    changed = shift_folder(folder, float(delta))
    rows = upsert_delta(rows, folder, cumulative_delta)
    write_delta_rows(paths.delta_path, rows)
    append_delta_log(paths.delta_log_path, folder, float(delta), cumulative_delta, changed, note=note)

    row = {
        "genus": folder.parent.name,
        "folder": folder.name,
        "prefix": shift_folder_prefix(folder),
        "delta": format_delta(cumulative_delta),
    }
    return row, changed
