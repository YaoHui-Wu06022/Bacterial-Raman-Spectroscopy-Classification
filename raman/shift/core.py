from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.config import config
from raman.data.build import get_prefix
from raman.data.profiles import get_dataset_dir, get_profile
from raman.data.spectrum import build_valid_mask, get_config_bad_bands, read_arc_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CM = 1000.0
GRID_STEP = 0.2
DELTA_NAME = "delta.txt"
PREFIX_PLOT_STATE_NAME = "prefix_plot_state.csv"
DELTA_FIELDS = ("genus", "folder", "prefix", "delta")


@dataclass(frozen=True)
class DatasetPaths:
    """数据集 shift 工具所需路径"""

    dataset_dir: Path
    init_dir: Path
    output_dir: Path
    delta_path: Path


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
    )


def build_plot_grid() -> np.ndarray:
    """按当前裁切配置生成绘图波数轴"""
    cut_min = float(getattr(config, "cut_min", 600.0))
    cut_max = float(getattr(config, "cut_max", 1800.0))
    return np.arange(cut_min, cut_max + GRID_STEP * 0.5, GRID_STEP, dtype=np.float32)


def iter_init_folders(init_dir: Path) -> list[Path]:
    """列出 init 下的全部小文件夹"""
    if not init_dir.is_dir():
        raise FileNotFoundError(f"Missing init folder: {init_dir}")
    folders: list[Path] = []
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        folders.extend(sorted(path for path in genus_dir.iterdir() if path.is_dir()))
    return folders


def folder_raw_median_curve(folder: Path, wn_ref: np.ndarray, wn_offset: float = 0.0) -> tuple[np.ndarray | None, int]:
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
                "prefix": get_prefix(folder.name) or folder.name,
                "delta": delta,
            }
        )
    return kept


def resolve_folder(init_dir: Path, folder_arg: str) -> Path:
    """解析 Genus/Folder 或唯一 Folder 名"""
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
    choices = "\n".join(f"- {path.relative_to(init_dir).as_posix()}" for path in matches)
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


def plot_prefix_overview(curves: dict[str, np.ndarray], out_path: Path, title: str, wn_ref: np.ndarray) -> None:
    """绘制同属同前缀 raw 中位谱总览图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    for name, curve in curves.items():
        ax.plot(wn_ref, curve, lw=1.3, label=name)
    plot_bad_bands(ax, wn_ref)
    ax.axvline(TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Raw intensity")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_shift_compare(before_curve: np.ndarray, after_curve: np.ndarray, out_path: Path, title: str, wn_ref: np.ndarray) -> None:
    """绘制平移前后 raw 中位谱对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for ax, curve, label in ((axes[0], before_curve, "Before shift"), (axes[1], after_curve, "After shift")):
        ax.plot(wn_ref, curve, lw=1.4, label=label)
        plot_bad_bands(ax, wn_ref)
        ax.axvline(TARGET_CM, color="black", ls="--", lw=0.8, alpha=0.35)
        ax.set_ylabel("Raw intensity")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.15)
    axes[0].set_title(title)
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
        if reader.fieldnames != list(DELTA_FIELDS):
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


def current_delta_state(paths: DatasetPaths, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """生成当前全部小文件夹 delta 状态"""
    deltas = delta_map(rows)
    state: list[dict[str, str]] = []
    for folder in iter_init_folders(paths.init_dir):
        prefix = get_prefix(folder.name) or folder.name
        state.append(
            {
                "genus": folder.parent.name,
                "folder": folder.name,
                "prefix": prefix,
                "delta": format_delta(deltas.get((folder.parent.name, folder.name), 0.0)),
            }
        )
    return state


def plot_prefix_dataset(dataset: str) -> list[Path]:
    """按 delta.txt 变化增量输出同前缀 raw 中位谱总览图"""
    paths = resolve_dataset(dataset)
    wn_ref = build_plot_grid()
    rows = ensure_delta_rows(paths)
    state = current_delta_state(paths, rows)
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
    for folder in iter_init_folders(paths.init_dir):
        prefix = get_prefix(folder.name) or folder.name
        folders_by_group.setdefault((folder.parent.name, prefix), []).append(folder)

    for (genus, prefix), folders in sorted(folders_by_group.items()):
        out_path = paths.output_dir / genus / f"{prefix}.png"
        if out_path.is_file() and (genus, prefix) not in changed_groups:
            continue

        curves: dict[str, np.ndarray] = {}
        for folder in folders:
            curve, _ = folder_raw_median_curve(folder, wn_ref)
            if curve is not None:
                curves[folder.name] = curve
        if not curves:
            continue

        plot_prefix_overview(curves, out_path, f"{genus} {prefix} raw median spectra", wn_ref)
        outputs.append(out_path)

    write_plot_state(paths.output_dir / PREFIX_PLOT_STATE_NAME, state)
    return outputs


def plot_shift_folder(dataset: str, folder_arg: str) -> Path:
    """按 delta.txt 累计平移量输出单文件夹平移前后 raw 对比图"""
    paths = resolve_dataset(dataset)
    wn_ref = build_plot_grid()
    folder = resolve_folder(paths.init_dir, folder_arg)
    cumulative_delta = delta_map(ensure_delta_rows(paths)).get((folder.parent.name, folder.name), 0.0)
    if abs(cumulative_delta) < 1e-9:
        raise ValueError(f"Folder has no delta in delta.txt, run apply first: {folder.parent.name}/{folder.name}")

    prefix = get_prefix(folder.name) or folder.name
    before_raw, _ = folder_raw_median_curve(folder, wn_ref, wn_offset=-cumulative_delta)
    after_raw, _ = folder_raw_median_curve(folder, wn_ref)
    if before_raw is None or after_raw is None:
        raise RuntimeError(f"Cannot compute raw median curve: {folder}")

    out_path = paths.output_dir / "figure" / prefix / f"{folder.name}_shift_compare.png"
    plot_shift_compare(
        before_raw,
        after_raw,
        out_path,
        f"{folder.parent.name} {prefix}: {folder.name} cumulative shift {cumulative_delta:+g} cm$^{{-1}}$",
        wn_ref,
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


def apply_shift(dataset: str, folder_arg: str, delta: float) -> tuple[dict[str, str], int]:
    """执行单个小文件夹平移并更新 delta.txt"""
    paths = resolve_dataset(dataset)
    folder = resolve_folder(paths.init_dir, folder_arg)
    rows = ensure_delta_rows(paths)
    existing = delta_map(rows).get((folder.parent.name, folder.name), 0.0)
    cumulative_delta = existing + float(delta)

    changed = shift_folder(folder, float(delta))
    rows = upsert_delta(rows, folder, cumulative_delta)
    write_delta_rows(paths.delta_path, rows)

    row = {
        "genus": folder.parent.name,
        "folder": folder.name,
        "prefix": get_prefix(folder.name) or folder.name,
        "delta": format_delta(cumulative_delta),
    }
    return row, changed
