"""新版 audit 工作流：原始坏谱、数据驱动平移和类内近邻审核。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.audit.common import preprocess_spectrum_for_audit, resolve_audit_folder
from raman.audit.test_pool import source_folder_from_audit_folder
from raman.pipeline import DEFAULT_PIPELINE_CONFIG
from raman.tool.dataset import resolve_dataset
from raman.tool.naming import prefix_of
from raman.tool.path import PROJECT_ROOT
from raman.tool.plotting import add_bad_band_spans, plot_segments_without_bad_bands


DELTA_FIELDS = ("genus", "folder", "prefix", "delta")
DELTA_LOG_FIELDS = ("time", "genus", "folder", "prefix", "step_delta", "cumulative_delta", "files_changed", "note")
RAW_MIN_POINTS = 20
RAW_COVERAGE_MIN = 0.98
RAW_FLAT_WINDOW = 40
RAW_LONG_FLAT_POINTS = 100
RAW_SATURATION_POINTS = 25
SIMILAR_MIN_REFS = 8
SHIFT_LIMIT = 30.0
SHIFT_ANCHOR_MIN = 980.0
SHIFT_ANCHOR_MAX = 1020.0
SHIFT_TARGET_CM = 1002.0
SHIFT_RESIDUAL_MIN = 1000.0
SHIFT_RESIDUAL_MAX = 1004.0
SHIFT_SMOOTH_WINDOW = 5
SHIFT_TOTAL_LIMIT = 10.0
SHIFT_LARGE_MOVE_FOLDERS = {"BCC01", "ECL04", "EC03", "KAE03", "KAE04"}


@dataclass
class RawRecord:
    path: Path
    rel_path: str
    genus: str
    folder: str
    prefix: str
    origin_path: Path | None = None
    origin_rel_path: str = ""
    state: str = "keep"
    reasons: tuple[str, ...] = ()
    points: int = 0
    coverage: float = math.nan
    malformed_lines: int = 0
    longest_flat_points: int = 0
    saturation_points: int = 0
    noise_ratio: float = math.nan
    raw_wn: np.ndarray | None = None
    raw_sp: np.ndarray | None = None


@dataclass
class SimilarityRecord:
    raw: RawRecord
    z: np.ndarray | None = None
    state: str = "keep"
    reasons: tuple[str, ...] = ()
    ref_count: int = 0
    neighbor_count: int = 0
    neighbor_corr: float = math.nan
    rmse: float = math.nan
    local_area: float = math.nan
    local_sign: str = ""
    reference: np.ndarray | None = None
    neighbors: tuple[RawRecord, ...] = ()


def _run_dir(dataset_dir: Path, stage: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dataset_dir / "audit_runs" / f"{stamp}_{stage}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _iter_target_records(dataset_dir: Path, test_dir: Path) -> list[RawRecord]:
    init_dir = dataset_dir / "init"
    records = []
    for path in sorted(init_dir.rglob("*.arc_data")):
        rel = path.relative_to(init_dir).as_posix()
        parts = Path(rel).parts
        if len(parts) < 3:
            continue
        genus, folder = parts[0], parts[1]
        source_folder = source_folder_from_audit_folder(folder)
        origin_path = None
        origin_rel = ""
        if source_folder is not None:
            origin_path = test_dir / "init" / source_folder / path.name
            origin_rel = f"{source_folder}/{path.name}"
        records.append(RawRecord(path, rel, genus, folder, prefix_of(folder), origin_path, origin_rel))
    return records


def _prefer_full_test_pool(records: list[RawRecord]) -> list[RawRecord]:
    """全量测试池存在时，排除常规 *t 文件夹中的重复训练副本。"""
    has_full_test_pool = any(source_folder_from_audit_folder(record.folder) is not None for record in records)
    if not has_full_test_pool:
        return records
    return [
        record
        for record in records
        if source_folder_from_audit_folder(record.folder) is not None or not record.folder.lower().endswith("t")
    ]


def _read_raw(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    wn, sp = [], []
    malformed = 0
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            cells = text.split()
            if len(cells) != 2:
                malformed += 1
                continue
            try:
                wn.append(float(cells[0]))
                sp.append(float(cells[1]))
            except ValueError:
                malformed += 1
    return np.asarray(wn, dtype=np.float64), np.asarray(sp, dtype=np.float64), malformed


def _longest_true(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _raw_metrics(record: RawRecord) -> None:
    try:
        wn, sp, malformed = _read_raw(record.path)
    except OSError:
        record.state = "unscorable"
        record.reasons = ("read_failed",)
        return
    record.raw_wn, record.raw_sp = wn, sp
    record.points = int(wn.size)
    record.malformed_lines = malformed
    if malformed:
        record.state = "unscorable"
        record.reasons = ("malformed_rows",)
        return
    if wn.size < RAW_MIN_POINTS:
        record.state = "unscorable"
        record.reasons = ("too_few_points",)
        return
    if not (np.isfinite(wn).all() and np.isfinite(sp).all()):
        record.state = "unscorable"
        record.reasons = ("non_finite_values",)
        return
    if np.any(np.diff(wn) <= 0):
        record.state = "unscorable"
        record.reasons = ("wavenumber_not_strictly_increasing",)
        return
    cfg = DEFAULT_PIPELINE_CONFIG
    low = max(float(wn.min()), float(cfg.cut_min))
    high = min(float(wn.max()), float(cfg.cut_max))
    record.coverage = max(high - low, 0.0) / max(float(cfg.cut_max) - float(cfg.cut_min), 1e-8)
    spread = float(np.quantile(sp, 0.95) - np.quantile(sp, 0.05))
    if spread <= 1e-10:
        record.longest_flat_points = int(sp.size)
        record.saturation_points = int(sp.size)
        record.noise_ratio = 0.0
        return
    if sp.size >= RAW_FLAT_WINDOW:
        flat = np.array(
            [np.ptp(sp[start : start + RAW_FLAT_WINDOW]) <= spread * 0.01 for start in range(sp.size - RAW_FLAT_WINDOW + 1)],
            dtype=bool,
        )
        record.longest_flat_points = _longest_true(flat) + RAW_FLAT_WINDOW - 1 if flat.any() else 0
    rounded = np.round(sp, 6)
    record.saturation_points = _longest_true(np.r_[False, np.diff(rounded) == 0])
    diff = np.diff(sp)
    record.noise_ratio = float(np.median(np.abs(diff - np.median(diff))) / spread)


def _raw_rows(records: list[RawRecord]) -> list[dict[str, object]]:
    return [
        {
            "state": item.state,
            "reasons": ";".join(item.reasons),
            "rel_path": item.rel_path,
            "origin_rel_path": item.origin_rel_path,
            "genus": item.genus,
            "folder": item.folder,
            "prefix": item.prefix,
            "points": item.points,
            "coverage": f"{item.coverage:.6f}" if np.isfinite(item.coverage) else "",
            "malformed_lines": item.malformed_lines,
            "longest_flat_points": item.longest_flat_points,
            "saturation_points": item.saturation_points,
            "noise_ratio": f"{item.noise_ratio:.8f}" if np.isfinite(item.noise_ratio) else "",
        }
        for item in records
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_raw(record: RawRecord, output: Path) -> None:
    if record.raw_wn is None or record.raw_sp is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(record.raw_wn, record.raw_sp, lw=0.7)
    axes[0].set_title(f"{record.rel_path} | {record.state}: {'; '.join(record.reasons)}")
    axes[0].set_ylabel("raw intensity")
    axes[1].plot(record.raw_wn[1:], np.diff(record.raw_sp), lw=0.6)
    axes[1].set_ylabel("raw first difference")
    axes[1].set_xlabel("wavenumber")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _stage_move(record: RawRecord, dataset_dir: Path, test_dir: Path, stage: str) -> None:
    """按来源归档候选：测试谱只保留在测试菌 delete，常规谱保留在 50cos delete。"""
    if record.origin_path is not None:
        if not record.origin_path.is_file():
            raise FileNotFoundError(f"Missing test-source spectrum: {record.origin_path}")
        origin_destination = test_dir / "delete" / stage / Path(record.origin_rel_path)
        origin_destination.parent.mkdir(parents=True, exist_ok=True)
        if origin_destination.exists():
            raise FileExistsError(f"Refusing to overwrite source delete target: {origin_destination}")
        shutil.move(str(record.origin_path), str(origin_destination))
        # record.path 是 50cos 中的临时 audit 副本，删除后避免后续 Stage3 重复扫描，不能归档为 *t 删除谱。
        if record.path.is_file():
            record.path.unlink()
        return

    destination = dataset_dir / "delete" / stage / Path(record.rel_path)
    if record.path.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite delete target: {destination}")
        shutil.move(str(record.path), str(destination))


def run_stage1(dataset_key: str = "cos", test_key: str = "test", move: bool = True) -> Path:
    """扫描原始坏谱，并可把强候选移到 delete/stage1。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    records = _prefer_full_test_pool(_iter_target_records(dataset_dir, test_dir))
    for record in records:
        _raw_metrics(record)
    usable_noise = np.asarray([item.noise_ratio for item in records if item.state == "keep" and np.isfinite(item.noise_ratio)], dtype=float)
    noise_limit = math.inf
    if usable_noise.size:
        median = float(np.median(usable_noise))
        mad = float(np.median(np.abs(usable_noise - median)))
        noise_limit = median + max(8.0 * 1.4826 * mad, median * 2.0, 1e-5)
    for record in records:
        if record.state != "keep":
            continue
        reasons = []
        if record.coverage < RAW_COVERAGE_MIN:
            reasons.append("insufficient_wavenumber_coverage")
        if record.longest_flat_points >= RAW_LONG_FLAT_POINTS:
            reasons.append("long_flat_raw_region")
        if record.saturation_points >= RAW_SATURATION_POINTS:
            reasons.append("repeated_raw_values")
        if record.noise_ratio > noise_limit:
            reasons.append("extreme_raw_noise")
        if reasons:
            record.state = "candidate"
            record.reasons = tuple(reasons)
    out_dir = _run_dir(dataset_dir, "stage1")
    rows = _raw_rows(records)
    _write_csv(out_dir / "stage1_raw_scores.csv", rows)
    candidates = [item for item in records if item.state in {"candidate", "unscorable"}]
    _write_csv(out_dir / "stage1_candidates.csv", _raw_rows(candidates))
    for item in candidates:
        _plot_raw(item, out_dir / "figures" / f"{hashlib.sha1(item.rel_path.encode()).hexdigest()[:12]}.png")
    moved = 0
    if move:
        for item in candidates:
            _stage_move(item, dataset_dir, test_dir, "stage1")
            moved += 1
    payload = {"stage": "stage1", "records": len(records), "candidates": len(candidates), "moved": moved, "noise_limit": noise_limit}
    (out_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "audit_stage1_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _folder_curve(folder: Path) -> np.ndarray | None:
    cfg = DEFAULT_PIPELINE_CONFIG
    grid = cfg.build_wn_ref()
    curves = []
    for path in folder.glob("*.arc_data"):
        try:
            wn, sp, malformed = _read_raw(path)
        except OSError:
            continue
        if malformed or wn.size < RAW_MIN_POINTS or np.any(np.diff(wn) <= 0):
            continue
        if wn.min() > cfg.cut_min or wn.max() < cfg.cut_max:
            continue
        curve = np.interp(grid, wn, sp)
        curve = curve - _moving_average(curve, 101)
        scale = float(np.median(np.abs(curve - np.median(curve))))
        if scale > 1e-8:
            curves.append(curve / (1.4826 * scale))
    return np.median(np.asarray(curves), axis=0) if curves else None


def _anchor_peak(source: np.ndarray, grid: np.ndarray) -> tuple[float, float] | None:
    """返回 950--1050 cm⁻¹ 内 1002 标准峰的位置及相对显著度。"""
    anchor = (grid >= SHIFT_ANCHOR_MIN) & (grid <= SHIFT_ANCHOR_MAX) & np.isfinite(source)
    if anchor.sum() < SHIFT_SMOOTH_WINDOW:
        return None
    values = source[anchor]
    smooth = _moving_average(values, SHIFT_SMOOTH_WINDOW)
    peak_index = int(np.argmax(smooth))
    prominence = float(smooth[peak_index] - np.median(smooth))
    spread = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    if spread <= 1e-8 or prominence <= spread * 0.15:
        return None
    return float(grid[anchor][peak_index]), prominence / spread


def _read_delta(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {(row["genus"], row["folder"]): float(row["delta"]) for row in csv.DictReader(file, delimiter="\t") if row.get("delta")}


def _write_delta(path: Path, values: dict[tuple[str, str], tuple[str, float]]) -> None:
    """写入最终文件夹的累计平移，不保留临时 audit 池名称。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_FIELDS, delimiter="\t")
        writer.writeheader()
        for (genus, folder), (prefix, delta) in sorted(values.items()):
            if source_folder_from_audit_folder(folder) is None and abs(delta) > 1e-9:
                writer.writerow({"genus": genus, "folder": folder, "prefix": prefix, "delta": f"{delta:+g}"})


def _append_delta_log(path: Path, rows: list[dict[str, object]]) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DELTA_LOG_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _shift_file(path: Path, delta: float) -> None:
    wn, sp, malformed = _read_raw(path)
    if malformed or wn.size == 0:
        return
    temp = path.with_suffix(path.suffix + ".audit_tmp")
    np.savetxt(temp, np.column_stack((wn + delta, sp)), fmt=["%.3f", "%.6f"], delimiter="\t")
    temp.replace(path)


def rollback_data_driven_shift(dataset_key: str = "cos") -> int:
    """撤销数据集根目录 delta.txt 所代表的当前新版平移。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    init_dir = dataset_dir / "init"
    current = _read_delta(dataset_dir / "delta.txt")
    logs = []
    now = datetime.now().isoformat(timespec="seconds")
    for (genus, folder), delta in current.items():
        target = init_dir / genus / folder
        files = list(target.glob("*.arc_data"))
        for path in files:
            _shift_file(path, -delta)
        logs.append({"time": now, "genus": genus, "folder": folder, "prefix": prefix_of(folder), "step_delta": f"{-delta:+g}", "cumulative_delta": "", "files_changed": len(files), "note": "rollback_before_peak_constrained_reestimate"})
    _write_delta(dataset_dir / "delta.txt", {})
    if logs:
        _append_delta_log(dataset_dir / "delta_log.txt", logs)
    return len(logs)


def _transfer_folder_map(dataset_dir: Path) -> dict[tuple[str, str], str]:
    """从临时 audit 文件夹名映射回测试来源，不依赖迁移 manifest。"""
    result = {}
    init_dir = dataset_dir / "init"
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        for folder in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            source_folder = source_folder_from_audit_folder(folder.name)
            if source_folder is not None:
                result[(genus_dir.name, folder.name)] = source_folder
    return result


def run_data_driven_shift(dataset_key: str = "cos", test_key: str = "test") -> Path:
    """以 1002 cm⁻¹ 标准峰为锚点，稀疏地平移各文件夹并同步测试菌来源。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    init_dir = dataset_dir / "init"
    out_dir = _run_dir(dataset_dir, "shift")
    grid = DEFAULT_PIPELINE_CONFIG.build_wn_ref()
    folders: list[tuple[Path, np.ndarray]] = []
    for genus_dir in sorted(path for path in init_dir.iterdir() if path.is_dir()):
        for folder in sorted(path for path in genus_dir.iterdir() if path.is_dir()):
            curve = _folder_curve(folder)
            if curve is not None:
                folders.append((folder, curve))
    current = _read_delta(dataset_dir / "delta.txt")
    legacy = _read_delta(dataset_dir / "fig_init" / "delta.txt")
    transfer_sources = _transfer_folder_map(dataset_dir)
    test_current = _read_delta(test_dir / "delta.txt")
    for target, source_folder in transfer_sources.items():
        current.setdefault(target, test_current.get((".", source_folder), 0.0))
    absolute: dict[tuple[str, str], tuple[str, float]] = {}
    plan = []
    for folder, curve in sorted(folders, key=lambda item: (item[0].parent.name, item[0].name)):
        genus, prefix = folder.parent.name, prefix_of(folder.name)
        key = (genus, folder.name)
        legacy_delta = legacy.get(key, 0.0)
        peak = _anchor_peak(curve, grid)
        step = 0.0
        status = "unresolved"
        peak_cm = ""
        prominence = ""
        if peak is not None:
            position, quality = peak
            peak_cm = f"{position:.3f}"
            prominence = f"{quality:.6f}"
            proposed = round(SHIFT_TARGET_CM - position, 1)
            current_delta = current.get(key, 0.0)
            if position < SHIFT_RESIDUAL_MIN or position > SHIFT_RESIDUAL_MAX:
                desired_total = current_delta + proposed
            else:
                desired_total = current_delta
            if abs(proposed) > SHIFT_LIMIT:
                status = "unresolved"
            elif folder.name not in SHIFT_LARGE_MOVE_FOLDERS and abs(desired_total) > SHIFT_TOTAL_LIMIT:
                if abs(current_delta) > SHIFT_TOTAL_LIMIT:
                    step = math.copysign(SHIFT_TOTAL_LIMIT, current_delta) - current_delta
                    status = "limited_by_total_shift_cap"
                else:
                    status = "unresolved"
            elif position < SHIFT_RESIDUAL_MIN or position > SHIFT_RESIDUAL_MAX:
                step = proposed
                status = "applied"
            else:
                status = "kept_within_residual"
        cumulative = current.get(key, 0.0) + step
        absolute[key] = (prefix, cumulative)
        plan.append({"genus": genus, "folder": folder.name, "prefix": prefix, "reference_folder": f"{SHIFT_TARGET_CM:.0f}cm-1", "step_delta": step, "anchor_peak_cm": peak_cm, "peak_prominence": prominence, "status": status, "legacy_delta": legacy_delta, "residual_range": f"{SHIFT_RESIDUAL_MIN:.0f}-{SHIFT_RESIDUAL_MAX:.0f}", "total_shift_limit": "" if folder.name in SHIFT_LARGE_MOVE_FOLDERS else f"{SHIFT_TOTAL_LIMIT:.0f}"})
    _write_csv(out_dir / "shift_plan.csv", plan)
    logs = []
    test_logs = []
    now = datetime.now().isoformat(timespec="seconds")
    for row in plan:
        step = float(row["step_delta"])
        if row["status"] not in {"applied", "limited_by_total_shift_cap"} or abs(step) < 1e-9:
            continue
        folder = init_dir / str(row["genus"]) / str(row["folder"])
        files = list(folder.glob("*.arc_data"))
        for path in files:
            _shift_file(path, step)
        _, cumulative = absolute[(str(row["genus"]), str(row["folder"]))]
        source_folder = transfer_sources.get((str(row["genus"]), str(row["folder"])))
        if source_folder:
            source_files = list((test_dir / "init" / source_folder).glob("*.arc_data"))
            for path in source_files:
                _shift_file(path, step)
            test_logs.append({"time": now, "genus": ".", "folder": source_folder, "prefix": prefix_of(source_folder), "step_delta": f"{step:+g}", "cumulative_delta": f"{cumulative:+g}", "files_changed": len(source_files), "note": f"synced_from_50cos={row['genus']}/{row['folder']}"})
        logs.append({"time": now, "genus": row["genus"], "folder": row["folder"], "prefix": row["prefix"], "step_delta": f"{step:+g}", "cumulative_delta": f"{cumulative:+g}", "files_changed": len(files), "note": f"anchor_peak={row['anchor_peak_cm']}; target={SHIFT_TARGET_CM:.0f}"})
    _write_delta(dataset_dir / "delta.txt", absolute)
    test_absolute = {
        key: (prefix_of(folder), delta)
        for key, delta in test_current.items()
        for _genus, folder in [key]
    }
    test_absolute.update({
        (".", source_folder): (prefix_of(source_folder), absolute[target][1])
        for target, source_folder in transfer_sources.items()
        if target in absolute
    })
    _write_delta(test_dir / "delta.txt", test_absolute)
    if logs:
        _append_delta_log(dataset_dir / "delta_log.txt", logs)
    if test_logs:
        _append_delta_log(test_dir / "delta_log.txt", test_logs)
    (out_dir / "run.json").write_text(json.dumps({"stage": "shift", "folders": len(plan), "applied": len(logs)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def _robust_limits(values: np.ndarray, direction: str) -> float:
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center))) * 1.4826
    scale = max(scale, 1e-6)
    return center - 3.5 * scale if direction == "low" else center + 3.5 * scale


def _score_similarity_group(group: list[SimilarityRecord]) -> None:
    """按同属同前缀的近邻关系更新一组谱的阶段二指标。"""
    n = len(group)
    if n - 1 < SIMILAR_MIN_REFS:
        for item in group:
            item.state = "insufficient_reference"
        return

    spectra = np.stack([item.z for item in group])
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    corr = (centered @ centered.T) / np.maximum(np.outer(norms, norms), 1e-8)
    k = max(3, int(math.sqrt(n - 1)))
    for index, item in enumerate(group):
        order = np.argsort(-corr[index])
        neighbors = [idx for idx in order if idx != index][:k]
        reference = np.median(spectra[neighbors], axis=0)
        item.ref_count = n - 1
        item.neighbor_count = len(neighbors)
        item.neighbor_corr = float(np.median(corr[index, neighbors]))
        item.rmse = float(np.sqrt(np.mean((spectra[index] - reference) ** 2)))
        smooth = _moving_average(spectra[index] - reference, 5)
        excess = np.maximum(np.abs(smooth) - 0.30, 0.0)
        item.local_area = float(excess.sum())
        item.local_sign = "positive" if float(np.max(smooth)) >= abs(float(np.min(smooth))) else "negative"
        item.reference = reference
        item.neighbors = tuple(group[idx].raw for idx in neighbors)
        for idx in neighbors:
            setattr(group[idx].raw, "_z", spectra[idx])

    corr_values = np.asarray([item.neighbor_corr for item in group])
    rmse_values = np.asarray([item.rmse for item in group])
    area_values = np.asarray([item.local_area for item in group])
    corr_limit = _robust_limits(corr_values, "low")
    rmse_limit = _robust_limits(rmse_values, "high")
    area_limit = float(np.median(area_values)) + 5.5 * max(
        float(np.median(np.abs(area_values - np.median(area_values)))) * 1.4826,
        1e-6,
    )
    corr_local_limit = float(np.median(corr_values)) - 2.0 * max(
        float(np.median(np.abs(corr_values - np.median(corr_values)))) * 1.4826,
        1e-6,
    )
    for item in group:
        global_mismatch = item.neighbor_corr < corr_limit and item.rmse > rmse_limit
        local_mismatch = item.local_area > area_limit and item.neighbor_corr < corr_local_limit
        if global_mismatch:
            item.state = "candidate"
            reasons = ["low_neighbor_agreement", "high_neighbor_rmse"]
            if local_mismatch:
                reasons.append(f"local_{item.local_sign}_deviation")
            item.reasons = tuple(reasons)
        elif local_mismatch:
            item.state = "local_review"
            item.reasons = (f"local_{item.local_sign}_deviation",)


def _shift_unresolved_folders(dataset_dir: Path) -> set[tuple[str, str]]:
    runs = [path for path in (dataset_dir / "audit_runs").glob("*_shift") if (path / "shift_plan.csv").is_file()]
    if not runs:
        return set()
    latest = max(runs, key=lambda path: path.stat().st_mtime)
    with (latest / "shift_plan.csv").open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row["genus"], row["folder"])
            for row in csv.DictReader(file)
            if row.get("status") in {"unresolved", "limited_by_total_shift_cap"}
        }


def _similar_rows(records: list[SimilarityRecord]) -> list[dict[str, object]]:
    return [
        {
            "state": item.state,
            "reasons": ";".join(item.reasons),
            "rel_path": item.raw.rel_path,
            "origin_rel_path": item.raw.origin_rel_path,
            "genus": item.raw.genus,
            "folder": item.raw.folder,
            "prefix": item.raw.prefix,
            "ref_count": item.ref_count,
            "neighbor_count": item.neighbor_count,
            "neighbor_corr": f"{item.neighbor_corr:.6f}" if np.isfinite(item.neighbor_corr) else "",
            "rmse": f"{item.rmse:.6f}" if np.isfinite(item.rmse) else "",
            "local_area": f"{item.local_area:.6f}" if np.isfinite(item.local_area) else "",
            "local_sign": item.local_sign,
        }
        for item in records
    ]


def _stage3_rows(records: list[SimilarityRecord]) -> list[dict[str, object]]:
    """在阶段二指标外补充同一 cell 的 000/001 重复测量相关性。"""
    pairs: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for item in records:
        name = item.raw.path.name
        cell_name = name.replace("_000_shift_cos.arc_data", "").replace("_001_shift_cos.arc_data", "")
        pairs.setdefault((item.raw.folder, cell_name), []).append(item)

    pair_corr: dict[int, float] = {}
    for pair in pairs.values():
        if len(pair) != 2 or pair[0].z is None or pair[1].z is None:
            continue
        first = pair[0].z - pair[0].z.mean()
        second = pair[1].z - pair[1].z.mean()
        denom = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8)
        value = float(first @ second / denom)
        pair_corr[id(pair[0])] = value
        pair_corr[id(pair[1])] = value

    rows = _similar_rows(records)
    for item, row in zip(records, rows):
        value = pair_corr.get(id(item))
        row["pair_corr"] = f"{value:.6f}" if value is not None else ""
    return rows


def _moved_similarity_rows(records: list[SimilarityRecord]) -> list[dict[str, object]]:
    """为已移动的相似度候选补充来源数据集字段。"""
    rows = _similar_rows(records)
    for item, row in zip(records, rows):
        row["source_dataset"] = "test" if item.raw.origin_path is not None else "cos"
    return rows


def run_stage3(
    dataset_key: str = "cos",
    test_key: str = "test",
    folder: str | None = None,
    draw: bool = False,
    move: bool = True,
) -> Path:
    """在单批次内部扫描异常谱，并可把候选移到 delete/stage3。"""
    profile, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    init_dir = dataset_dir / "init"
    target_key = None
    target_prefix = None
    if folder:
        target_dir = resolve_audit_folder(folder, dataset_dir, profile, init_dir)
        rel = target_dir.relative_to(init_dir.resolve())
        if len(rel.parts) < 2:
            raise ValueError(f"审核文件夹必须位于 init/属/文件夹：{target_dir}")
        target_key = (rel.parts[0], rel.parts[1])
        target_prefix = prefix_of(target_key[1])

    raws = _prefer_full_test_pool(_iter_target_records(dataset_dir, test_dir))
    has_full_test_pool = any(source_folder_from_audit_folder(raw.folder) is not None for raw in raws)
    if target_key is not None:
        raws = [
            raw
            for raw in raws
            if raw.genus == target_key[0] and raw.prefix == target_prefix
        ]

    unresolved = _shift_unresolved_folders(dataset_dir)
    records = []
    for raw in raws:
        if (raw.genus, raw.folder) in unresolved:
            records.append(SimilarityRecord(raw=raw, state="shift_unresolved", reasons=("shift_anchor_unresolved",)))
            continue
        payload = preprocess_spectrum_for_audit(raw.path, profile, DEFAULT_PIPELINE_CONFIG)
        item = SimilarityRecord(raw=raw, z=np.asarray(payload["z"], dtype=np.float32) if not payload.get("skip_reason") else None)
        if item.z is None:
            item.state = "unscorable"
            item.reasons = (payload.get("skip_reason", "preprocess_failed"),)
        records.append(item)

    by_class: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for item in records:
        if item.z is not None:
            by_class.setdefault((item.raw.genus, item.raw.prefix), []).append(item)

    if target_key is not None:
        selected = {(target_key[0], target_prefix)}
    else:
        selected = {
            key
            for key, group in by_class.items()
            if len({item.raw.folder for item in group}) > 1
        }
    for key in selected:
        by_folder: dict[str, list[SimilarityRecord]] = {}
        for item in by_class.get(key, []):
            by_folder.setdefault(item.raw.folder, []).append(item)
        for folder_group in by_folder.values():
            _score_similarity_group(folder_group)

    selected_records = [
        item
        for item in records
        if (item.raw.genus, item.raw.prefix) in selected
        and (target_key is None or (item.raw.genus, item.raw.folder) == target_key)
    ]
    out_dir = _run_dir(dataset_dir, "stage3")
    _write_csv(out_dir / "stage3_scores.csv", _stage3_rows(selected_records))
    candidates = [item for item in selected_records if item.state == "candidate"]
    moved = 0
    moved_records: list[SimilarityRecord] = []
    if move:
        for item in candidates:
            _stage_move(item.raw, dataset_dir, test_dir, "stage3")
            moved_records.append(item)
            moved += 1
        _write_csv(out_dir / "stage3_moved.csv", _moved_similarity_rows(moved_records))
    if draw:
        for item in selected_records:
            if item.state not in {"candidate", "local_review"}:
                continue
            output = out_dir / "figures" / item.state / item.raw.genus / item.raw.folder / f"{item.raw.path.stem}.png"
            _plot_similarity(item, output)
    payload = {
        "stage": "stage3",
        "folder": folder or "all_multi_batch_classes",
        "records": len(selected_records),
        "groups": len(selected),
        "candidates": len(candidates),
        "local_reviews": sum(item.state == "local_review" for item in selected_records),
        "moved": moved,
        "moved_from_main_dataset": sum(item.raw.origin_path is None for item in moved_records),
        "moved_from_test_dataset": sum(item.raw.origin_path is not None for item in moved_records),
        "full_test_pool": has_full_test_pool,
        "reference_scope": "within_batch",
        "figures": sum(item.state in {"candidate", "local_review"} for item in selected_records) if draw else 0,
    }
    (out_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def _plot_similarity(record: SimilarityRecord, output: Path) -> None:
    if record.z is None or record.reference is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    grid = DEFAULT_PIPELINE_CONFIG.build_wn_ref()
    bad_bands = DEFAULT_PIPELINE_CONFIG.bad_bands
    for low, high in bad_bands:
        grid = grid[(grid < low) | (grid > high)]
    for neighbor in record.neighbors:
        if hasattr(neighbor, "_z"):
            plot_segments_without_bad_bands(axes[0], grid, neighbor._z, bad_bands, show_bad_bands=False, color="0.75", lw=0.5)
    plot_segments_without_bad_bands(axes[0], grid, record.reference, bad_bands, show_bad_bands=False, color="black", lw=1.2, label="neighbor median")
    plot_segments_without_bad_bands(axes[0], grid, record.z, bad_bands, show_bad_bands=False, color="C3", lw=0.9, label="sample")
    add_bad_band_spans(axes[0], bad_bands, alpha=0.35)
    axes[0].legend()
    residual = record.z - record.reference
    plot_segments_without_bad_bands(axes[1], grid, residual, bad_bands, show_bad_bands=False, color="C4", lw=0.8)
    add_bad_band_spans(axes[1], bad_bands, alpha=0.35)
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_title(f"{record.raw.rel_path} | {'; '.join(record.reasons)}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run_stage2(dataset_key: str = "cos", test_key: str = "test", move: bool = True) -> Path:
    """对阶段一保留谱按类内近邻进行低相似性审核。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    _, test_dir = resolve_dataset(test_key, PROJECT_ROOT)
    if not (dataset_dir / "audit_stage1_complete.json").is_file():
        raise RuntimeError("Run stage1 before stage2")
    raws = _prefer_full_test_pool(_iter_target_records(dataset_dir, test_dir))
    profile, _ = resolve_dataset(dataset_key, PROJECT_ROOT)
    unresolved = _shift_unresolved_folders(dataset_dir)
    records = []
    for raw in raws:
        if (raw.genus, raw.folder) in unresolved:
            records.append(SimilarityRecord(raw=raw, state="shift_unresolved", reasons=("shift_anchor_unresolved",)))
            continue
        payload = preprocess_spectrum_for_audit(raw.path, profile, DEFAULT_PIPELINE_CONFIG)
        item = SimilarityRecord(raw=raw, z=np.asarray(payload["z"], dtype=np.float32) if not payload.get("skip_reason") else None)
        if item.z is None:
            item.state = "unscorable"
            item.reasons = (payload.get("skip_reason", "preprocess_failed"),)
        records.append(item)
    by_class: dict[tuple[str, str], list[SimilarityRecord]] = {}
    for item in records:
        if item.z is not None:
            by_class.setdefault((item.raw.genus, item.raw.prefix), []).append(item)
    for group in by_class.values():
        _score_similarity_group(group)
    out_dir = _run_dir(dataset_dir, "stage2")
    _write_csv(out_dir / "stage2_similarity_scores.csv", _similar_rows(records))
    candidates = [item for item in records if item.state == "candidate"]
    _write_csv(out_dir / "stage2_candidates.csv", _similar_rows(candidates))
    for item in candidates:
        _plot_similarity(item, out_dir / "figures" / f"{hashlib.sha1(item.raw.rel_path.encode()).hexdigest()[:12]}.png")
    moved = 0
    moved_records: list[SimilarityRecord] = []
    if move:
        for item in candidates:
            _stage_move(item.raw, dataset_dir, test_dir, "stage2")
            moved_records.append(item)
            moved += 1
        _write_csv(out_dir / "stage2_moved.csv", _moved_similarity_rows(moved_records))
    payload = {
        "stage": "stage2",
        "records": len(records),
        "candidates": len(candidates),
        "moved": moved,
        "moved_from_main_dataset": sum(item.raw.origin_path is None for item in moved_records),
        "moved_from_test_dataset": sum(item.raw.origin_path is not None for item in moved_records),
    }
    (out_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def run_cleaning_pipeline(dataset_key: str = "cos", test_key: str = "test") -> Path:
    """执行连续清洗：三个阶段的强候选均在各自阶段直接删除。"""
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)

    stage1_dir = run_stage1(dataset_key, test_key, move=True)
    shift_dir = run_data_driven_shift(dataset_key, test_key)
    stage2_dir = run_stage2(dataset_key, test_key, move=True)
    stage3_dir = run_stage3(dataset_key, test_key, move=True)

    out_dir = _run_dir(dataset_dir, "clean")
    stage2_payload = json.loads((stage2_dir / "run.json").read_text(encoding="utf-8"))
    stage3_payload = json.loads((stage3_dir / "run.json").read_text(encoding="utf-8"))
    payload = {
        "stage": "clean",
        "stage1_run": stage1_dir.name,
        "shift_run": shift_dir.name,
        "stage2_run": stage2_dir.name,
        "stage3_run": stage3_dir.name,
        "stage1_moved": json.loads((stage1_dir / "run.json").read_text(encoding="utf-8"))["moved"],
        "stage2_moved": stage2_payload["moved"],
        "stage2_moved_from_main_dataset": stage2_payload["moved_from_main_dataset"],
        "stage2_moved_from_test_dataset": stage2_payload["moved_from_test_dataset"],
        "stage3_moved": stage3_payload["moved"],
        "stage3_moved_from_main_dataset": stage3_payload["moved_from_main_dataset"],
        "stage3_moved_from_test_dataset": stage3_payload["moved_from_test_dataset"],
        "moved": stage2_payload["moved"] + stage3_payload["moved"],
    }
    (out_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir
