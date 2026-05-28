"""数据审核公共工具"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from raman.data.build import DEFAULT_PIPELINE_CONFIG, _cosmic_ray_kwargs, resolve_pipeline_config
from raman.data.input import normalize_spectrum
from raman.data.preprocess import remove_cosmic_rays
from raman.data.preprocess import preprocess_single_spectrum
from raman.data.io import read_arc_data
from raman.tool.dataset import iter_arc_dirs


def preprocess_spectrum_for_audit(path, profile, cfg=None, wn_ref=None, include_raw=False):
    """按当前离线流程预处理单谱并返回审核载荷"""
    cfg = resolve_pipeline_config(cfg or DEFAULT_PIPELINE_CONFIG)
    path = Path(path)
    payload = {"path": path, "skip_reason": ""}
    wn, sp = read_arc_data(path)
    payload["raw_points"] = int(wn.size)
    if wn.size:
        payload["raw_wn_min"] = float(np.min(wn))
        payload["raw_wn_max"] = float(np.max(wn))
        low = max(payload["raw_wn_min"], float(cfg.cut_min))
        high = min(payload["raw_wn_max"], float(cfg.cut_max))
        payload["coverage_ratio"] = max(0.0, high - low) / max(float(cfg.cut_max) - float(cfg.cut_min), 1e-8)

    if include_raw:
        payload["raw_wn"] = wn
        payload["raw_sp"] = sp

    if wn.size == 0 or sp.size == 0:
        payload["skip_reason"] = "read_failed"
        return payload

    if wn_ref is None:
        wn_ref = cfg.build_wn_ref()

    wn_u, sp_u, cosmic_stats = preprocess_single_spectrum(
        wn,
        sp,
        cut_min=cfg.cut_min,
        cut_max=cfg.cut_max,
        wn_ref=wn_ref,
        bad_bands=cfg.bad_bands,
        baseline_method=cfg.baseline_method,
        baseline_lam=cfg.baseline_lam,
        baseline_asls_p=cfg.baseline_asls_p,
        baseline_max_iter=cfg.baseline_max_iter,
        baseline_fit_min=cfg.baseline_fit_min,
        baseline_fit_max=cfg.baseline_fit_max,
        **_cosmic_ray_kwargs(profile, cfg),
    )
    if wn_u is None or sp_u is None:
        payload["skip_reason"] = "preprocess_failed"
        return payload

    payload.update(
        {
            "wn": wn_u,
            "sp": sp_u,
            "z": normalize_spectrum(sp_u, "snv"),
            "cosmic_replaced": int(cosmic_stats),
            "cosmic_stats": cosmic_stats,
        }
    )
    return payload


def write_csv(path, rows, fieldnames=None):
    """写 CSV，空结果也保留表头"""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else None)
    if not fieldnames:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_audit_records(profile, cfg, input_root, record_cls):
    """读取审核输入根目录下所有叶子目录并执行当前离线预处理"""
    records = []
    wn_ref = cfg.build_wn_ref()
    for root, arc_files in iter_arc_dirs(input_root):
        rel_group = root.relative_to(input_root)
        genus = rel_group.parts[0] if len(rel_group.parts) >= 1 else "."
        folder = rel_group.parts[1] if len(rel_group.parts) >= 2 else root.name
        group = rel_group.as_posix()
        print(f"[Preprocess] {group}: {len(arc_files)} files")

        for filename in arc_files:
            path = root / filename
            payload = preprocess_spectrum_for_audit(path, profile, cfg, wn_ref=wn_ref, include_raw=False)
            record = record_cls(
                path=path,
                rel_path=path.relative_to(input_root).as_posix(),
                group=group,
                genus=genus,
                folder=folder,
                file=filename,
                skip_reason=payload.get("skip_reason", ""),
            )
            record.raw_points = int(payload.get("raw_points", 0))
            record.raw_wn_min = float(payload.get("raw_wn_min", np.nan))
            record.raw_wn_max = float(payload.get("raw_wn_max", np.nan))
            record.coverage_ratio = float(payload.get("coverage_ratio", np.nan))
            if not record.skip_reason:
                stats = payload["cosmic_stats"]
                record.z = np.asarray(payload["z"], dtype=np.float32)
                record.sp = np.asarray(payload["sp"], dtype=np.float32)
                record.cosmic_ray_replaced = int(stats)
            records.append(record)
    return records


def cosmic_clean_for_plot(wn, sp, profile, cfg):
    """仅用于复核图展示宇宙射线清理结果"""
    if profile.profile_id not in set(cfg.cosmic_ray_enabled_profile_ids):
        return np.asarray(sp, dtype=np.float32)
    cleaned, _ = remove_cosmic_rays(
        sp,
        window_points=cfg.cosmic_ray_window_points,
        threshold=cfg.cosmic_ray_threshold,
        max_iter=cfg.cosmic_ray_max_iter,
        valid_mask=None,
    )
    return cleaned


def relative_to_init(path, dataset_dir, init_root, profile):
    """把路径转换成相对 init 的路径"""
    path = Path(path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    init_root = Path(init_root).resolve()
    try:
        return path.relative_to(init_root)
    except ValueError:
        pass

    try:
        rel = path.relative_to(dataset_dir)
    except ValueError:
        return Path(path.name)
    if rel.parts and rel.parts[0] == profile.root_init:
        return Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")
    return rel


def resolve_audit_folder(folder, dataset_dir, profile, init_root):
    """解析绝对路径、属名/文件夹名或唯一末级文件夹名"""
    dataset_dir = Path(dataset_dir)
    folder = Path(str(folder).strip().strip('"').strip("'"))
    candidates = []

    if folder.is_absolute() and folder.is_dir():
        candidates.append(folder.resolve())
    elif folder.is_dir():
        candidates.append(folder.resolve())
    else:
        candidates.extend(
            path.resolve()
            for path in (
                dataset_dir / folder,
                init_root / folder,
            )
            if path.is_dir()
        )
        if len(folder.parts) == 1:
            candidates.extend(path.resolve() for path in sorted(init_root.glob(f"*/{folder.name}")) if path.is_dir())

    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        joined = "\n".join(str(path) for path in unique)
        raise ValueError(f"Audit folder name is ambiguous. Use Genus/Folder:\n{joined}")
    raise FileNotFoundError(f"Audit folder not found under init: {folder}")


def resolve_audit_input(dataset_dir, profile, subdir=None, folder=None):
    """解析审核输入根目录和相对基准路径"""
    dataset_dir = Path(dataset_dir).resolve()
    init_root = (dataset_dir / (subdir or profile.root_init)).resolve()
    if folder is None:
        return init_root, Path(".")
    input_root = resolve_audit_folder(folder, dataset_dir, profile, init_root)
    rel_base = relative_to_init(input_root, dataset_dir, init_root, profile)
    return input_root, rel_base
