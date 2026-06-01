from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from raman.data.build import DEFAULT_PIPELINE_CONFIG
from raman.data.io import read_arc_data
from raman.data.preprocess import save_mean_plot, save_mean_summary_plot
from raman.tool.dataset import iter_arc_dirs
from raman.tool.hierarchy import iter_ancestor_level_keys, safe_key_name
from raman.tool.path import resolve_under_base


@dataclass(frozen=True)
class TrainPlotConfig:
    """训练集均值图配置"""

    norm_method: str | None = None
    bad_bands: tuple[tuple[float, float], ...] = DEFAULT_PIPELINE_CONFIG.bad_bands


DEFAULT_TRAIN_PLOT_CONFIG = TrainPlotConfig()


def resolve_train_plot_config(plot_config=None):
    """补全训练集绘图配置"""
    cfg = plot_config or DEFAULT_TRAIN_PLOT_CONFIG
    if cfg.norm_method:
        return cfg

    from raman.config import config as runtime_config

    return replace(cfg, norm_method=str(runtime_config.norm_method))


def _read_train_group(folder, arc_files):
    """读取一个 train 叶子类别目录"""
    wn_ref = None
    spectra = []
    for filename in arc_files:
        wn, sp = read_arc_data(folder / filename)
        if wn.size == 0 or sp.size == 0:
            continue
        if wn_ref is None:
            wn_ref = wn
        elif wn.shape != wn_ref.shape or not np.allclose(wn, wn_ref):
            raise ValueError(f"Inconsistent wavenumber axis: {folder / filename}")
        spectra.append(sp)

    if wn_ref is None or not spectra:
        return None
    return {
        "wn": wn_ref,
        "spectra": np.vstack(spectra),
    }


def _append_mean_group(groups, level_idx, parts, wn, spectra):
    """将一组光谱追加到指定层级类别"""
    payload = groups.setdefault(
        (level_idx, tuple(parts)),
        {
            "wn": wn,
            "spectra": [],
        },
    )
    payload["spectra"].append(spectra)


def _save_leaf_plot(root_figure, rel_dir, payload, cfg):
    """输出一个 train 叶子类别的均值图"""
    fig_path = root_figure / rel_dir.parent / f"{rel_dir.name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    save_mean_plot(
        wn=payload["wn"],
        spectra=payload["spectra"],
        out_path=fig_path,
        norm_method=cfg.norm_method,
        bad_bands=cfg.bad_bands,
        title=" - ".join(rel_dir.parts),
    )
    print(f"  Mean spectrum saved: {fig_path}")


def _save_hierarchy_mean_plots(hierarchy_groups, root_figure, cfg):
    """输出每个高层类别的独立均值图"""
    generated = 0
    output_root = root_figure / "_hierarchy_mean"
    for (level_idx, parts), payload in sorted(hierarchy_groups.items()):
        level_dir = output_root / f"level_{level_idx}"
        level_dir.mkdir(parents=True, exist_ok=True)
        fig_path = level_dir / f"{safe_key_name(parts)}.png"
        save_mean_plot(
            wn=payload["wn"],
            spectra=np.vstack(payload["spectra"]),
            out_path=fig_path,
            norm_method=cfg.norm_method,
            bad_bands=cfg.bad_bands,
            title="/".join(parts),
        )
        print(f"  Hierarchy mean spectrum saved: {fig_path}")
        generated += 1
    return generated


def _save_hierarchy_summary_plots(summary_groups, root_figure, cfg):
    """输出每个层级的纵向均值谱长图"""
    grouped_by_level = {}
    for (level_idx, parts), payload in sorted(summary_groups.items()):
        grouped_by_level.setdefault(level_idx, []).append(
            {
                "label": parts[-1],
                "wn": payload["wn"],
                "spectra": np.vstack(payload["spectra"]),
            }
        )

    generated = 0
    summary_dir = root_figure / "_hierarchy_mean" / "summary"
    for level_idx, groups in sorted(grouped_by_level.items()):
        fig_path = summary_dir / f"level_{level_idx}.png"
        save_mean_summary_plot(
            groups=groups,
            out_path=fig_path,
            norm_method=cfg.norm_method,
            bad_bands=cfg.bad_bands,
        )
        print(f"  Hierarchy summary saved: {fig_path}")
        generated += 1
    return generated


def _validate_managed_dir(path, dataset_dir):
    """确认生成目录位于目标数据集内部"""
    path = Path(path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    if path == dataset_dir:
        raise ValueError(f"Refuse to replace dataset root: {path}")
    path.relative_to(dataset_dir)
    return path


def _remove_generated_dir(path, dataset_dir):
    """删除已验证位于数据集内部的生成目录"""
    path = _validate_managed_dir(path, dataset_dir)
    if path.exists():
        shutil.rmtree(path)


def _replace_generated_dir(temp_dir, final_dir, dataset_dir):
    """用已生成的临时目录安全替换 fig_train"""
    temp_dir = _validate_managed_dir(temp_dir, dataset_dir)
    final_dir = _validate_managed_dir(final_dir, dataset_dir)
    backup_dir = _validate_managed_dir(final_dir.parent / f".{final_dir.name}_backup", dataset_dir)
    if backup_dir.exists():
        raise FileExistsError(f"Stale plot backup exists: {backup_dir}")

    if final_dir.exists():
        final_dir.rename(backup_dir)
    try:
        temp_dir.rename(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise
    if backup_dir.exists():
        _remove_generated_dir(backup_dir, dataset_dir)


def _generate_train_plots(root_train, root_figure, cfg):
    """从 train 目录生成全部叶子图、层级图和汇总图"""
    hierarchy_groups = {}
    summary_groups = {}
    leaf_count = 0

    for folder, arc_files in iter_arc_dirs(root_train):
        rel_dir = folder.relative_to(root_train)
        payload = _read_train_group(folder, arc_files)
        if payload is None:
            continue
        _save_leaf_plot(root_figure, rel_dir, payload, cfg)
        leaf_count += 1

        for level_idx, parts in iter_ancestor_level_keys(rel_dir):
            _append_mean_group(hierarchy_groups, level_idx, parts, payload["wn"], payload["spectra"])

        parts = tuple(rel_dir.parts)
        for level_idx in range(1, len(parts) + 1):
            _append_mean_group(summary_groups, level_idx, parts[:level_idx], payload["wn"], payload["spectra"])

    if leaf_count == 0:
        raise ValueError(f"No readable train spectra found: {root_train}")

    hierarchy_count = _save_hierarchy_mean_plots(hierarchy_groups, root_figure, cfg)
    summary_count = _save_hierarchy_summary_plots(summary_groups, root_figure, cfg)
    return leaf_count, hierarchy_count, summary_count


def plot_train(profile, base_dir, plot_config=None):
    """从已有 train 安全重建 fig_train"""
    cfg = resolve_train_plot_config(plot_config)
    base_dir = Path(base_dir).resolve()
    root_train = resolve_under_base(base_dir, profile.root_train_clean)
    root_figure = resolve_under_base(base_dir, profile.root_train_fig)
    if not root_train.is_dir():
        raise FileNotFoundError(f"Missing train folder: {root_train}")

    root_figure.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{root_figure.name}_tmp_", dir=root_figure.parent))
    _validate_managed_dir(temp_dir, base_dir)
    try:
        leaf_count, hierarchy_count, summary_count = _generate_train_plots(root_train, temp_dir, cfg)
        _replace_generated_dir(temp_dir, root_figure, base_dir)
    except Exception:
        _remove_generated_dir(temp_dir, base_dir)
        raise

    print("\nTraining plot generation finished:")
    print(f"- Source train spectra: {root_train}")
    print(f"- Mean plots: {root_figure}")
    print(f"- Leaf mean plots: {leaf_count}")
    print(f"- Hierarchy mean plots: {hierarchy_count}")
    print(f"- Hierarchy summary plots: {summary_count}")
