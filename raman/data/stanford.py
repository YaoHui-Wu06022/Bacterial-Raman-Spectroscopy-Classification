"""导入 Stanford 公开细菌拉曼参考集。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

from raman.tool.path import PROJECT_ROOT


# 标签顺序与 Ho et al. 公开数据中 y_reference.npy 的整数标签一致。
REFERENCE_LABELS = (
    ("Candida", "CA01", "C. albicans"),
    ("Candida", "CG01", "C. glabrata"),
    ("Klebsiella", "KAE01", "K. aerogenes"),
    ("Escherichia", "EC01", "E. coli 1"),
    ("Escherichia", "EC02", "E. coli 2"),
    ("Enterococcus", "EFM01", "E. faecium"),
    ("Enterococcus", "EFA01", "E. faecalis 1"),
    ("Enterococcus", "EFB01", "E. faecalis 2"),
    ("Enterobacter", "ECL01", "E. cloacae"),
    ("Klebsiella", "KPA01", "K. pneumoniae 1"),
    ("Klebsiella", "KPB01", "K. pneumoniae 2"),
    ("Proteus", "PMI01", "P. mirabilis"),
    ("Pseudomonas", "PAA01", "P. aeruginosa 1"),
    ("Pseudomonas", "PAB01", "P. aeruginosa 2"),
    ("Staphylococcus", "SAA01", "MSSA 1"),
    ("Staphylococcus", "SAB01", "MSSA 3"),
    ("Staphylococcus", "SAC01", "MRSA 1 (isogenic)"),
    ("Staphylococcus", "SAD01", "MRSA 2"),
    ("Staphylococcus", "SAE01", "MSSA 2"),
    ("Salmonella", "SAL01", "S. enterica"),
    ("Staphylococcus", "SEP01", "S. epidermidis"),
    ("Staphylococcus", "SLU01", "S. lugdunensis"),
    ("Serratia", "SMA01", "S. marcescens"),
    ("Streptococcus", "SPA01", "S. pneumoniae 2"),
    ("Streptococcus", "SPB01", "S. pneumoniae 1"),
    ("Streptococcus", "SSG01", "S. sanguinis"),
    ("Streptococcus", "GAS01", "Group A Strep."),
    ("Streptococcus", "GBS01", "Group B Strep."),
    ("Streptococcus", "GCS01", "Group C Strep."),
    ("Streptococcus", "GGS01", "Group G Strep."),
)

TRANSFER_AXIS_FILENAME = "reference_wavenumbers.npy"
REFERENCE_DOWNLOAD_URL = (
    "https://www.dropbox.com/sh/gmgduvzyl5tken6/"
    "AABtSWXWPjoUBkKyC2e7Ag6Da?dl=1"
)
REFERENCE_DATA_FILENAMES = (
    "X_reference.npy",
    "y_reference.npy",
    "wavenumbers.npy",
)


def download_reference_data(dataset_dir: Path | None = None) -> Path:
    """下载公开 Stanford 源数组，并只保留构建参考集所需的三个文件。"""
    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "dataset" / "Stanforddataset")
    target_dir = dataset_dir / "data"
    if target_dir.is_dir() and all((target_dir / name).is_file() for name in REFERENCE_DATA_FILENAMES):
        return target_dir
    if target_dir.exists():
        raise FileExistsError(f"Stanford data 目录不完整，不能覆盖：{target_dir}")

    archive_path = dataset_dir / "stanford_reference_download.zip"
    extract_dir = dataset_dir / "stanford_reference_extracting"
    prepared_dir = dataset_dir / "data_downloading"
    if archive_path.exists() or extract_dir.exists() or prepared_dir.exists():
        raise FileExistsError("检测到未完成的 Stanford 下载临时文件，请检查后再重试")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        print("[Download] downloading Stanford reference source files...", flush=True)
        urlretrieve(REFERENCE_DOWNLOAD_URL, archive_path)
        with ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

        candidates = [
            path.parent
            for path in extract_dir.rglob("X_reference.npy")
            if all((path.parent / name).is_file() for name in REFERENCE_DATA_FILENAMES)
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"下载包中未能唯一定位 Stanford 参考数组，候选数={len(candidates)}")
        source_dir = candidates[0]
        prepared_dir.mkdir(parents=True)
        for name in REFERENCE_DATA_FILENAMES:
            shutil.copy2(source_dir / name, prepared_dir / name)

        spectra = np.load(prepared_dir / "X_reference.npy", mmap_mode="r")
        labels = np.load(prepared_dir / "y_reference.npy", mmap_mode="r")
        wavenumbers = np.load(prepared_dir / "wavenumbers.npy", mmap_mode="r")
        if spectra.shape != (60000, 1000) or labels.shape != (60000,) or wavenumbers.shape != (1000,):
            raise ValueError(
                f"下载的数据形状不符合预期：X={spectra.shape}, y={labels.shape}, wn={wavenumbers.shape}"
            )
        prepared_dir.replace(target_dir)
    finally:
        if target_dir.is_dir():
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"[Download] Stanford source data ready: {target_dir}")
    return target_dir


def ensure_transfer_reference_axis(dataset_dir: Path | None = None) -> Path:
    """从 Stanford 原始波数轴提取连续 896 点的共享训练网格。"""
    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "dataset" / "Stanforddataset")
    source_path = dataset_dir / "data" / "wavenumbers.npy"
    target_path = dataset_dir / TRANSFER_AXIS_FILENAME
    axis = np.load(source_path)
    if np.all(np.diff(axis) < 0):
        axis = axis[::-1]
    elif not np.all(np.diff(axis) > 0):
        raise ValueError("Stanford 波数轴不是单调序列")
    reference_axis = axis[-896:]
    if reference_axis.size != 896 or not np.all(np.diff(reference_axis) > 0):
        raise ValueError("无法从 Stanford 波数轴提取连续 896 点")
    if target_path.exists():
        existing = np.load(target_path)
        if not np.allclose(existing, reference_axis, rtol=0.0, atol=1e-10):
            raise ValueError(f"已有共享波数轴与 Stanford 原始轴不一致：{target_path}")
        return target_path
    np.save(target_path, reference_axis)
    return target_path


def import_reference_init(dataset_dir: Path | None = None, replace_existing: bool = False) -> Path:
    """将 X_reference/y_reference 导入为 30 个菌株类别的标准 init 目录。"""
    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "dataset" / "Stanforddataset")
    source_dir = dataset_dir / "data"
    wn_path = source_dir / "wavenumbers.npy"
    x_path = source_dir / "X_reference.npy"
    y_path = source_dir / "y_reference.npy"
    output_dir = dataset_dir / "init"
    temp_dir = dataset_dir / "init_importing"
    backup_dir = dataset_dir / "init_grouped_backup"

    for path in (wn_path, x_path, y_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 Stanford 参考集文件：{path}")
    if output_dir.exists() and not replace_existing:
        raise FileExistsError(f"目标 init 已存在，不覆盖已有数据：{output_dir}")
    if output_dir.exists() and backup_dir.exists():
        raise FileExistsError(f"已有 init 备份，不能再次覆盖：{backup_dir}")
    if temp_dir.exists() and not temp_dir.is_dir():
        raise FileExistsError(f"临时导入路径不是目录：{temp_dir}")

    wavenumbers = np.load(wn_path)
    spectra = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path).astype(np.int64, copy=False)
    if wavenumbers.ndim != 1 or spectra.ndim != 2 or labels.ndim != 1:
        raise ValueError("Stanford 参考集数组维度不符合预期")
    if spectra.shape != (labels.size, wavenumbers.size):
        raise ValueError(
            f"强度、标签、波数轴形状不匹配：X={spectra.shape}, y={labels.shape}, wn={wavenumbers.shape}"
        )
    if not (np.isfinite(wavenumbers).all() and np.isfinite(spectra).all()):
        raise ValueError("Stanford 参考集中存在非有限数值")

    if np.all(np.diff(wavenumbers) > 0):
        output_wn = wavenumbers
        reverse_spectrum = False
    elif np.all(np.diff(wavenumbers) < 0):
        output_wn = wavenumbers[::-1]
        reverse_spectrum = True
    else:
        raise ValueError("Stanford 波数轴不是单调序列，不能安全导入")

    known_labels = np.arange(len(REFERENCE_LABELS), dtype=np.int64)
    if not np.array_equal(np.unique(labels), known_labels):
        raise ValueError(f"标签不符合预期 0-{len(REFERENCE_LABELS) - 1}")

    temp_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    class_counts = {}
    try:
        for label_id, (genus, isolate, display_name) in enumerate(REFERENCE_LABELS):
            indices = np.flatnonzero(labels == label_id)
            target_dir = temp_dir / isolate
            target_dir.mkdir(parents=True, exist_ok=True)
            for serial, row_index in enumerate(indices):
                intensity = spectra[row_index]
                if reverse_spectrum:
                    intensity = intensity[::-1]
                filename = f"{isolate}_ref_{serial:04d}.arc_data"
                output_path = target_dir / filename
                if output_path.is_file():
                    skipped += 1
                    continue
                output = np.column_stack((output_wn, intensity))
                np.savetxt(output_path, output, fmt="%.10g")
                written += 1
            class_counts[str(label_id)] = {
                "genus": genus,
                "folder": isolate,
                "display_name": display_name,
                "spectra": int(indices.size),
            }
            print(
                f"[Import] {label_id:02d} {isolate}: {indices.size} "
                f"(new={written}, existing={skipped})",
                flush=True,
            )

        imported = sum(1 for _ in temp_dir.rglob("*.arc_data"))
        if imported != labels.size:
            raise RuntimeError(f"导入文件数不一致：expected={labels.size}, actual={imported}")
        metadata = {
            "source": "Stanford Ho et al. reference split",
            "source_files": [wn_path.name, x_path.name, y_path.name],
            "spectra": int(labels.size),
            "points_per_spectrum": int(wavenumbers.size),
            "wavenumber_min": float(output_wn.min()),
            "wavenumber_max": float(output_wn.max()),
            "source_axis_reversed_to_ascending": reverse_spectrum,
            "intensity_transform": "preserved_source_values_without_rescaling",
            "directory_layout": "flat_30_isolate_folders",
            "classes": class_counts,
        }
        (temp_dir / "import_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.replace(backup_dir)
        try:
            temp_dir.replace(output_dir)
        except Exception:
            if backup_dir.exists() and not output_dir.exists():
                backup_dir.replace(output_dir)
            raise
    except Exception:
        raise

    print(f"[Import] completed: {output_dir}")
    ensure_transfer_reference_axis(dataset_dir)
    return output_dir


def flatten_grouped_reference_init(dataset_dir: Path | None = None) -> Path:
    """把旧版属/菌株两层 Stanford init 安全迁移为 30 个直属菌株目录。"""
    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "dataset" / "Stanforddataset")
    output_dir = dataset_dir / "init"
    temp_dir = dataset_dir / "init_importing"
    backup_dir = dataset_dir / "init_grouped_backup"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"缺少待迁移的 Stanford init：{output_dir}")
    if backup_dir.exists():
        raise FileExistsError(f"已有 init 备份，不能重复迁移：{backup_dir}")

    temp_dir.mkdir(parents=True, exist_ok=True)
    for genus, isolate, _ in REFERENCE_LABELS:
        source_dir = output_dir / genus / isolate
        target_dir = temp_dir / isolate
        if not source_dir.is_dir():
            raise FileNotFoundError(f"旧版类别目录不完整：{source_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in source_dir.glob("*.arc_data"):
            target_path = target_dir / source_path.name
            if target_path.exists():
                continue
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)

    imported = sum(1 for _ in temp_dir.rglob("*.arc_data"))
    if imported != 60000:
        raise RuntimeError(f"扁平迁移文件数不一致：expected=60000, actual={imported}")
    class_counts = {
        str(label_id): {
            "genus": genus,
            "folder": isolate,
            "display_name": display_name,
            "spectra": sum(1 for _ in (temp_dir / isolate).glob("*.arc_data")),
        }
        for label_id, (genus, isolate, display_name) in enumerate(REFERENCE_LABELS)
    }
    if any(item["spectra"] != 2000 for item in class_counts.values()):
        raise RuntimeError("扁平迁移后存在样本数不是 2000 的类别")
    (temp_dir / "import_metadata.json").write_text(
        json.dumps(
            {
                "source": "Stanford Ho et al. reference split",
                "spectra": imported,
                "directory_layout": "flat_30_isolate_folders",
                "intensity_transform": "preserved_source_values_without_rescaling",
                "classes": class_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir.replace(backup_dir)
    try:
        temp_dir.replace(output_dir)
    except Exception:
        if not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
    print(f"[Import] flattened to 30 isolate folders: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import_reference_init()
