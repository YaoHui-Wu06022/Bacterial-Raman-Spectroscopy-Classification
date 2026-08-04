"""Stanford 数据导入、参考轴和预训练集构建。"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

from ramanv2.core.config import InputConfig
from ramanv2.core.paths import DATASET_ROOT, stanford_reference_wavenumbers_path
from ramanv2.data.io import iter_arc_dirs, read_arc_data, unpack_init, write_arc_data
from ramanv2.spectra.bands import build_valid_mask

from .config import StanfordPrepareConfig


REFERENCE_DOWNLOAD_URL = (
    "https://www.dropbox.com/sh/gmgduvzyl5tken6/"
    "AABtSWXWPjoUBkKyC2e7Ag6Da?dl=1"
)
REFERENCE_DATA_FILENAMES = ("X_reference.npy", "y_reference.npy", "wavenumbers.npy")
REFERENCE_LABELS = (
    ("Candida", "CA01"), ("Candida", "CG01"), ("Klebsiella", "KAE01"),
    ("Escherichia", "EC01"), ("Escherichia", "EC02"), ("Enterococcus", "EFM01"),
    ("Enterococcus", "EFA01"), ("Enterococcus", "EFB01"), ("Enterobacter", "ECL01"),
    ("Klebsiella", "KPA01"), ("Klebsiella", "KPB01"), ("Proteus", "PMI01"),
    ("Pseudomonas", "PAA01"), ("Pseudomonas", "PAB01"), ("Staphylococcus", "SAA01"),
    ("Staphylococcus", "SAB01"), ("Staphylococcus", "SAC01"), ("Staphylococcus", "SAD01"),
    ("Staphylococcus", "SAE01"), ("Salmonella", "SAL01"), ("Staphylococcus", "SEP01"),
    ("Staphylococcus", "SLU01"), ("Serratia", "SMA01"), ("Streptococcus", "SPA01"),
    ("Streptococcus", "SPB01"), ("Streptococcus", "SSG01"), ("Streptococcus", "GAS01"),
    ("Streptococcus", "GBS01"), ("Streptococcus", "GCS01"), ("Streptococcus", "GGS01"),
)


def stanford_dataset_dir() -> Path:
    """返回包外 Stanford 数据集固定目录。"""
    return DATASET_ROOT / "Stanforddataset"


def load_reference_wavenumbers() -> np.ndarray:
    """读取并校验固定 Stanford 共享参考波数轴。"""
    path = stanford_reference_wavenumbers_path()
    if not path.is_file():
        raise FileNotFoundError(f"缺少 Stanford 参考波数轴：{path}")
    values = np.load(path)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
        raise ValueError(f"Stanford 参考波数轴无效：{path}")
    return values.astype(np.float64, copy=False)


def download_reference_data(dataset_dir: Path | None = None) -> Path:
    """下载 Stanford 公开源数组，并发布为固定 data 目录。"""
    target_dir = (stanford_dataset_dir() if dataset_dir is None else Path(dataset_dir)) / "data"
    if target_dir.is_dir() and all((target_dir / name).is_file() for name in REFERENCE_DATA_FILENAMES):
        return target_dir
    if target_dir.exists():
        raise FileExistsError(f"Stanford 源数据目录不完整：{target_dir}")
    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stanford_download_", dir=parent) as temp_name:
        temp_dir = Path(temp_name)
        archive_path = temp_dir / "reference.zip"
        extract_dir = temp_dir / "extract"
        urlretrieve(REFERENCE_DOWNLOAD_URL, archive_path)
        with ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        candidates = [
            path.parent
            for path in extract_dir.rglob("X_reference.npy")
            if all((path.parent / name).is_file() for name in REFERENCE_DATA_FILENAMES)
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"无法唯一定位 Stanford 源数组，候选数：{len(candidates)}")
        prepared_dir = temp_dir / "data"
        prepared_dir.mkdir()
        for name in REFERENCE_DATA_FILENAMES:
            shutil.copy2(candidates[0] / name, prepared_dir / name)
        _validate_reference_arrays(prepared_dir)
        prepared_dir.replace(target_dir)
    return target_dir


def import_reference_init(dataset_dir: Path | None = None) -> Path:
    """将 Stanford 公开数组导入为 30 个直属菌株 init 目录。"""
    root = stanford_dataset_dir() if dataset_dir is None else Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = root / "data"
    _validate_reference_arrays(source_dir)
    output_dir = root / "init"
    if output_dir.exists():
        raise FileExistsError(f"Stanford init 已存在：{output_dir}")
    wavenumbers = np.load(source_dir / "wavenumbers.npy")
    spectra = np.load(source_dir / "X_reference.npy", mmap_mode="r")
    labels = np.load(source_dir / "y_reference.npy").astype(np.int64, copy=False)
    axis = _ascending_axis(wavenumbers)
    reverse_enable = bool(wavenumbers[0] > wavenumbers[-1])
    with tempfile.TemporaryDirectory(prefix="stanford_import_", dir=root) as temp_name:
        temp_dir = Path(temp_name)
        for label_id, (_genus, isolate) in enumerate(REFERENCE_LABELS):
            target_dir = temp_dir / isolate
            target_dir.mkdir()
            for serial, row_index in enumerate(np.flatnonzero(labels == label_id)):
                intensities = spectra[row_index]
                if reverse_enable:
                    intensities = intensities[::-1]
                write_arc_data(target_dir / f"{isolate}_ref_{serial:04d}.arc_data", axis, intensities, fmt="%.10g")
        _write_import_metadata(temp_dir, axis, labels)
        temp_dir.replace(output_dir)
    _publish_reference_axis(axis[-896:])
    return output_dir


def prepare_stanford_dataset(
    config: StanfordPrepareConfig,
    input_config: InputConfig,
) -> Path:
    """按配置恢复或导入 Stanford init，并按共享轴构建预训练 train。"""
    root = stanford_dataset_dir()
    root.mkdir(parents=True, exist_ok=True)
    init_dir = root / "init"
    if config.download_enable:
        download_reference_data(root)
    if config.import_enable:
        import_reference_init(root)
    if not init_dir.is_dir():
        packed_path = root / "init.npz"
        if not packed_path.is_file():
            raise FileNotFoundError(f"缺少 Stanford init 或 init.npz：{root}")
        unpack_init(packed_path, init_dir)
    train_dir = root / "train"
    if config.rebuild_train_enable or not train_dir.is_dir():
        build_stanford_train(root, input_config=input_config)
    return train_dir


def build_stanford_train(
    dataset_dir: Path | str,
    input_config: InputConfig,
    reference_wavenumbers: np.ndarray | None = None,
) -> Path:
    """按原生参考轴选点并删除坏段，保留 Stanford 30 个叶子类别。"""
    root = Path(dataset_dir)
    init_dir = root / "init"
    output_dir = root / "train"
    if not init_dir.is_dir():
        raise FileNotFoundError(f"缺少 Stanford init：{init_dir}")
    reference_axis = load_reference_wavenumbers() if reference_wavenumbers is None else np.asarray(reference_wavenumbers, dtype=np.float64)
    _validate_reference_axis(reference_axis, input_config)
    valid_mask = build_valid_mask(reference_axis, input_config.bad_bands)
    output_axis = reference_axis if valid_mask is None else reference_axis[valid_mask]
    temp_dir = Path(tempfile.mkdtemp(prefix="stanford_train_", dir=root))
    copied = 0
    try:
        for source_dir, filenames in iter_arc_dirs(init_dir):
            target_dir = temp_dir / source_dir.relative_to(init_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                wavenumbers, intensities = read_arc_data(source_dir / filename)
                selected = _select_reference_points(wavenumbers, reference_axis, source_dir / filename)
                if intensities.size != wavenumbers.size:
                    raise ValueError(f"Stanford 光谱列长度不一致：{source_dir / filename}")
                output_values = intensities[selected]
                if valid_mask is not None:
                    output_values = output_values[valid_mask]
                write_arc_data(target_dir / filename, output_axis, output_values, fmt="%.10g")
                copied += 1
        if not copied:
            raise RuntimeError("Stanford train 构建没有写入任何光谱")
        (temp_dir / "train_build_metadata.json").write_text(
            json.dumps(
                {
                    "spectra": copied,
                    "input_grid_mode": "stanford_transfer",
                    "output_points": int(output_axis.size),
                    "intensity_transform": "native_point_selection_and_bad_band_removal_only",
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        _publish_directory(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir


def _validate_reference_arrays(source_dir: Path) -> None:
    """检查公开数组存在、维度和标签空间与固定数据集契约一致。"""
    paths = [source_dir / name for name in REFERENCE_DATA_FILENAMES]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Stanford 源数组不完整：{source_dir}")
    spectra = np.load(source_dir / "X_reference.npy", mmap_mode="r")
    labels = np.load(source_dir / "y_reference.npy", mmap_mode="r")
    wavenumbers = np.load(source_dir / "wavenumbers.npy", mmap_mode="r")
    if spectra.shape != (60000, 1000) or labels.shape != (60000,) or wavenumbers.shape != (1000,):
        raise ValueError(f"Stanford 源数组形状不符合预期：X={spectra.shape}, y={labels.shape}, wn={wavenumbers.shape}")
    if not np.array_equal(np.unique(labels), np.arange(len(REFERENCE_LABELS))):
        raise ValueError("Stanford 标签空间不符合 30 类固定顺序")


def _ascending_axis(wavenumbers: np.ndarray) -> np.ndarray:
    """将单调原始波数轴规范为升序。"""
    if np.all(np.diff(wavenumbers) > 0):
        return wavenumbers
    if np.all(np.diff(wavenumbers) < 0):
        return wavenumbers[::-1]
    raise ValueError("Stanford 原始波数轴不是单调序列")


def _publish_reference_axis(axis: np.ndarray) -> None:
    """发布固定 Stanford 参考轴，并拒绝覆盖不一致内容。"""
    path = stanford_reference_wavenumbers_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = np.load(path)
        if not np.allclose(existing, axis, rtol=0.0, atol=1e-10):
            raise ValueError(f"已有 Stanford 参考轴不一致：{path}")
        return
    np.save(path, axis)


def _validate_reference_axis(axis: np.ndarray, input_config: InputConfig) -> None:
    """确认选点参考轴与 Stanford 输入规格的点数和单调性一致。"""
    if axis.ndim != 1 or axis.size != input_config.target_points:
        raise ValueError("Stanford 参考轴长度与输入规格不一致")
    if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError("Stanford 参考轴必须是有限严格递增数组")


def _select_reference_points(wavenumbers: np.ndarray, reference_axis: np.ndarray, path: Path) -> np.ndarray:
    """在原始 Stanford 谱中定位共享轴对应点，不插值相邻波数。"""
    if wavenumbers.size < 2 or wavenumbers.min() > reference_axis[0] or wavenumbers.max() < reference_axis[-1]:
        raise ValueError(f"Stanford 光谱不覆盖共享参考轴：{path}")
    right = np.searchsorted(wavenumbers, reference_axis)
    left = np.clip(right - 1, 0, wavenumbers.size - 1)
    right = np.clip(right, 0, wavenumbers.size - 1)
    selected = np.where(np.abs(wavenumbers[right] - reference_axis) < np.abs(wavenumbers[left] - reference_axis), right, left)
    if not np.allclose(wavenumbers[selected], reference_axis, rtol=0.0, atol=1e-3):
        raise ValueError(f"Stanford 光谱与共享参考轴不一致：{path}")
    return selected


def _write_import_metadata(output_dir: Path, axis: np.ndarray, labels: np.ndarray) -> None:
    """写入 Stanford init 的数据来源、波数轴和类别摘要。"""
    payload = {
        "spectra": int(labels.size),
        "points_per_spectrum": int(axis.size),
        "wavenumber_min": float(axis.min()),
        "wavenumber_max": float(axis.max()),
        "classes": [isolate for _genus, isolate in REFERENCE_LABELS],
    }
    (output_dir / "import_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _publish_directory(temp_dir: Path, output_dir: Path) -> None:
    """用已完成的临时目录发布 train，并保留此前产物副本。"""
    backup_dir = None
    if output_dir.exists():
        backup_dir = output_dir.parent / f"{output_dir.name}_previous"
        if backup_dir.exists():
            raise FileExistsError(f"存在待处理的 Stanford train 备份：{backup_dir}")
        output_dir.replace(backup_dir)
    try:
        temp_dir.replace(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise
