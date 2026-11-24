"""init 生成前的全库异质性分析。

只识别一种可自动剔除的情形：来源文件夹内部存在一个可分的少数样本子簇。
同时读取 ``shift_data``（cosmic 来源）与 ``extra_data``（补充来源），用与训练一致的
单谱预处理链得到 SNV 特征，对每个文件夹做 average-linkage 二聚类；只保留
“少数簇占比 ≤ 20% 或去除最远 10% 后离散度比 ≤ 0.70，且两簇质心分离度 ≥ 0.05”
的文件夹。本模块只做数值分析，不移动文件、不写流水线状态。
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform

from ramanv2.common.arc_data import iter_arc_data_paths, read_arc_data
from ramanv2.common.naming import (
    SPECIAL_FOLDER_NAMES,
    build_natural_key,
    parse_folder_prefix,
)
from ramanv2.core.config import InputConfig
from ramanv2.data.dataset_mapping import DatasetMapping, build_prefix_to_genus
from ramanv2.data.config import DataBuildConfig
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.bad_bands import build_valid_mask
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import preprocess_single_spectrum


INPUT_CONFIG = InputConfig()
BUILD_CONFIG = DataBuildConfig()
REFERENCE_AXIS = build_wn_ref(
    INPUT_CONFIG.cut_min,
    INPUT_CONFIG.cut_max,
    INPUT_CONFIG.target_points,
)
_REFERENCE_MASK = build_valid_mask(REFERENCE_AXIS, INPUT_CONFIG.bad_bands)
# 特征轴为去掉坏波段后的有效点，与模型输入一致。
FEATURE_AXIS = (
    REFERENCE_AXIS if _REFERENCE_MASK is None else REFERENCE_AXIS[_REFERENCE_MASK]
)

SMOOTH_WINDOW = 15
PREPROCESS_WORKERS = 8
MIN_FOLDER_SAMPLES = 5
MIN_CLUSTER_SIZE = 3
N_CLUSTERS = 2
MINORITY_FRACTION_SPLIT = 0.20
SUBCLUSTER_RATIO_SPLIT = 0.70
# 少数簇质心必须与多数簇确实分开，避免随机二分被当成可剔除子簇。
AUTO_MIN_SEPARATION = 0.05
TOP_BANDS = 5
BAND_MERGE_GAP = 2

EXTRA_SOURCE_PREFIX = "extra"
COSMIC_SOURCE = "cosmic"
EXTRA_SOURCE = "extra"
VERDICT_MINORITY = "少数样本子簇"

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class AnalysisFolder:
    """一个待分析的来源文件夹；source_dir 指向实际读取的光谱目录。"""

    folder_key: str
    date_name: str
    folder_name: str
    source_kind: str
    source_dir: Path
    label: str
    genus: str
    file_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return len(self.file_names)


@dataclass
class SampleReport:
    """单条光谱在夹内的子簇归属与到质心距离。"""

    sample_id: str
    folder_key: str
    date_name: str
    folder_name: str
    source_kind: str
    label: str
    genus: str
    file_name: str
    features: np.ndarray
    cluster: str = "majority"
    distance_to_centroid: float = float("nan")
    auto_candidate: bool = False


@dataclass
class FolderReport:
    """单个入选文件夹的夹内结构；只保存少数样本子簇成立的结果。"""

    folder_key: str
    date_name: str
    folder_name: str
    source_kind: str
    label: str
    genus: str
    n_samples: int
    dispersion: float
    dispersion_trimmed: float
    dispersion_ratio: float
    minority_count: int
    minority_fraction: float
    cluster_separation: float
    bands: tuple[tuple[float, float, float], ...] = ()
    majority_mean: np.ndarray | None = None
    majority_low: np.ndarray | None = None
    majority_high: np.ndarray | None = None
    samples: tuple[SampleReport, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        return VERDICT_MINORITY

    @property
    def auto_candidate_count(self) -> int:
        return int(sum(1 for sample in self.samples if sample.auto_candidate))


@dataclass(frozen=True)
class HeterogeneityReport:
    """全库分析结果；只包含少数样本子簇成立的文件夹。"""

    folders: tuple[FolderReport, ...]
    generated_at: str
    errors: tuple[str, ...] = ()

    @property
    def sample_count(self) -> int:
        return sum(folder.n_samples for folder in self.folders)

    @property
    def folders_by_key(self) -> dict[str, FolderReport]:
        return {folder.folder_key: folder for folder in self.folders}

    @property
    def samples_by_id(self) -> dict[str, SampleReport]:
        return {
            sample.sample_id: sample
            for folder in self.folders
            for sample in folder.samples
        }

    def folder_records(self) -> list[dict[str, object]]:
        return [
            {
                "folder_key": folder.folder_key,
                "label": folder.label,
                "genus": folder.genus,
                "source_kind": folder.source_kind,
                "n_samples": folder.n_samples,
                "minority_count": folder.minority_count,
                "minority_fraction": folder.minority_fraction,
                "cluster_separation": folder.cluster_separation,
                "dispersion": folder.dispersion,
                "dispersion_ratio": folder.dispersion_ratio,
                "auto_candidate_count": folder.auto_candidate_count,
            }
            for folder in self.folders
        ]

    def sample_records(self) -> list[dict[str, object]]:
        return [
            {
                "sample_id": sample.sample_id,
                "folder_key": sample.folder_key,
                "date_name": sample.date_name,
                "folder_name": sample.folder_name,
                "source_kind": sample.source_kind,
                "label": sample.label,
                "genus": sample.genus,
                "file_name": sample.file_name,
                "cluster": sample.cluster,
                "distance_to_centroid": sample.distance_to_centroid,
                "auto_candidate": sample.auto_candidate,
            }
            for folder in self.folders
            for sample in folder.samples
        ]


def _parse_date(folder_name: str) -> str:
    match = re.search(r"(\d{8})", str(folder_name))
    return match.group(1) if match else ""


def _collect_arc_files(folder: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            (path.name for path in iter_arc_data_paths(folder)),
            key=build_natural_key,
        )
    )


def scan_analysis_folders(
    shift_root: Path | str,
    supplement_root: Path | str,
    mapping: DatasetMapping,
    date_names: tuple[str, ...] | list[str] | set[str] | None = None,
    cosmic_folder_keys: set[str] | None = None,
) -> list[AnalysisFolder]:
    """扫描 shift_data 与 extra_data，生成待分析文件夹清单。

    ``cosmic_folder_keys`` 用于排除 shift_data 中残留的、cosmic 已不存在的文件夹。
    """
    shift = Path(shift_root).resolve()
    supplement = Path(supplement_root).resolve()
    prefix_to_genus = build_prefix_to_genus(mapping)
    selected_dates = None if date_names is None else {str(name) for name in date_names}
    folders: list[AnalysisFolder] = []

    if shift.is_dir():
        for date_dir in sorted(
            (
                path
                for path in shift.iterdir()
                if path.is_dir() and path.name.isdigit() and len(path.name) == 8
            ),
            key=lambda path: build_natural_key(path.name),
        ):
            if selected_dates is not None and date_dir.name not in selected_dates:
                continue
            for folder_dir in sorted(
                (path for path in date_dir.iterdir() if path.is_dir()),
                key=lambda path: build_natural_key(path.name),
            ):
                if folder_dir.name in SPECIAL_FOLDER_NAMES or folder_dir.name.startswith("."):
                    continue
                files = _collect_arc_files(folder_dir)
                if not files:
                    continue
                label = parse_folder_prefix(folder_dir.name, uppercase_enable=True)
                genus = prefix_to_genus.get(label)
                if genus is None:
                    raise ValueError(f"shift 前缀没有属映射：{label}（{folder_dir}）")
                folder_key = f"{date_dir.name}/{folder_dir.name}"
                if cosmic_folder_keys is not None and folder_key not in cosmic_folder_keys:
                    continue
                folders.append(
                    AnalysisFolder(
                        folder_key=folder_key,
                        date_name=date_dir.name,
                        folder_name=folder_dir.name,
                        source_kind=COSMIC_SOURCE,
                        source_dir=folder_dir,
                        label=label,
                        genus=genus,
                        file_names=files,
                    )
                )

    if supplement.is_dir():
        for folder_dir in sorted(
            (
                path
                for path in supplement.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            key=lambda path: build_natural_key(path.name),
        ):
            nested = [
                path
                for path in iter_arc_data_paths(folder_dir, recursive=True)
                if path.parent != folder_dir
            ]
            if nested:
                raise ValueError(f"补充文件夹只支持单层光谱：{folder_dir}")
            files = _collect_arc_files(folder_dir)
            if not files:
                continue
            label = parse_folder_prefix(folder_dir.name, uppercase_enable=True)
            genus = prefix_to_genus.get(label)
            if genus is None:
                raise ValueError(f"补充前缀没有属映射：{label}（{folder_dir}）")
            date_name = _parse_date(folder_dir.name) or EXTRA_SOURCE_PREFIX
            folders.append(
                AnalysisFolder(
                    folder_key=f"{EXTRA_SOURCE_PREFIX}/{folder_dir.name}",
                    date_name=date_name,
                    folder_name=folder_dir.name,
                    source_kind=EXTRA_SOURCE,
                    source_dir=folder_dir,
                    label=label,
                    genus=genus,
                    file_names=files,
                )
            )
    return folders


def preprocess_sample(path: Path) -> np.ndarray | None:
    """按训练同一链路得到 SNV 特征；无效光谱返回 None。"""
    wavenumbers, intensities = read_arc_data(path)
    axis, corrected = preprocess_single_spectrum(
        wavenumbers,
        intensities,
        cut_min=INPUT_CONFIG.cut_min,
        cut_max=INPUT_CONFIG.cut_max,
        reference_wavenumbers=REFERENCE_AXIS,
        bad_bands=INPUT_CONFIG.bad_bands,
        baseline_lam=BUILD_CONFIG.baseline_lam,
        baseline_asls_p=BUILD_CONFIG.baseline_asls_p,
        baseline_max_iter=BUILD_CONFIG.baseline_max_iter,
        baseline_fit_min=BUILD_CONFIG.baseline_fit_min,
        baseline_fit_max=BUILD_CONFIG.baseline_fit_max,
        baseline_method=BUILD_CONFIG.baseline_method,
    )
    if axis is None or corrected is None:
        return None
    smooth = savgol_filter(
        np.asarray(corrected, dtype=np.float64),
        SMOOTH_WINDOW,
        3,
        mode="interp",
    )
    return normalize_spectrum(smooth, "snv").astype(np.float32)


def _preprocess_guarded(path: Path) -> tuple[np.ndarray | None, str | None]:
    try:
        features = preprocess_sample(path)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        return None, f"{path}: {error}"
    if features is None:
        return None, f"{path}: 预处理后没有有效数据"
    return features, None


def _build_samples(
    folders: list[AnalysisFolder],
    progress_callback: ProgressCallback | None,
) -> tuple[list[SampleReport], list[str]]:
    jobs = [
        (folder, file_name)
        for folder in folders
        for file_name in folder.file_names
    ]
    results: list[tuple[np.ndarray | None, str | None]] = [None] * len(jobs)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        futures = {
            executor.submit(_preprocess_guarded, folder.source_dir / file_name): index
            for index, (folder, file_name) in enumerate(jobs)
        }
        for done, (future, index) in enumerate(futures.items(), start=1):
            results[index] = future.result()
            if progress_callback is not None and done % 50 == 0:
                progress_callback("预处理光谱", done, len(jobs))

    samples: list[SampleReport] = []
    errors: list[str] = []
    for (folder, file_name), (features, error) in zip(jobs, results):
        if error is not None or features is None:
            errors.append(error or f"{folder.folder_key}/{file_name}: 预处理失败")
            continue
        samples.append(
            SampleReport(
                sample_id=f"{folder.folder_key}/{file_name}",
                folder_key=folder.folder_key,
                date_name=folder.date_name,
                folder_name=folder.folder_name,
                source_kind=folder.source_kind,
                label=folder.label,
                genus=folder.genus,
                file_name=file_name,
                features=features,
            )
        )
    if progress_callback is not None:
        progress_callback("预处理光谱", len(jobs), len(jobs))
    return samples, errors


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.where(norms > 0, norms, 1.0)


def _split_clusters(
    pairwise: np.ndarray,
    distance_to_centroid: np.ndarray,
) -> np.ndarray:
    """average-linkage 二聚类；退化时全部标记为同一簇。"""
    count = pairwise.shape[0]
    if count < N_CLUSTERS * MIN_CLUSTER_SIZE:
        return np.zeros(count, dtype=np.int64)
    linkage_matrix = linkage(squareform(pairwise, checks=False), method="average")
    raw_labels = fcluster(linkage_matrix, t=N_CLUSTERS, criterion="maxclust")
    counts = np.bincount(raw_labels)
    effective = counts[counts > 0]
    if effective.size < N_CLUSTERS or int(effective.min()) < MIN_CLUSTER_SIZE:
        return np.zeros(count, dtype=np.int64)
    mean_distances = {
        int(label): float(distance_to_centroid[raw_labels == label].mean())
        for label in np.unique(raw_labels)
    }
    minority_raw = min(
        mean_distances,
        key=lambda label: (
            int((raw_labels == label).sum()),
            -mean_distances[label],
        ),
    )
    return (raw_labels == minority_raw).astype(np.int64)


def _merge_top_bands(
    top_index: np.ndarray,
    axis: np.ndarray,
    difference: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    ordered = np.sort(top_index)
    groups: list[tuple[int, int]] = []
    start = previous = int(ordered[0])
    for current in ordered[1:]:
        current = int(current)
        if current - previous <= BAND_MERGE_GAP:
            previous = current
            continue
        groups.append((start, previous))
        start = previous = current
    groups.append((start, previous))
    bands = [
        (
            float(axis[low]),
            float(axis[high]),
            float(difference[low : high + 1].mean()),
        )
        for low, high in groups
    ]
    bands.sort(key=lambda item: -abs(item[2]))
    return tuple(bands[:TOP_BANDS])


def _is_minority_subcluster(
    minority_count: int,
    minority_fraction: float,
    dispersion_ratio: float,
    cluster_separation: float,
) -> bool:
    """少数簇占比或离散度比达标，且质心确实分开时才成立。"""
    if minority_count == 0:
        return False
    if not (
        minority_fraction <= MINORITY_FRACTION_SPLIT
        or dispersion_ratio <= SUBCLUSTER_RATIO_SPLIT
    ):
        return False
    return math.isfinite(cluster_separation) and cluster_separation >= AUTO_MIN_SEPARATION


def _analyze_folder_structure(
    folder: AnalysisFolder,
    samples: list[SampleReport],
) -> FolderReport | None:
    """分析单个文件夹；少数样本子簇不成立时返回 None。"""
    if len(samples) < MIN_FOLDER_SAMPLES:
        return None

    features = np.vstack([sample.features for sample in samples]).astype(np.float64)
    count = features.shape[0]
    normalized = _normalize_rows(features)
    pairwise = np.clip(1.0 - normalized @ normalized.T, 0.0, None)
    np.fill_diagonal(pairwise, 0.0)
    upper = pairwise[np.triu_indices(count, k=1)]
    dispersion = float(upper.mean()) if upper.size else 0.0

    centroid = features.mean(axis=0)
    centroid_unit = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    distance_to_centroid = 1.0 - normalized @ centroid_unit
    keep_count = max(int(np.ceil(0.9 * count)), 3)
    keep_index = np.argsort(distance_to_centroid)[:keep_count]
    trimmed = pairwise[np.ix_(keep_index, keep_index)]
    trimmed_upper = trimmed[np.triu_indices(keep_count, k=1)]
    dispersion_trimmed = float(trimmed_upper.mean()) if trimmed_upper.size else dispersion
    dispersion_ratio = (
        dispersion_trimmed / dispersion if dispersion > 0 else 1.0
    )

    labels = _split_clusters(pairwise, distance_to_centroid)
    if labels.max() == 0:
        return None
    minority_count = int(labels.sum())
    minority_fraction = float(minority_count / count)
    majority_centroid = features[labels == 0].mean(axis=0)
    minority_centroid = features[labels == 1].mean(axis=0)
    cluster_separation = 1.0 - float(
        np.dot(
            _normalize_rows(minority_centroid.reshape(1, -1)).ravel(),
            _normalize_rows(majority_centroid.reshape(1, -1)).ravel(),
        )
    )
    if not _is_minority_subcluster(
        minority_count,
        minority_fraction,
        dispersion_ratio,
        cluster_separation,
    ):
        return None

    for index, sample in enumerate(samples):
        sample.cluster = "minority" if labels[index] == 1 else "majority"
        sample.distance_to_centroid = float(distance_to_centroid[index])
        sample.auto_candidate = sample.cluster == "minority"

    difference = minority_centroid - majority_centroid
    top_index = np.argsort(-np.abs(difference))[: max(TOP_BANDS * 3, 12)]
    return FolderReport(
        folder_key=folder.folder_key,
        date_name=folder.date_name,
        folder_name=folder.folder_name,
        source_kind=folder.source_kind,
        label=folder.label,
        genus=folder.genus,
        n_samples=count,
        dispersion=dispersion,
        dispersion_trimmed=dispersion_trimmed,
        dispersion_ratio=dispersion_ratio,
        minority_count=minority_count,
        minority_fraction=minority_fraction,
        cluster_separation=cluster_separation,
        bands=_merge_top_bands(top_index, FEATURE_AXIS, difference),
        majority_mean=features[labels == 0].mean(axis=0),
        majority_low=np.percentile(features[labels == 0], 25, axis=0),
        majority_high=np.percentile(features[labels == 0], 75, axis=0),
        samples=tuple(samples),
    )


def analyze_library(
    shift_root: Path | str,
    supplement_root: Path | str,
    mapping: DatasetMapping,
    *,
    date_names: tuple[str, ...] | list[str] | set[str] | None = None,
    cosmic_folder_keys: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HeterogeneityReport:
    """扫描、预处理并分析全库；只返回少数样本子簇成立的文件夹。"""
    if progress_callback is not None:
        progress_callback("扫描来源文件夹", 0, 1)
    folders = scan_analysis_folders(
        shift_root,
        supplement_root,
        mapping,
        date_names,
        cosmic_folder_keys,
    )
    if progress_callback is not None:
        progress_callback("扫描来源文件夹", 1, 1)
    samples, errors = _build_samples(folders, progress_callback)

    samples_by_folder: dict[str, list[SampleReport]] = {}
    for sample in samples:
        samples_by_folder.setdefault(sample.folder_key, []).append(sample)
    folder_reports: list[FolderReport] = []
    total = len(folders)
    for index, folder in enumerate(folders, start=1):
        report = _analyze_folder_structure(
            folder,
            samples_by_folder.get(folder.folder_key, []),
        )
        if report is not None:
            folder_reports.append(report)
        if progress_callback is not None:
            progress_callback("分析夹内结构", index, total)

    return HeterogeneityReport(
        folders=tuple(folder_reports),
        generated_at=datetime.now(timezone.utc).isoformat(),
        errors=tuple(errors),
    )


__all__ = [
    "AnalysisFolder",
    "EXTRA_SOURCE",
    "EXTRA_SOURCE_PREFIX",
    "FEATURE_AXIS",
    "FolderReport",
    "HeterogeneityReport",
    "MIN_FOLDER_SAMPLES",
    "SampleReport",
    "analyze_library",
    "preprocess_sample",
    "scan_analysis_folders",
]
