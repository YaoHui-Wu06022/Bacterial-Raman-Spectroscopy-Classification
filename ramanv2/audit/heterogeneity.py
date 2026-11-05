"""init 生成前的全库异质性分析。

同时读取 ``shift_data``（cosmic 来源）与 ``extra_data``（补充来源），用与训练一致的
单谱预处理链得到 SNV 特征。输出两类信息：

1. 全库样本级偏离：以同种其它文件夹为参照的余弦、kNN 与粗糙度投票；
2. 夹内结构：二聚类、主成分结构与采集顺序漂移。

本模块只做数值分析，不移动文件、不写流水线状态。
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
from scipy.stats import spearmanr

from ramanv2.common.arc_data import iter_arc_data_paths, read_arc_data
from ramanv2.common.naming import (
    NON_TRAINING_FOLDER_NAMES,
    PIPELINE_RESERVED_FOLDER_NAMES,
    build_natural_key,
    parse_folder_prefix,
)
from ramanv2.core.config import InputConfig
from ramanv2.data.catalog import DatasetMapping, build_prefix_to_genus
from ramanv2.data.config import DataBuildConfig
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.bands import build_valid_mask
from ramanv2.spectra.normalize import normalize_spectrum
from ramanv2.spectra.preprocess import preprocess_single_spectrum


INPUT_CONFIG = InputConfig()
BUILD_CONFIG = DataBuildConfig()
COSMIC_RAY_OPTIONS = BUILD_CONFIG.build_cosmic_ray_options("GN")
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
ROUGHNESS_WINDOW = 31
KNN_K = 5
OUTLIER_Z = 3.0
MIN_REFERENCE_FOLDERS = 2
PREPROCESS_WORKERS = 8
MIN_FOLDER_SAMPLES = 5
MIN_CLUSTER_SIZE = 3
N_CLUSTERS = 2
SUBCLUSTER_RATIO_SPLIT = 0.70
SMEAR_RATIO_SPLIT = 0.85
DRIFT_RHO = 0.5
DRIFT_P = 0.01
TOP_BANDS = 5
BAND_MERGE_GAP = 2
# 自动勾选还需子簇质心确实分开，避免随机二分被当成可剔除子簇。
AUTO_MIN_SEPARATION = 0.05

SPECIAL_FOLDER_NAMES = PIPELINE_RESERVED_FOLDER_NAMES | NON_TRAINING_FOLDER_NAMES
EXTRA_SOURCE_PREFIX = "extra"
COSMIC_SOURCE = "cosmic"
EXTRA_SOURCE = "extra"
VERDICT_TOO_FEW = "样本过少"
VERDICT_NO_CLUSTER = "无清晰子簇"
VERDICT_MINORITY = "少数样本子簇"
VERDICT_SPLIT = "双群分裂"
VERDICT_SMEAR = "整体弥散"
VERDICT_MIXED = "中度异质"

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
    """单条光谱的夹内位置与全库偏离指标。"""

    sample_id: str
    folder_key: str
    date_name: str
    folder_name: str
    source_kind: str
    label: str
    genus: str
    file_name: str
    order: int
    features: np.ndarray
    roughness: float
    cluster: str = "majority"
    distance_to_centroid: float = float("nan")
    distance_rank: int = 0
    pc1: float = 0.0
    pc2: float = 0.0
    cos_z: float = 0.0
    knn_z: float = 0.0
    roughness_z: float = 0.0
    outlier_vote: int = 0
    is_outlier: bool = False
    is_severe: bool = False
    auto_candidate: bool = False

    @property
    def is_review_candidate(self) -> bool:
        """自动候选、样本级奇异或夹内最远 3 条进入人工复核列表。"""
        return self.auto_candidate or self.is_outlier or 0 < self.distance_rank <= 3


@dataclass
class FolderReport:
    """单个文件夹的夹内结构与判定。"""

    folder_key: str
    date_name: str
    folder_name: str
    source_kind: str
    label: str
    genus: str
    n_samples: int
    dispersion: float = float("nan")
    dispersion_trimmed: float = float("nan")
    dispersion_ratio: float = float("nan")
    pc1_ratio: float = float("nan")
    pc2_ratio: float = float("nan")
    minority_count: int = 0
    minority_fraction: float = 0.0
    cluster_separation: float = float("nan")
    minority_mean_distance: float = float("nan")
    majority_mean_distance: float = float("nan")
    drift_rho: float = 0.0
    drift_p: float = 1.0
    drift_flag: bool = False
    verdict: str = VERDICT_TOO_FEW
    recommendation: str = ""
    bands: tuple[tuple[float, float, float], ...] = ()
    majority_mean: np.ndarray | None = None
    majority_low: np.ndarray | None = None
    majority_high: np.ndarray | None = None
    samples: tuple[SampleReport, ...] = field(default_factory=tuple)

    @property
    def auto_enabled(self) -> bool:
        """只有少数样本子簇且质心确实分开时，才自动勾选 minority。"""
        return (
            self.verdict == VERDICT_MINORITY
            and math.isfinite(self.cluster_separation)
            and self.cluster_separation >= AUTO_MIN_SEPARATION
        )

    @property
    def auto_candidate_count(self) -> int:
        return int(sum(1 for sample in self.samples if sample.auto_candidate))


@dataclass(frozen=True)
class HeterogeneityReport:
    """全库分析结果；只读，供界面预览与剔除事务使用。"""

    folders: tuple[FolderReport, ...]
    drift_threshold: float
    generated_at: str
    errors: tuple[str, ...] = ()

    @property
    def sample_count(self) -> int:
        return sum(folder.n_samples for folder in self.folders)

    @property
    def auto_sample_ids(self) -> tuple[str, ...]:
        return tuple(
            sample.sample_id
            for folder in self.folders
            for sample in folder.samples
            if sample.auto_candidate
        )

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
        records: list[dict[str, object]] = []
        for folder in self.folders:
            records.append(
                {
                    "folder_key": folder.folder_key,
                    "label": folder.label,
                    "genus": folder.genus,
                    "source_kind": folder.source_kind,
                    "n_samples": folder.n_samples,
                    "dispersion": folder.dispersion,
                    "dispersion_ratio": folder.dispersion_ratio,
                    "pc1_ratio": folder.pc1_ratio,
                    "minority_count": folder.minority_count,
                    "minority_fraction": folder.minority_fraction,
                    "cluster_separation": folder.cluster_separation,
                    "drift_rho": folder.drift_rho,
                    "drift_flag": folder.drift_flag,
                    "verdict": folder.verdict,
                    "auto_candidate_count": folder.auto_candidate_count,
                    "recommendation": folder.recommendation,
                }
            )
        return records

    def sample_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for folder in self.folders:
            for sample in folder.samples:
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "folder_key": sample.folder_key,
                        "date_name": sample.date_name,
                        "folder_name": sample.folder_name,
                        "source_kind": sample.source_kind,
                        "label": sample.label,
                        "genus": sample.genus,
                        "file_name": sample.file_name,
                        "verdict": folder.verdict,
                        "cluster": sample.cluster,
                        "n_samples": folder.n_samples,
                        "minority_count": folder.minority_count,
                        "minority_fraction": folder.minority_fraction,
                        "cluster_separation": folder.cluster_separation,
                        "dispersion": folder.dispersion,
                        "dispersion_ratio": folder.dispersion_ratio,
                        "drift_rho": folder.drift_rho,
                        "drift_flag": folder.drift_flag,
                        "distance_to_centroid": sample.distance_to_centroid,
                        "distance_rank": sample.distance_rank,
                        "order": sample.order,
                        "pc1": sample.pc1,
                        "pc2": sample.pc2,
                        "cos_z": sample.cos_z,
                        "knn_z": sample.knn_z,
                        "roughness_z": sample.roughness_z,
                        "outlier_vote": sample.outlier_vote,
                        "is_outlier": sample.is_outlier,
                        "is_severe": sample.is_severe,
                        "auto_candidate": sample.auto_candidate,
                        "review_candidate": sample.is_review_candidate,
                    }
                )
        return records


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
            nested = [path for path in folder_dir.rglob("*.arc_data") if path.parent != folder_dir]
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


def preprocess_sample(path: Path) -> tuple[np.ndarray, float] | None:
    """按训练同一链路得到 SNV 特征与粗糙度；无效光谱返回 None。"""
    wavenumbers, intensities = read_arc_data(path)
    axis, corrected, _ = preprocess_single_spectrum(
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
        **COSMIC_RAY_OPTIONS,
    )
    if axis is None or corrected is None:
        return None
    smooth = savgol_filter(
        np.asarray(corrected, dtype=np.float64),
        SMOOTH_WINDOW,
        3,
        mode="interp",
    )
    smooth_snv = normalize_spectrum(smooth, "snv").astype(np.float32)
    roughness = float(
        np.std(
            smooth_snv
            - savgol_filter(
                smooth_snv.astype(np.float64),
                ROUGHNESS_WINDOW,
                3,
                mode="interp",
            )
        )
    )
    return smooth_snv, roughness


def _preprocess_guarded(path: Path) -> tuple[np.ndarray | None, float, str | None]:
    try:
        result = preprocess_sample(path)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        return None, 0.0, f"{path}: {error}"
    if result is None:
        return None, 0.0, f"{path}: 预处理后没有有效数据"
    return result[0], result[1], None


def _build_samples(
    folders: list[AnalysisFolder],
    progress_callback: ProgressCallback | None,
) -> tuple[list[SampleReport], list[str]]:
    jobs = [
        (folder, file_name)
        for folder in folders
        for file_name in folder.file_names
    ]
    results: list[tuple[np.ndarray | None, float, str | None]] = [None] * len(jobs)  # type: ignore[list-item]
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
    order_counts: dict[str, int] = {}
    for (folder, file_name), (features, roughness, error) in zip(jobs, results):
        if error is not None or features is None:
            errors.append(error or f"{folder.folder_key}/{file_name}: 预处理失败")
            continue
        order_counts[folder.folder_key] = order_counts.get(folder.folder_key, 0) + 1
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
                order=order_counts[folder.folder_key],
                features=features,
                roughness=float(roughness),
            )
        )
    if progress_callback is not None:
        progress_callback("预处理光谱", len(jobs), len(jobs))
    return samples, errors


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.where(norms > 0, norms, 1.0)


def _cosine_to_centroid(values: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    normalized = _normalize_rows(np.asarray(values, dtype=np.float64))
    center = np.asarray(centroid, dtype=np.float64)
    center = center / max(float(np.linalg.norm(center)), 1e-12)
    return normalized @ center


def _robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.size == 0:
        return np.zeros_like(values)
    center = float(np.median(reference))
    scale = 1.4826 * float(np.median(np.abs(reference - center)))
    if scale <= 0:
        scale = float(np.std(reference, ddof=1)) if reference.size > 1 else 0.0
    if scale <= 0:
        return np.zeros_like(values)
    return (values - center) / scale


def _knn_cosine_distance(
    pairwise: np.ndarray,
    target_index: np.ndarray,
    reference_index: np.ndarray,
    k: int,
) -> np.ndarray:
    block = pairwise[np.ix_(target_index, reference_index)]
    k_eff = min(k, block.shape[1])
    return np.partition(block, k_eff - 1, axis=1)[:, :k_eff].mean(axis=1)


def _compute_library_metrics(samples: list[SampleReport]) -> None:
    """以同种其它文件夹为参照，写入样本级 cos/kNN/粗糙度 z 与投票。"""
    labels = np.array([sample.label for sample in samples])
    folders = np.array([sample.folder_key for sample in samples])
    features = np.vstack([sample.features for sample in samples]).astype(np.float64)
    roughness = np.array([sample.roughness for sample in samples], dtype=np.float64)
    cos_values = np.ones(len(samples), dtype=np.float64)
    knn_values = np.zeros(len(samples), dtype=np.float64)
    cos_z = np.zeros(len(samples), dtype=np.float64)
    knn_z = np.zeros(len(samples), dtype=np.float64)
    roughness_z = np.zeros(len(samples), dtype=np.float64)

    for label in np.unique(labels):
        label_index = np.flatnonzero(labels == label)
        label_folders = np.unique(folders[label_index])
        if label_folders.size < MIN_REFERENCE_FOLDERS:
            continue
        normalized = _normalize_rows(features[label_index])
        pairwise = 1.0 - normalized @ normalized.T
        np.fill_diagonal(pairwise, np.inf)
        for folder in label_folders:
            target_index = np.flatnonzero(folders[label_index] == folder)
            reference_index = np.setdiff1d(
                np.arange(label_index.size),
                target_index,
                assume_unique=False,
            )
            if reference_index.size < 10:
                continue
            centroid = features[label_index[reference_index]].mean(axis=0)
            cos_values[label_index[target_index]] = _cosine_to_centroid(
                features[label_index[target_index]], centroid
            )
            knn_values[label_index[target_index]] = _knn_cosine_distance(
                pairwise, target_index, reference_index, KNN_K
            )
        cos_z[label_index] = _robust_z(cos_values[label_index], cos_values[label_index])
        knn_z[label_index] = _robust_z(knn_values[label_index], knn_values[label_index])
        roughness_z[label_index] = _robust_z(
            roughness[label_index], roughness[label_index]
        )

    for index, sample in enumerate(samples):
        sample.cos_z = float(cos_z[index])
        sample.knn_z = float(knn_z[index])
        sample.roughness_z = float(roughness_z[index])
        vote = (
            int(cos_z[index] <= -OUTLIER_Z)
            + int(knn_z[index] >= OUTLIER_Z)
            + int(roughness_z[index] >= OUTLIER_Z)
        )
        sample.outlier_vote = vote
        sample.is_outlier = vote >= 2
        sample.is_severe = vote >= 3


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


def _compute_pc_scores(features: np.ndarray) -> tuple[np.ndarray, float, float]:
    centered = features - features.mean(axis=0)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    scores = left * singular_values
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 0:
        return scores, 0.0, 0.0
    return scores, float(energy[0] / total), float(energy[1] / total)


def _analyze_folder_structure(
    folder: AnalysisFolder,
    samples: list[SampleReport],
) -> FolderReport:
    report = FolderReport(
        folder_key=folder.folder_key,
        date_name=folder.date_name,
        folder_name=folder.folder_name,
        source_kind=folder.source_kind,
        label=folder.label,
        genus=folder.genus,
        n_samples=len(samples),
        samples=tuple(samples),
    )
    if len(samples) < MIN_FOLDER_SAMPLES:
        report.verdict = VERDICT_TOO_FEW
        report.recommendation = "样本不足，需人工判断"
        return report

    features = np.vstack([sample.features for sample in samples]).astype(np.float64)
    count = features.shape[0]
    normalized = _normalize_rows(features)
    pairwise = np.clip(1.0 - normalized @ normalized.T, 0.0, None)
    np.fill_diagonal(pairwise, 0.0)
    upper = pairwise[np.triu_indices(count, k=1)]
    report.dispersion = float(upper.mean()) if upper.size else 0.0

    centroid = features.mean(axis=0)
    centroid_unit = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    distance_to_centroid = 1.0 - normalized @ centroid_unit
    keep_count = max(int(np.ceil(0.9 * count)), 3)
    keep_index = np.argsort(distance_to_centroid)[:keep_count]
    trimmed = pairwise[np.ix_(keep_index, keep_index)]
    trimmed_upper = trimmed[np.triu_indices(keep_count, k=1)]
    report.dispersion_trimmed = (
        float(trimmed_upper.mean()) if trimmed_upper.size else report.dispersion
    )
    report.dispersion_ratio = (
        report.dispersion_trimmed / report.dispersion
        if report.dispersion > 0
        else 1.0
    )

    labels = _split_clusters(pairwise, distance_to_centroid)
    scores, report.pc1_ratio, report.pc2_ratio = _compute_pc_scores(features)
    if labels.max() > 0 and (
        scores[labels == 1, 0].mean() > scores[labels == 0, 0].mean()
    ):
        scores[:, 0] *= -1.0

    ranks = distance_to_centroid.argsort().argsort()
    for index, sample in enumerate(samples):
        sample.cluster = "minority" if labels[index] == 1 else "majority"
        sample.distance_to_centroid = float(distance_to_centroid[index])
        sample.distance_rank = int(count - ranks[index])
        sample.pc1 = float(scores[index, 0])
        sample.pc2 = float(scores[index, 1])

    report.minority_count = int((labels == 1).sum())
    report.minority_fraction = float(report.minority_count / count)
    if labels.max() > 0:
        majority_features = features[labels == 0]
        minority_features = features[labels == 1]
        majority_centroid = majority_features.mean(axis=0)
        minority_centroid = minority_features.mean(axis=0)
        report.cluster_separation = 1.0 - float(
            np.dot(
                _normalize_rows(minority_centroid.reshape(1, -1)).ravel(),
                _normalize_rows(majority_centroid.reshape(1, -1)).ravel(),
            )
        )
        report.minority_mean_distance = float(
            distance_to_centroid[labels == 1].mean()
        )
        report.majority_mean_distance = float(
            distance_to_centroid[labels == 0].mean()
        )
        difference = minority_centroid - majority_centroid
        top_index = np.argsort(-np.abs(difference))[: max(TOP_BANDS * 3, 12)]
        report.bands = _merge_top_bands(top_index, FEATURE_AXIS, difference)
        report.majority_mean = majority_features.mean(axis=0)
        report.majority_low = np.percentile(majority_features, 25, axis=0)
        report.majority_high = np.percentile(majority_features, 75, axis=0)
    else:
        report.majority_mean = features.mean(axis=0)
        report.majority_low = np.percentile(features, 25, axis=0)
        report.majority_high = np.percentile(features, 75, axis=0)

    rho, p_value = spearmanr(np.arange(count, dtype=np.float64), scores[:, 0])
    report.drift_rho = float(rho) if np.isfinite(rho) else 0.0
    report.drift_p = float(p_value) if np.isfinite(p_value) else 1.0
    return report


def _classify_folder(report: FolderReport) -> str:
    if report.n_samples < MIN_FOLDER_SAMPLES:
        return VERDICT_TOO_FEW
    if report.minority_count == 0:
        return VERDICT_NO_CLUSTER
    ratio = report.dispersion_ratio
    fraction = report.minority_fraction
    separation = report.cluster_separation
    if fraction <= 0.20 or ratio <= SUBCLUSTER_RATIO_SPLIT:
        return VERDICT_MINORITY
    if fraction >= 0.30 and separation >= 0.10:
        return VERDICT_SPLIT
    if ratio >= SMEAR_RATIO_SPLIT:
        return VERDICT_SMEAR
    return VERDICT_MIXED


def _recommend_action(report: FolderReport) -> str:
    if report.verdict == VERDICT_MINORITY:
        return "优先复核 minority 子簇样本"
    if report.verdict == VERDICT_SPLIT:
        if report.drift_flag:
            return "两组沿采集顺序分段，按时间段分批复核"
        return "两组规模相近，需确认是否不同批次或来源"
    if report.verdict == VERDICT_SMEAR:
        if report.drift_flag:
            return "疑似采集顺序漂移，按顺序分段复核"
        return "无明确剔除对象，保留或整夹重采"
    if report.verdict == VERDICT_MIXED:
        return "抽查最远样本"
    if report.verdict == VERDICT_NO_CLUSTER:
        return "按样本级奇异投票抽查" if not report.drift_flag else "沿采集顺序分段复核"
    return "样本不足，需人工判断"


def analyze_library(
    shift_root: Path | str,
    supplement_root: Path | str,
    mapping: DatasetMapping,
    *,
    date_names: tuple[str, ...] | list[str] | set[str] | None = None,
    cosmic_folder_keys: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HeterogeneityReport:
    """扫描、预处理并分析全库；返回只读结果与跳过记录。"""
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
    if progress_callback is not None:
        progress_callback("计算全库同种偏离", 0, 1)
    _compute_library_metrics(samples)
    if progress_callback is not None:
        progress_callback("计算全库同种偏离", 1, 1)

    samples_by_folder: dict[str, list[SampleReport]] = {}
    for sample in samples:
        samples_by_folder.setdefault(sample.folder_key, []).append(sample)
    folder_reports: list[FolderReport] = []
    total = len(folders)
    for index, folder in enumerate(folders, start=1):
        folder_samples = samples_by_folder.get(folder.folder_key, [])
        report = _analyze_folder_structure(folder, folder_samples)
        folder_reports.append(report)
        if progress_callback is not None:
            progress_callback("分析夹内结构", index, total)

    rhos = np.array(
        [
            abs(report.drift_rho)
            for report in folder_reports
            if report.n_samples >= MIN_FOLDER_SAMPLES
        ],
        dtype=np.float64,
    )
    drift_threshold = DRIFT_RHO
    if rhos.size:
        drift_threshold = max(DRIFT_RHO, float(np.percentile(rhos, 90)))
    for report in folder_reports:
        report.drift_flag = (
            abs(report.drift_rho) >= drift_threshold and report.drift_p < DRIFT_P
        )
        report.verdict = _classify_folder(report)
        report.recommendation = _recommend_action(report)
        for sample in report.samples:
            sample.auto_candidate = (
                report.auto_enabled and sample.cluster == "minority"
            )

    return HeterogeneityReport(
        folders=tuple(folder_reports),
        drift_threshold=drift_threshold,
        generated_at=datetime.now(timezone.utc).isoformat(),
        errors=tuple(errors),
    )


__all__ = [
    "AUTO_MIN_SEPARATION",
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
