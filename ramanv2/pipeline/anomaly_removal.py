"""init 生成前的全库高异常剔除事务。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np

from ramanv2.audit.heterogeneity import (
    EXTRA_SOURCE,
    FEATURE_AXIS,
    FolderReport,
    HeterogeneityReport,
    SampleReport,
)
from ramanv2.common.arc_data import count_arc_data, iter_arc_data_paths
from ramanv2.common.naming import NON_TRAINING_FOLDER_NAMES, build_natural_key
from ramanv2.common.file_ops import (
    move_files,
    preserve_paths,
    remove_empty_parent_directories,
    write_text_atomic,
)
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.pipeline.cleaning import scan_date_names, scan_review_folders
from ramanv2.pipeline.invalidation import invalidate_downstream
from ramanv2.pipeline.lineage import (
    ANALYSIS_DELETE_RELATIVE,
    lineage_path,
    record_analysis_removal,
)
from ramanv2.pipeline.shift import load_date_affine_parameters
from ramanv2.pipeline.state import (
    COMMIT_LOG_NAME,
    PipelineState,
    write_pipeline_state,
)
from ramanv2.pipeline.supplement import scan_supplement_plan


ANALYSIS_REMOVAL_LOG_NAME = "analysis_removal_log.jsonl"
ANALYSIS_REMOVAL_REPORT_NAME = "analysis_removal_report.md"
ANALYSIS_REMOVAL_REASON = "analysis_subcluster_removal"
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class AnalysisGate:
    """分析入口的门禁结果；issues 非空时禁止分析。"""

    ready_enable: bool
    issues: tuple[str, ...]
    dates: tuple[str, ...]
    cosmic_folder_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RemovalTarget:
    """一条待归档样本的源路径与归档路径。"""

    sample: SampleReport
    folder: FolderReport
    archive_path: Path
    cosmic_source: Path | None
    shift_source: Path | None
    extra_source: Path | None


def _arc_data_names(folder: Path) -> set[str]:
    """列出文件夹的直接光谱名。"""
    return {path.name for path in iter_arc_data_paths(folder)}


def check_analysis_gate(
    data_root: Path | str,
    shift_root: Path | str,
    supplement_root: Path | str,
    init_root: Path | str,
    mapping: DatasetMapping,
    folder_catalog: dict[str, set[str]],
    completed_folder_keys: set[str],
) -> AnalysisGate:
    """检查清洗进度与 shift 可用性，不读取光谱内容。"""
    root = Path(data_root).resolve()
    shift = Path(shift_root).resolve()
    issues: list[str] = []
    try:
        dates = scan_date_names(root)
    except FileNotFoundError as error:
        return AnalysisGate(False, (str(error),), ())

    remaining = [
        folder
        for folder in scan_review_folders(root, dates, folder_catalog, completed_folder_keys)
        if folder.folder_name not in NON_TRAINING_FOLDER_NAMES
    ]
    if remaining:
        names = "、".join(folder.key for folder in remaining[:5])
        issues.append(f"仍有 {len(remaining)} 个文件夹未完成审核：{names}")

    dates_with_folders: list[str] = []
    cosmic_folder_keys: list[str] = []
    for date_name in dates:
        date_dir = root / date_name
        active_folders: list[tuple[Path, set[str]]] = []
        for folder in date_dir.iterdir():
            if not folder.is_dir() or folder.name == "小球" or folder.name.startswith("."):
                continue
            cosmic_names = _arc_data_names(folder)
            if cosmic_names:
                active_folders.append((folder, cosmic_names))
        if not active_folders:
            continue
        dates_with_folders.append(date_name)
        cosmic_folder_keys.extend(
            f"{date_name}/{folder.name}" for folder, _ in active_folders
        )
        try:
            parameters = load_date_affine_parameters(shift, date_name)
        except (OSError, ValueError, TypeError) as error:
            issues.append(f"{date_name}：{error}")
            continue
        if parameters is None:
            issues.append(f"{date_name}：shift_data 尚未生成或仿射参数未批准")
            continue
        for folder, cosmic_names in sorted(
            active_folders, key=lambda item: build_natural_key(item[0].name)
        ):
            shift_names = _arc_data_names(shift / date_name / folder.name)
            if cosmic_names != shift_names:
                issues.append(
                    f"{date_name}/{folder.name}：shift 与 cosmic 文件不一致"
                    f"（cosmic {len(cosmic_names)} / shift {len(shift_names)}）"
                )

    try:
        plan = scan_supplement_plan(supplement_root, init_root, mapping)
    except (OSError, ValueError, TypeError) as error:
        issues.append(f"补充数据扫描失败：{error}")
        plan = None
    if plan is not None:
        issues.extend(plan.errors)

    if not dates_with_folders and not (plan is not None and plan.groups):
        issues.append("没有可分析的光谱来源")
    return AnalysisGate(
        not issues,
        tuple(issues),
        tuple(dates_with_folders),
        tuple(sorted(cosmic_folder_keys)),
    )


def build_removal_targets(
    data_root: Path | str,
    shift_root: Path | str,
    supplement_root: Path | str,
    report: HeterogeneityReport,
    sample_ids: list[str] | tuple[str, ...] | set[str],
) -> list[RemovalTarget]:
    """校验选中样本并生成归档路径；任何冲突都在移动前报错。"""
    root = Path(data_root).resolve()
    shift = Path(shift_root).resolve()
    supplement = Path(supplement_root).resolve()
    dataset_root = root.parent
    archive_root = dataset_root / ANALYSIS_DELETE_RELATIVE
    samples = report.samples_by_id
    folders = report.folders_by_key
    targets: list[RemovalTarget] = []
    seen: set[str] = set()
    for sample_id in sorted(set(sample_ids), key=build_natural_key):
        if sample_id in seen:
            continue
        seen.add(sample_id)
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"未知样本：{sample_id}")
        folder = folders[sample.folder_key]
        archive_path = (
            archive_root
            / sample.date_name
            / sample.folder_name
            / sample.file_name
        )
        if sample.source_kind == EXTRA_SOURCE:
            source = supplement / sample.folder_name / sample.file_name
            targets.append(
                RemovalTarget(
                    sample=sample,
                    folder=folder,
                    archive_path=archive_path,
                    cosmic_source=None,
                    shift_source=None,
                    extra_source=source,
                )
            )
        else:
            targets.append(
                RemovalTarget(
                    sample=sample,
                    folder=folder,
                    archive_path=archive_path,
                    cosmic_source=root / sample.date_name / sample.folder_name / sample.file_name,
                    shift_source=shift / sample.date_name / sample.folder_name / sample.file_name,
                    extra_source=None,
                )
            )
    if not targets:
        raise ValueError("没有选中任何样本")
    for target in targets:
        for source in (target.cosmic_source, target.shift_source, target.extra_source):
            if source is not None and not source.is_file():
                raise FileNotFoundError(f"源文件不存在：{source}")
        if target.archive_path.exists():
            raise FileExistsError(f"归档目标已存在：{target.archive_path}")
        figure_path = target.archive_path.with_suffix(".png")
        if figure_path.exists():
            raise FileExistsError(f"证据图目标已存在：{figure_path}")
    return targets


def _insert_axis_gap(
    axis: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gaps = np.flatnonzero(np.diff(axis) > 5.0) + 1
    if gaps.size == 0:
        return axis, values
    axis_with_gap = np.insert(axis, gaps, np.nan)
    return axis_with_gap, np.insert(values, gaps, np.nan, axis=-1)


def _plot_evidence(
    folder: FolderReport,
    sample: SampleReport,
    output_path: Path,
) -> None:
    """上图为多数样本均值与 IQR 对照，下图为该样本的差值谱。"""
    if folder.majority_mean is None or folder.majority_low is None or folder.majority_high is None:
        raise ValueError(f"{sample.sample_id} 缺少多数样本统计")
    x_axis, majority_curves = _insert_axis_gap(
        FEATURE_AXIS,
        np.vstack([folder.majority_mean, folder.majority_low, folder.majority_high]),
    )
    _, sample_curve = _insert_axis_gap(FEATURE_AXIS, sample.features)
    _, difference_curve = _insert_axis_gap(
        FEATURE_AXIS,
        sample.features - folder.majority_mean,
    )

    figure, (top_axis, bottom_axis) = plt.subplots(
        2,
        1,
        figsize=(13.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    top_axis.fill_between(
        x_axis,
        majority_curves[1],
        majority_curves[2],
        color="#2b6cb0",
        alpha=0.12,
        linewidth=0,
    )
    top_axis.plot(
        x_axis,
        majority_curves[0],
        color="#2b6cb0",
        linewidth=1.1,
        label="majority mean, IQR shaded",
    )
    top_axis.plot(x_axis, sample_curve, color="#c53030", linewidth=1.1, label="sample")
    top_axis.set_ylabel("SNV intensity", fontsize=8)
    top_axis.set_title(
        f"{folder.folder_key} | {sample.file_name} | {folder.verdict}",
        fontsize=10,
    )
    top_axis.legend(fontsize=7, frameon=False, loc="upper right")
    top_axis.tick_params(labelsize=7)

    limit = float(np.percentile(np.abs(sample.features - folder.majority_mean), 99.5))
    limit = max(limit * 1.15, 1e-6)
    for low, high, _ in folder.bands:
        bottom_axis.axvspan(low, high, color="#f6ad55", alpha=0.35, linewidth=0)
    bottom_axis.plot(x_axis, difference_curve, color="#2d3748", linewidth=1.0)
    bottom_axis.axhline(0.0, color="#a0aec0", linewidth=0.8, linestyle="--")
    bottom_axis.set_ylim(-limit, limit)
    bottom_axis.set_xlim(float(FEATURE_AXIS.min()), float(FEATURE_AXIS.max()))
    bottom_axis.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8)
    bottom_axis.set_ylabel("sample - majority", fontsize=8)
    bottom_axis.tick_params(labelsize=7)

    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _mark_full_init_pending(state: PipelineState) -> None:
    """补充数据没有日期键，只能把全量 init 整体标记为待重建。"""
    report = state.dataset_builds.get("full_init")
    if isinstance(report, dict):
        report["status"] = "pending"
    state.dataset_builds.pop("latest", None)


def _format_separation(value: object) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.3f}"
    return "—"


def write_removal_report(metadata_dir: Path | str, archive_root: Path | str) -> Path:
    """把剔除日志汇总成归档目录下的 Markdown 报告，供人工核对删除数量。"""
    log_path = Path(metadata_dir) / ANALYSIS_REMOVAL_LOG_NAME
    if not log_path.is_file():
        raise FileNotFoundError(f"剔除日志不存在：{log_path}")
    runs = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not runs:
        raise ValueError("剔除日志为空")
    latest = runs[-1]
    archive = Path(archive_root)
    archived_count = count_arc_data(archive, recursive=True)
    figure_count = sum(1 for _ in archive.rglob("*.png"))

    folder_rows: dict[str, dict[str, object]] = {}
    dates: set[str] = set()
    for sample in latest.get("samples", []):
        sample_id = str(sample.get("sample_id", ""))
        parts = sample_id.split("/")
        if len(parts) < 2:
            continue
        key = "/".join(parts[:2])
        row = folder_rows.setdefault(
            key,
            {
                "source_kind": sample.get("source_kind", ""),
                "verdict": sample.get("verdict", ""),
                "separation": sample.get("cluster_separation"),
                "count": 0,
            },
        )
        row["count"] += 1
        if not sample_id.startswith(f"{EXTRA_SOURCE}/"):
            dates.add(parts[0])

    lines = [
        "# 全库高异常剔除记录",
        "",
        f"生成时间：{latest.get('committed_at', '')}",
        f"归档目录：`{archive.name}`",
        "",
        "## 当前归档",
        "",
        f"- 光谱：{archived_count} 条",
        f"- 证据图：{figure_count} 张",
        "",
        "## 最近一次剔除",
        "",
        f"- 合计：{latest.get('sample_count', 0)} 条"
        f"（cosmic {latest.get('cosmic_count', 0)} / extra {latest.get('extra_count', 0)}）",
        f"- 删除 shift 派生谱：{latest.get('shift_removed_count', 0)} 条",
        f"- 涉及日期：{len(dates)} 个（{', '.join(sorted(dates))}）",
        f"- 涉及文件夹：{len(folder_rows)} 个",
        "",
        "| 文件夹 | 类型 | 数量 | 判定 | 质心分离度 |",
        "|---|---|---:|---|---:|",
    ]
    for key in sorted(folder_rows, key=build_natural_key):
        row = folder_rows[key]
        lines.append(
            f"| {key} | {row['source_kind']} | {row['count']} | "
            f"{row['verdict']} | {_format_separation(row['separation'])} |"
        )
    lines.extend(
        [
            "",
            "## 历史剔除批次",
            "",
            "| 批次时间 | 合计 | cosmic | extra | 涉及文件夹 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        folders = {
            "/".join(str(sample.get("sample_id", "")).split("/")[:2])
            for sample in run.get("samples", [])
        }
        timestamp = str(run.get("committed_at", "")).replace("T", " ")[:19]
        lines.append(
            f"| {timestamp} | {run.get('sample_count', 0)} | "
            f"{run.get('cosmic_count', 0)} | {run.get('extra_count', 0)} | "
            f"{len(folders)} |"
        )
    lines.extend(["", ""])
    report_path = archive / ANALYSIS_REMOVAL_REPORT_NAME
    write_text_atomic(report_path, "\n".join(lines))
    return report_path


def execute_analysis_removal(
    data_root: Path | str,
    shift_root: Path | str,
    supplement_root: Path | str,
    state: PipelineState,
    state_path: Path | str,
    report: HeterogeneityReport,
    sample_ids: list[str] | tuple[str, ...] | set[str],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """归档选中样本并更新派生记录；失败时恢复已移动文件。"""
    root = Path(data_root).resolve()
    shift = Path(shift_root).resolve()
    supplement = Path(supplement_root).resolve()
    state_file = Path(state_path)
    metadata_dir = state_file.parent
    targets = build_removal_targets(root, shift, supplement, report, sample_ids)
    archive_folders = {target.archive_path.parent for target in targets}
    cosmic_folders = {
        target.cosmic_source.parent
        for target in targets
        if target.cosmic_source is not None
    }
    shift_folders = {
        target.shift_source.parent
        for target in targets
        if target.shift_source is not None
    }
    extra_folders = {
        target.extra_source.parent
        for target in targets
        if target.extra_source is not None
    }
    snapshots = [
        lineage_path(root),
        state_file,
        metadata_dir / COMMIT_LOG_NAME,
        metadata_dir / ANALYSIS_REMOVAL_LOG_NAME,
        root.parent / ANALYSIS_DELETE_RELATIVE / ANALYSIS_REMOVAL_REPORT_NAME,
        *sorted(archive_folders),
        *sorted(cosmic_folders),
        *sorted(shift_folders),
        *sorted(extra_folders),
    ]
    working = PipelineState.from_payload(state.to_payload())
    total = len(targets)

    if progress_callback is not None:
        progress_callback("校验源文件", total, total)
    with preserve_paths(snapshots):
        move_files(
            {
                target.cosmic_source or target.extra_source: target.archive_path
                for target in targets
            }
        )
        if progress_callback is not None:
            progress_callback("写入证据图", 0, total)
        for index, target in enumerate(targets, start=1):
            figure_path = target.archive_path.with_suffix(".png")
            _plot_evidence(target.folder, target.sample, figure_path)
            if progress_callback is not None:
                progress_callback("写入证据图", index, total)

        removed_shifts = 0
        for target in targets:
            if target.shift_source is not None:
                target.shift_source.unlink()
                removed_shifts += 1
        if progress_callback is not None:
            progress_callback("清理 shift 派生谱", removed_shifts, removed_shifts)

        cosmic_removed = {
            f"{target.sample.date_name}/{target.sample.folder_name}/{target.sample.file_name}": (
                target.archive_path.relative_to(root.parent).as_posix()
            )
            for target in targets
            if target.cosmic_source is not None
        }
        if cosmic_removed:
            record_analysis_removal(root, cosmic_removed)
        dates = sorted(
            {
                target.sample.date_name
                for target in targets
                if target.cosmic_source is not None
            },
            key=build_natural_key,
        )
        if dates:
            invalidate_downstream(working, dates)
        if any(target.extra_source is not None for target in targets):
            _mark_full_init_pending(working)
        write_pipeline_state(state_file, working)

        for folder in sorted(cosmic_folders, key=str):
            remove_empty_parent_directories(folder, root)
        for folder in sorted(shift_folders, key=str):
            remove_empty_parent_directories(folder, shift)
        for folder in sorted(extra_folders, key=str):
            remove_empty_parent_directories(folder, supplement)

        timestamp = datetime.now().astimezone().isoformat()
        commit_paths = sorted(cosmic_removed)
        extra_paths = sorted(
            f"{target.sample.folder_name}/{target.sample.file_name}"
            for target in targets
            if target.extra_source is not None
        )
        commit_payload = {
            "committed_at": timestamp,
            "operation": "analysis_anomaly_removal",
            "paths": commit_paths,
            "extra_paths": extra_paths,
            "reasons": {path: [ANALYSIS_REMOVAL_REASON] for path in commit_paths},
            "derived_removed": {
                "shift_data": removed_shifts,
                "extra_data": len(extra_paths),
            },
        }
        with (metadata_dir / COMMIT_LOG_NAME).open("a", encoding="utf-8") as file:
            file.write(json.dumps(commit_payload, ensure_ascii=False) + "\n")

        removal_payload = {
            "committed_at": timestamp,
            "operation": "analysis_anomaly_removal",
            "reason": ANALYSIS_REMOVAL_REASON,
            "drift_threshold": report.drift_threshold,
            "sample_count": total,
            "cosmic_count": len(commit_paths),
            "extra_count": len(extra_paths),
            "shift_removed_count": removed_shifts,
            "samples": [
                {
                    "sample_id": target.sample.sample_id,
                    "source_kind": target.sample.source_kind,
                    "archive_path": target.archive_path.relative_to(root.parent).as_posix(),
                    "verdict": target.folder.verdict,
                    "cluster": target.sample.cluster,
                    "cluster_separation": target.folder.cluster_separation,
                    "outlier_vote": target.sample.outlier_vote,
                }
                for target in targets
            ],
        }
        with (metadata_dir / ANALYSIS_REMOVAL_LOG_NAME).open("a", encoding="utf-8") as file:
            file.write(json.dumps(removal_payload, ensure_ascii=False) + "\n")
        report_path = write_removal_report(metadata_dir, root.parent / ANALYSIS_DELETE_RELATIVE)

    state.__dict__.update(working.__dict__)
    return {
        "sample_count": total,
        "cosmic_count": len(commit_paths),
        "extra_count": len(extra_paths),
        "shift_removed_count": removed_shifts,
        "dates": dates,
        "report_path": report_path.relative_to(root.parent).as_posix(),
        "archive_paths": [
            target.archive_path.relative_to(root.parent).as_posix()
            for target in targets
        ],
    }


__all__ = [
    "ANALYSIS_REMOVAL_LOG_NAME",
    "ANALYSIS_REMOVAL_REPORT_NAME",
    "AnalysisGate",
    "RemovalTarget",
    "build_removal_targets",
    "check_analysis_gate",
    "execute_analysis_removal",
    "write_removal_report",
]
