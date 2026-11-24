"""文件夹入库：清洗一个文件夹并发布到 cosmic_data。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ramanv2.common.arc_data import read_arc_data
from ramanv2.common.file_ops import create_temporary_path, rename_directory, write_json
from ramanv2.common.naming import BEAD_FOLDER_NAME, is_date_name
from ramanv2.data.datasets import DATASET_RAW_DIR_NAME, DatasetSpec
from ramanv2.data.folder_cosmic import (
    build_bead_folder_report,
    build_folder_cosmic_payload,
    clean_spectrum_folder,
    read_spectrum_folder,
    write_spectrum_folder,
)
from ramanv2.pipeline.db import refresh_fingerprints
from ramanv2.pipeline.state import (
    invalidate_downstream,
    load_pipeline_state,
    pipeline_state_path_for_spec,
    write_pipeline_state,
)
from ramanv2.pipeline.lineage import initialize_folder_lineage, sync_pipeline_state_from_lineage


_PREFIX_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_REPORT_NAME = "cosmic_ingest_report.json"
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class FolderIngestRequest:
    """描述一次单文件夹入库请求。"""

    spec: DatasetSpec
    source_dir: Path | str
    date_name: str
    prefix: str
    apply: bool = False
    bead: bool = False
    progress_callback: ProgressCallback | None = None


def load_folder_ingest_manifest(path: Path | str) -> dict:
    """读取 ASCII 命令行的 JSON 清单，支持非 ASCII 源路径。"""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("入库清单必须是 JSON 对象")
    return dict(payload)


def _notify(callback: ProgressCallback | None, message: str, done: int, total: int) -> None:
    if callback is not None:
        callback(message, done, total)


def is_valid_prefix(name: str) -> bool:
    """判断 raw_data 前缀是否为字母开头的 ASCII 标识。"""
    return bool(_PREFIX_PATTERN.fullmatch(str(name).strip()))


def resolve_batch_names(date_name: str, prefix: str, *, bead: bool = False) -> tuple[str, str]:
    """校验并返回 raw_data 批次对应的日期与前缀；小球批次固定为 小球。"""
    date_text = str(date_name).strip()
    if not is_date_name(date_text):
        raise ValueError("日期必须是八位数字，例如 20240101")
    if bead:
        return date_text, BEAD_FOLDER_NAME
    prefix_text = str(prefix).strip()
    if not is_valid_prefix(prefix_text):
        raise ValueError("前缀只能包含字母、数字、下划线或短横线，且以字母开头")
    return date_text, prefix_text


def _resolve_source(
    spec: DatasetSpec,
    source_dir: Path | str,
    date_text: str,
    prefix_text: str,
) -> Path:
    """校验来源；数据集内只允许 raw_data/<date>/<prefix>，外部路径保持开放。"""
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"源文件夹不存在：{source}")
    dataset_dir = spec.dataset_dir
    if source == dataset_dir:
        raise ValueError("源文件夹不能是数据集根目录")
    if source.is_relative_to(dataset_dir):
        expected = (spec.raw_dir / date_text / prefix_text).resolve()
        if source != expected:
            raise ValueError(
                f"数据集内来源必须是 {DATASET_RAW_DIR_NAME}/{date_text}/{prefix_text}，当前为 {source}"
            )
    return source


def _folder_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _validate_written_folder(target_dir: Path, folder) -> None:
    """校验临时目录中每个文件的行数、波数轴和有限值。"""
    for index, name in enumerate(folder.names):
        path = target_dir / name
        wavenumbers, intensities = read_arc_data(path)
        if wavenumbers.size != folder.wavenumbers[index].size or intensities.size != wavenumbers.size:
            raise ValueError(f"写出校验失败：{path}")
        if not np.allclose(wavenumbers, folder.wavenumbers[index], rtol=0.0, atol=1e-9):
            raise ValueError(f"写出波数轴变化：{path}")
        if not np.isfinite(intensities).all():
            raise ValueError(f"写出包含非有限值：{path}")


def _blocked_reason(report: dict) -> str | None:
    if report["status"] == "skipped_small":
        return f"有效谱数不足 min_spectra：{report['spectrum_count']}"
    return None


def run_folder_ingest(request: FolderIngestRequest) -> dict:
    """执行 dry-run 或从 raw_data/外部源生成 cosmic_data；目标存在时不覆盖。"""
    spec = request.spec
    date_text, prefix_text = resolve_batch_names(
        request.date_name, request.prefix, bead=request.bead,
    )
    source = _resolve_source(spec, request.source_dir, date_text, prefix_text)
    target = spec.cosmic_dir / date_text / prefix_text
    target_exists = target.exists()
    config_enabled = spec.cosmic_removal.enable
    if request.apply and not request.bead and not config_enabled:
        raise ValueError("数据集未启用 cosmic_removal，请先在数据集配置中开启")
    if not request.apply and target_exists:
        return {
            "dataset_id": spec.dataset_id,
            "folder_key": f"{date_text}/{prefix_text}",
            "source_dir": str(source),
            "target_dir": str(target),
            "target_exists": True,
            "config_enabled": config_enabled,
            "bead": request.bead,
            "status": "already_generated",
            "applied": False,
            "blocked_reason": "目标已存在，不覆盖",
        }
    if request.apply and target_exists:
        raise ValueError(f"目标已存在，不覆盖：{target}")

    _notify(request.progress_callback, "读取源文件夹", 0, 3)
    folder = read_spectrum_folder(source, folder_key=f"{date_text}/{prefix_text}")
    if request.bead:
        _notify(request.progress_callback, "去文件头", 1, 3)
        cleaned = folder.intensities.copy()
        report = build_bead_folder_report(folder)
        options_payload = None
        options_hash = None
    else:
        options = spec.cosmic_removal if config_enabled else replace(spec.cosmic_removal, enable=True)
        _notify(request.progress_callback, "检测与替换尖峰", 1, 3)
        cleaned, report = clean_spectrum_folder(folder, options)
        options_payload = build_folder_cosmic_payload(options)
        options_hash = hashlib.sha256(
            json.dumps(options_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
    report = dict(report)
    report.update({
        "dataset_id": spec.dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_enabled": config_enabled,
        "bead": request.bead,
        "source_fingerprint": _folder_fingerprint(folder.paths),
        "target_dir": str(target),
        "target_exists": target_exists,
        "options": options_payload,
        "options_hash": options_hash,
    })
    blocked = _blocked_reason(report)
    if blocked is not None:
        report["blocked_reason"] = blocked
    if not request.apply:
        report["applied"] = False
        return report
    if blocked is not None:
        raise ValueError(blocked)

    temporary = create_temporary_path(target)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _notify(request.progress_callback, "写出无文件头光谱", 2, 3)
        write_spectrum_folder(folder, cleaned, temporary)
        _validate_written_folder(temporary, folder)
        if target.exists():
            raise ValueError(f"目标已存在，不覆盖：{target}")
        _notify(request.progress_callback, "生成 cosmic_data", 3, 3)
        rename_directory(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    _notify(request.progress_callback, "刷新入库记录", 3, 3)
    initialize_folder_lineage(spec.cosmic_dir, date_names={date_text})
    refresh_fingerprints(spec.metadata_dir, spec.cosmic_dir)
    state_path = pipeline_state_path_for_spec(spec)
    state = load_pipeline_state(state_path)
    sync_pipeline_state_from_lineage(spec.cosmic_dir, state)
    invalidate_downstream(state, [date_text])
    write_pipeline_state(state_path, state)
    report["applied"] = True
    report["report_path"] = str(spec.metadata_dir / _REPORT_NAME)
    write_json(spec.metadata_dir / _REPORT_NAME, report)
    return report


def format_ingest_summary(report: dict) -> str:
    """把入库报告压缩成命令行可读摘要。"""
    lines = [
        f"status={report.get('status')} applied={report.get('applied')}",
        f"source={report.get('source_dir')}",
        f"target={report.get('target_dir')}",
    ]
    if "spectrum_count" in report:
        lines.append(
            f"files={report.get('spectrum_count')} removed_points={report.get('removed_points')} "
            f"removed_spectra={report.get('removed_spectra')}"
        )
    if report.get("blocked_reason"):
        lines.append("blocked=" + str(report["blocked_reason"]))
    return "\n".join(lines)