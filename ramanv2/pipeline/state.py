"""前置流水线的持久化状态。

状态文件只描述页面流程进度和可恢复的业务记录，不保存 Streamlit 控件对象。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PIPELINE_STATE_VERSION = 1
STATE_DIR_NAME = ".raman_pipeline"
STATE_FILE_NAME = "state.json"
COMMIT_LOG_NAME = "commit_log.jsonl"


@dataclass
class CleaningState:
    """保存单日期增量审核的可恢复状态。"""

    current_date_name: str | None = None
    current_folder_key: str | None = None
    folder_catalog: dict[str, set[str]] = field(default_factory=dict)
    marked_paths: set[str] = field(default_factory=set)
    mark_reasons: dict[str, set[str]] = field(default_factory=dict)
    completed_folder_keys: set[str] = field(default_factory=set)
    completed_date_names: set[str] = field(default_factory=set)
    seeded_folder_keys: set[str] = field(default_factory=set)
    folder_page_indices: dict[str, int] = field(default_factory=dict)
    committed_paths: set[str] = field(default_factory=set)

    def to_payload(self) -> dict[str, object]:
        return {
            "current_date_name": self.current_date_name,
            "current_folder_key": self.current_folder_key,
            "folder_catalog": {
                date_name: sorted(folder_names)
                for date_name, folder_names in sorted(self.folder_catalog.items())
            },
            "marked_paths": sorted(self.marked_paths),
            "mark_reasons": {
                path: sorted(reasons)
                for path, reasons in sorted(self.mark_reasons.items())
                if reasons
            },
            "completed_folder_keys": sorted(self.completed_folder_keys),
            "completed_date_names": sorted(self.completed_date_names),
            "seeded_folder_keys": sorted(self.seeded_folder_keys),
            "folder_page_indices": dict(sorted(self.folder_page_indices.items())),
            "committed_paths": sorted(self.committed_paths),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CleaningState":
        if not isinstance(payload, dict):
            raise ValueError("cleaning 状态必须是 JSON 对象")

        def read_paths(name: str) -> set[str]:
            value = payload.get(name, [])
            if not isinstance(value, list):
                raise ValueError(f"cleaning.{name} 必须是列表")
            return {str(item).replace("\\", "/") for item in value}

        raw_catalog = payload.get("folder_catalog", {})
        if not isinstance(raw_catalog, dict):
            raise ValueError("cleaning.folder_catalog 必须是对象")
        folder_catalog = {}
        for date_name, names in raw_catalog.items():
            if not isinstance(names, list):
                raise ValueError("cleaning.folder_catalog 的值必须是列表")
            folder_catalog[str(date_name)] = {str(name) for name in names}

        raw_reasons = payload.get("mark_reasons", {})
        if not isinstance(raw_reasons, dict):
            raise ValueError("cleaning.mark_reasons 必须是对象")
        mark_reasons = {}
        for path, reasons in raw_reasons.items():
            if not isinstance(reasons, list):
                raise ValueError("cleaning.mark_reasons 的值必须是列表")
            mark_reasons[str(path).replace("\\", "/")] = {str(reason) for reason in reasons}

        raw_pages = payload.get("folder_page_indices", {})
        if not isinstance(raw_pages, dict):
            raise ValueError("cleaning.folder_page_indices 必须是对象")
        folder_page_indices = {}
        for key, value in raw_pages.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError("cleaning.folder_page_indices 必须是非负整数")
            folder_page_indices[str(key)] = value

        return cls(
            current_date_name=_optional_string(payload.get("current_date_name")),
            current_folder_key=_optional_string(payload.get("current_folder_key")),
            folder_catalog=folder_catalog,
            marked_paths=read_paths("marked_paths"),
            mark_reasons=mark_reasons,
            completed_folder_keys=read_paths("completed_folder_keys"),
            completed_date_names=read_paths("completed_date_names"),
            seeded_folder_keys=read_paths("seeded_folder_keys"),
            folder_page_indices=folder_page_indices,
            committed_paths=read_paths("committed_paths"),
        )


@dataclass
class PipelineState:
    """保存六个页面共享的流程门禁。"""

    active_page: str = "数据清洗"
    cleaning: CleaningState = field(default_factory=CleaningState)
    affine_completed_dates: set[str] = field(default_factory=set)
    audit_completed_dates: set[str] = field(default_factory=set)
    init_generated_dates: set[str] = field(default_factory=set)
    dataset_builds: dict[str, dict[str, object]] = field(default_factory=dict)
    updated_at: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "version": PIPELINE_STATE_VERSION,
            "active_page": self.active_page,
            "cleaning": self.cleaning.to_payload(),
            "affine_completed_dates": sorted(self.affine_completed_dates),
            "audit_completed_dates": sorted(self.audit_completed_dates),
            "init_generated_dates": sorted(self.init_generated_dates),
            "dataset_builds": self.dataset_builds,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "PipelineState":
        if not isinstance(payload, dict):
            raise ValueError("流水线状态必须是 JSON 对象")
        if payload.get("version") != PIPELINE_STATE_VERSION:
            raise ValueError(f"流水线状态版本必须为 {PIPELINE_STATE_VERSION}")
        cleaning = CleaningState.from_payload(payload.get("cleaning", {}))
        return cls(
            active_page=str(payload.get("active_page", "数据清洗")),
            cleaning=cleaning,
            affine_completed_dates=_read_string_set(payload.get("affine_completed_dates", []), "affine_completed_dates"),
            audit_completed_dates=_read_string_set(payload.get("audit_completed_dates", []), "audit_completed_dates"),
            init_generated_dates=_read_string_set(payload.get("init_generated_dates", []), "init_generated_dates"),
            dataset_builds=_read_dict(payload.get("dataset_builds", {}), "dataset_builds"),
            updated_at=_optional_string(payload.get("updated_at")),
        )


def pipeline_state_path(project_root: Path | str) -> Path:
    return Path(project_root).resolve() / "dataset" / STATE_DIR_NAME / STATE_FILE_NAME


def pipeline_metadata_dir(project_root: Path | str) -> Path:
    return pipeline_state_path(project_root).parent


def load_pipeline_state(path: Path | str) -> PipelineState:
    target = Path(path)
    if not target.is_file():
        return PipelineState()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return PipelineState.from_payload(payload)


def save_pipeline_state(path: Path | str, state: PipelineState) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    # 采集目录在 Windows 上可能继承只读属性，先确保状态目录和旧状态文件可写。
    try:
        target.parent.chmod(0o700)
    except OSError:
        pass
    if target.exists():
        try:
            target.chmod(0o600)
        except OSError:
            pass

    # 使用唯一临时文件，避免多个浏览器会话同时保存时互相覆盖临时文件。
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(state.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for attempt, delay in enumerate((0.0, 0.05, 0.2, 0.5)):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                # 状态文件可能正被另一个页面刷新短暂占用，下一次循环重试。
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _read_string_set(value: object, name: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是列表")
    return {str(item) for item in value}


def _read_dict(value: object, name: str) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是对象")
    result = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ValueError(f"{name}.{key} 必须是对象")
        result[str(key)] = dict(item)
    return result


__all__ = [
    "COMMIT_LOG_NAME",
    "CleaningState",
    "PIPELINE_STATE_VERSION",
    "PipelineState",
    "load_pipeline_state",
    "pipeline_metadata_dir",
    "pipeline_state_path",
    "save_pipeline_state",
]
