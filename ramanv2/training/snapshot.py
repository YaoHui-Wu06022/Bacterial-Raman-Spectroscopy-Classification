"""保存每个训练 run 实际使用的原始 train 光谱快照。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ramanv2.common.publishing import create_temporary_path, publish_directory
from ramanv2.core.paths import normalize_relpath


def save_or_validate_train_snapshot(dataset: Any, train_task: Any, run_context: Any) -> Path:
    """原子复制 train split；恢复训练时校验已有 manifest 与当前任务一致。"""
    target_dir = Path(run_context.run_dir) / "train_data"
    manifest_path = target_dir / "manifest.json"
    records = _build_sample_records(dataset, train_task)
    if manifest_path.is_file():
        _validate_existing_snapshot(target_dir, manifest_path, records, train_task)
        return manifest_path
    if target_dir.exists():
        raise ValueError(f"训练快照目录缺少 manifest.json：{target_dir}")

    temporary = create_temporary_path(target_dir)
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        for record in records:
            source = Path(record["source_path"])
            destination = temporary / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        payload = {
            "version": 1,
            "experiment_dir": str(run_context.experiment_context.experiment_dir),
            "run_dir": str(run_context.run_dir),
            "train_root": str(dataset.root_dir),
            "level": train_task.level_name,
            "level_name": train_task.level_name,
            "parent_id": train_task.parent_id,
            "model_tag": train_task.model_tag,
            "sample_count": len(records),
            "samples": [{key: value for key, value in item.items() if key != "source_path"} for item in records],
            "config_path": str(run_context.resolved_config_path),
            "config_snapshot_path": str(run_context.resolved_config_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_directory(temporary, target_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return manifest_path


def _build_sample_records(dataset: Any, train_task: Any) -> list[dict[str, Any]]:
    root_dir = Path(dataset.root_dir).resolve()
    records: list[dict[str, Any]] = []
    for index in train_task.train_indices:
        source = Path(dataset.samples[int(index)]).resolve()
        relative = source.relative_to(root_dir)
        records.append(
            {
                "source_path": str(source),
                "path": normalize_relpath(relative),
                "relative_path": normalize_relpath(relative),
                "labels": [int(value) for value in dataset.level_labels[int(index)].tolist()],
                "hierarchy": dataset.get_hierarchy(int(index)),
                "target_class": dataset.get_level_key(int(index), train_task.level_name),
                "target_label": int(dataset.level_labels[int(index), train_task.level_index]),
            }
        )
    return records


def _validate_existing_snapshot(target_dir: Path, manifest_path: Path, records: list[dict[str, Any]], train_task: Any) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = [item["path"] for item in records]
    actual_paths = [item.get("path") for item in payload.get("samples", [])]
    if (
        payload.get("sample_count") != len(records)
        or actual_paths != expected_paths
        or payload.get("level") != train_task.level_name
        or payload.get("parent_id") != train_task.parent_id
        or payload.get("model_tag") != train_task.model_tag
    ):
        raise ValueError(f"训练快照与当前任务不一致：{manifest_path}")
    missing = [path for path in expected_paths if not (target_dir / path).is_file()]
    if missing:
        raise FileNotFoundError(f"训练快照缺少原始光谱：{target_dir / missing[0]}")


__all__ = ["save_or_validate_train_snapshot"]
