"""保存每个实验实际使用的 train split 原始光谱压缩包。"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ramanv2.common.file_ops import create_temporary_path, publish_file
from ramanv2.common.paths import normalize_relpath
from ramanv2.core.run_context import ExperimentContext


TRAIN_ARCHIVE_NAME = "train_data.zip"
MANIFEST_NAME = "manifest.json"
ARCHIVE_VERSION = 2


def save_or_validate_train_archive(
    experiment_context: ExperimentContext,
    dataset: Any,
    train_indices: Any,
) -> Path:
    """原子保存实验级 train split 压缩包；已有压缩包时校验内容一致。"""
    target_path = Path(experiment_context.experiment_dir) / TRAIN_ARCHIVE_NAME
    root_dir = Path(dataset.root_dir).resolve()
    paths = _build_sample_paths(dataset, train_indices, root_dir)
    if target_path.is_file():
        _validate_archive(target_path, paths)
        return target_path

    payload = {
        "version": ARCHIVE_VERSION,
        "sample_count": len(paths),
        "samples": paths,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = create_temporary_path(target_path)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            for path in paths:
                archive.write(root_dir / path, arcname=path)
        publish_file(temporary, target_path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target_path


def _build_sample_paths(dataset: Any, train_indices: Any, root_dir: Path) -> list[str]:
    """按 train split 顺序返回相对数据根的样本路径。"""
    return [
        normalize_relpath(
            Path(dataset.samples[int(index)]).resolve().relative_to(root_dir)
        )
        for index in train_indices
    ]


def _validate_archive(path: Path, paths: list[str]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            manifest_data = archive.read(MANIFEST_NAME)
            names = set(archive.namelist())
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"训练数据压缩包无效：{path}") from error
    payload = json.loads(manifest_data)
    actual_paths = payload.get("samples", [])
    if not all(isinstance(item, str) for item in actual_paths):
        raise ValueError(f"训练数据压缩包清单格式无效：{path}")
    if (
        payload.get("sample_count") != len(paths)
        or sorted(actual_paths) != sorted(paths)
    ):
        raise ValueError(f"训练数据压缩包与当前切分不一致：{path}")
    missing = [sample_path for sample_path in paths if sample_path not in names]
    if missing:
        raise FileNotFoundError(
            f"训练数据压缩包缺少原始光谱：{path} -> {missing[0]}"
        )


__all__ = ["TRAIN_ARCHIVE_NAME", "save_or_validate_train_archive"]
