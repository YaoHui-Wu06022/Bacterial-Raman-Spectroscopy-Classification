"""为同一份 profile 数据准备共享切分的训练实验组。"""

from __future__ import annotations

import re
from pathlib import Path

from ramanv2.common.file_ops import file_sha256
from ramanv2.core.config import Config, build_config
from ramanv2.core.run_context import open_experiment_context
from ramanv2.common.paths import PROJECT_ROOT
from ramanv2.data.profiles import resolve_selection
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.training.split import (
    load_split_files,
    save_split_files,
    split_by_lowest_level_ratio,
)


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def build_experiment_group(
    profile_id: str,
    tag: str,
    experiment_names: tuple[str, ...] = ("main",),
    seed: int = 42,
    train_ratio: float = 0.8,
    project_root: Path | str = PROJECT_ROOT,
    config: Config | None = None,
) -> dict[str, object]:
    """生成一份 canonical 切分，并写入同一 profile 的多个实验目录。

    ``experiment_names`` 为 ``("main",)`` 时实验目录为 ``<tag>_<seed>``；
    多个实验时使用 ``<tag>_<name>_<seed>``。每个实验目录都会写入相同的
    train/val 切分和共享配置，后续训练会复用该切分。
    """
    _validate_name(tag, "tag")
    if not experiment_names:
        raise ValueError("experiment_names 不能为空")
    for name in experiment_names:
        _validate_name(name, "experiment name")
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio 必须在 0 和 1 之间")

    selection = resolve_selection(profile_id=profile_id, project_root=project_root)
    profile = selection.profile
    train_dir = selection.dataset_dir / profile.root_train_clean
    if not train_dir.is_dir():
        raise FileNotFoundError(f"缺少 profile train 目录：{train_dir}")

    dataset = DatasetIndex(
        train_dir,
        hierarchy=selection.hierarchy,
        load_intensity_enable=False,
    )
    train_indices, val_indices = split_by_lowest_level_ratio(
        dataset,
        train_ratio=float(train_ratio),
        seed=int(seed),
    )
    train_indices = sorted(int(index) for index in train_indices)
    val_indices = sorted(int(index) for index in val_indices)

    config = config or build_config(
        {
            "dataset_id": selection.spec.dataset_id,
            "profile_id": profile.profile_id,
            "label_schema": selection.hierarchy.label_schema,
            "bad_bands": selection.spec.bad_bands,
        }
    )
    if (
        config.dataset.dataset_id != selection.spec.dataset_id
        or config.dataset.profile_id != profile.profile_id
    ):
        raise ValueError("传入的 config 与当前数据集不匹配")
    output_root = (
        Path(project_root).resolve()
        / "output"
        / selection.spec.directory.name
    )
    experiments: dict[str, dict[str, object]] = {}
    targets: list[tuple[str, Path, bool]] = []
    for name in experiment_names:
        experiment_dir = (
            output_root / f"{tag}_{int(seed)}"
            if len(experiment_names) == 1 and name == "main"
            else output_root / f"{tag}_{name}_{int(seed)}"
        )
        existing = load_split_files(dataset, experiment_dir)
        reused = existing is not None
        if existing is not None:
            existing_train, existing_val = (sorted(int(item) for item in part) for part in existing)
            if existing_train != train_indices or existing_val != val_indices:
                raise ValueError(
                    f"实验目录已有不同切分：{experiment_dir}；请更换 tag 或清理该目录"
                )
        targets.append((name, experiment_dir, reused))

    for name, experiment_dir, reused in targets:
        open_experiment_context(experiment_dir, config)
        save_split_files(dataset, train_indices, val_indices, experiment_dir)
        experiments[name] = {
            "experiment_dir": str(experiment_dir),
            "split_reused": reused,
            "train_split_sha256": file_sha256(experiment_dir / "train_split.json"),
            "val_split_sha256": file_sha256(experiment_dir / "val_split.json"),
        }

    return {
        "dataset_id": selection.spec.dataset_id,
        "profile_id": profile.profile_id,
        "tag": tag,
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "sample_count": len(dataset),
        "level_names": list(dataset.level_names),
        "experiments": experiments,
    }


def _validate_name(value: str, label: str) -> None:
    """拒绝空值和非 ASCII 名称，避免跨平台命令行与路径问题。"""
    text = str(value).strip()
    if not text or NAME_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} 只能包含 ASCII 字母、数字、下划线和短横线：{value!r}")
