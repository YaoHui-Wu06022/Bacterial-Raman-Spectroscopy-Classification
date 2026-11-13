"""按训练 scope 批量执行模型训练，并校验实验级训练快照。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramanv2.common.paths import resolve_path
from ramanv2.core.config import Config
from ramanv2.training.snapshot import TRAIN_ARCHIVE_NAME, load_train_archive_summary
from ramanv2.training.workflow import TrainRequest, run_training


@dataclass(frozen=True)
class TrainingScopeResult:
    """一次 scope 训练产生的槽位与快照信息。"""

    level_name: str
    parent_ids: tuple[int | None, ...]
    slot_dirs: tuple[Path, ...]
    last_slot_dir: Path | None
    snapshot_path: Path | None
    sample_count: int | None
    intensity_shape: tuple[int, ...] | None


def run_training_scope(
    config: Config,
    experiment_dir: Path | str,
    level_name: str,
    parent_ids: list[int | None] | tuple[int | None, ...],
    global_head_enable: bool = False,
) -> TrainingScopeResult:
    """在固定实验目录内依次训练指定层级的所有父类槽位。"""
    target_dir = resolve_path(experiment_dir)
    slot_dirs: list[Path] = []
    for parent_id in parent_ids:
        slot_dir = target_dir / level_name
        if parent_id is not None:
            slot_dir = slot_dir / f"{level_name}_{parent_id}"
        print(f"=== 训练 {level_name}  parent={parent_id}  -> {slot_dir} ===")
        run_training(
            TrainRequest(
                config=config,
                level_name=level_name,
                parent_id=parent_id,
                global_head_enable=global_head_enable,
                experiment_dir=target_dir,
            )
        )
        slot_dirs.append(slot_dir)

    if not slot_dirs:
        return TrainingScopeResult(
            level_name=level_name,
            parent_ids=tuple(parent_ids),
            slot_dirs=(),
            last_slot_dir=None,
            snapshot_path=None,
            sample_count=None,
            intensity_shape=None,
        )

    snapshot_path = target_dir / TRAIN_ARCHIVE_NAME
    if not snapshot_path.is_file():
        raise RuntimeError(f"缺少训练数据快照：{snapshot_path}")
    sample_count, intensity_shape = load_train_archive_summary(snapshot_path)
    return TrainingScopeResult(
        level_name=level_name,
        parent_ids=tuple(parent_ids),
        slot_dirs=tuple(slot_dirs),
        last_slot_dir=slot_dirs[-1],
        snapshot_path=snapshot_path,
        sample_count=sample_count,
        intensity_shape=intensity_shape,
    )


__all__ = ["TrainingScopeResult", "run_training_scope"]
