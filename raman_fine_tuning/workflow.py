"""供 notebook 调用的跨数据集微调流程辅助函数。"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from raman.data.build import build_train
from raman.data.io import unpack_init
from raman.data.profiles import get_dataset_dir, get_profile
from raman.pipeline import PipelineConfig
from raman_fine_tuning.config import FineTuneConfig
from raman_fine_tuning.fine_tune import run_fine_tuning


@dataclass(frozen=True)
class FineTuneDatasetContext:
    """目标数据集准备完成后的必要路径。"""

    profile: object
    dataset_dir: Path
    init_dir: Path
    train_dir: Path
    reused_init: bool
    reused_train: bool


def prepare_target_dataset(
    project_root,
    fine_tune_config: FineTuneConfig,
    *,
    rebuild_train: bool = False,
    clear_init_before_unpack: bool = True,
) -> FineTuneDatasetContext:
    """按微调输入轴准备目标数据；已有 init/train 时默认复用。"""
    project_root = Path(project_root)
    profile = get_profile(fine_tune_config.target_dataset_name)
    dataset_dir = get_dataset_dir(profile, project_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    init_pack = dataset_dir / profile.root_init_pack
    init_dir = dataset_dir / profile.root_init
    train_dir = dataset_dir / profile.root_train_clean

    reused_init = init_dir.is_dir()
    if not reused_init:
        if not init_pack.is_file():
            raise FileNotFoundError(
                f"找不到目标 init.npz：{init_pack}；请先将压缩包放到该位置。"
            )
        if clear_init_before_unpack:
            shutil.rmtree(init_dir, ignore_errors=True)
        init_dir.mkdir(parents=True, exist_ok=True)
        unpack_init(init_pack, init_dir)
        print(f"unpacked target init: {init_pack}")
    else:
        print(f"reuse target init: {init_dir}")

    reused_train = train_dir.is_dir() and not rebuild_train
    if not reused_train:
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(
            dataset_dir / f"{profile.root_train_clean}_building",
            ignore_errors=True,
        )
        build_train(
            profile,
            dataset_dir,
            pipeline_config=PipelineConfig(
                input_grid_mode=fine_tune_config.input_grid_mode
            ),
        )
        print(f"built target train: {train_dir}")
    else:
        print(f"reuse target train: {train_dir}")

    return FineTuneDatasetContext(
        profile=profile,
        dataset_dir=dataset_dir,
        init_dir=init_dir,
        train_dir=train_dir,
        reused_init=reused_init,
        reused_train=reused_train,
    )


def _group_to_dict(group):
    """兼容 dataclass 与普通配置对象的简洁展示。"""
    if is_dataclass(group):
        return asdict(group)
    return {
        key: value
        for key, value in vars(group).items()
        if not key.startswith("_")
    }


def print_fine_tune_plan(base_config, fine_tune_config: FineTuneConfig):
    """打印将实际用于微调的临时配置，并返回该配置。"""
    target_config = fine_tune_config.build_target_config(base_config)
    print("\n===== Fine-tune Input =====")
    print(f"  dataset_root: {target_config.dataset_root}")
    print(f"  in_channels: {target_config.in_channels}")
    print(f"  delta: {target_config.delta}")
    for title, group in (
        ("Fine-tune Shared Input Config", target_config.shared),
        ("Fine-tune Model Run Config", target_config.model),
        ("Fine-tune Runtime Config", target_config.runtime),
        ("Fine-tune Mode", fine_tune_config),
    ):
        print(f"\n===== {title} =====")
        for key, value in _group_to_dict(group).items():
            print(f"  {key}: {value}")
    return target_config


def run_single_fine_tuning(base_config, fine_tune_config: FineTuneConfig):
    """执行单个层级或单个属的微调，并返回实验根和实际 run 路径。"""
    result = run_fine_tuning(base_config, fine_tune_config)
    run_dirs = result.get("run_dirs", [])
    if len(run_dirs) != 1:
        raise RuntimeError(f"微调期望得到一个 run，实际为：{run_dirs}")
    run_dir = Path(run_dirs[0])
    if not run_dir.is_absolute():
        run_dir = Path(result["output_dir"]) / run_dir
    return result, str(run_dir.resolve())
