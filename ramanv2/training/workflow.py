"""训练任务的范围解析、run 编排与层级元数据汇总。"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from ramanv2.core.config import Config
from ramanv2.core.run_snapshot import load_run_snapshot
from ramanv2.common.hierarchy import (
    assert_hierarchy_compatible,
    build_hierarchy_entry,
    build_hierarchy_meta,
    load_hierarchy_meta,
    merge_hierarchy_meta,
    write_hierarchy_meta,
)
from ramanv2.core.input_spec import build_input_spec
from ramanv2.common.paths import PROJECT_ROOT, relpath, resolve_path
from ramanv2.core.run_context import (
    CHECKPOINT_FILE_NAME,
    ExperimentContext,
    RunContext,
    open_experiment_context,
    open_run_context,
    resolve_run_dir,
)
from ramanv2.data.runtime.augmentation import build_augmentation_spec
from ramanv2.data.runtime.dataset import RamanDataset
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.profiles import DatasetSelection, resolve_selection
from ramanv2.modeling.factory import build_model, validate_model_input
from ramanv2.modeling.model_spec import ModelSpec, build_model_spec
from ramanv2.training.loop import TrainArtifacts, TrainResult, run_train_loop
from ramanv2.training.optimizer import build_loader
from ramanv2.training.snapshot import save_or_validate_train_archive
from ramanv2.training.split import (
    TrainTask,
    build_global_train_task,
    build_parent_train_task,
    resolve_parent_target,
    resolve_train_split,
)
from ramanv2.training.train_spec import (
    ExecutionSpec,
    TrainingSpec,
    build_execution_spec,
    build_training_spec,
)


@dataclass(frozen=True)
class TrainRequest:
    """一次训练入口请求及可选的隔离微调回调。"""

    config: Config
    level_name: str
    parent_id: int | None = None
    parent_name: str | None = None
    experiment_dir: Path | str | None = None
    train_dir: Path | str | None = None
    resume_run_dir: Path | str | None = None
    initialize_model: Callable[[torch.nn.Module, TrainTask], None] | None = None
    apply_training_mode: Callable[[torch.nn.Module], None] | None = None


def run_training(request: TrainRequest) -> dict[str, Any]:
    """训练一个模型，并把它的条目合并进实验层级元数据。"""
    config = request.config
    input_spec = build_input_spec(config.input)
    train_spec = build_training_spec(config.training, config.execution)
    runtime_spec = build_execution_spec(config.execution)
    model_spec = build_model_spec(config.model, input_spec)
    validate_model_input(model_spec, input_spec)
    _set_random_seed(train_spec.seed, runtime_spec.deterministic_enable)

    selection = resolve_selection(config.dataset.dataset_id, config.dataset.profile_id)
    experiment_dir = _resolve_experiment_dir(request, selection)
    existing_meta = load_hierarchy_meta(experiment_dir / "hierarchy_meta.json")
    if request.resume_run_dir is not None:
        load_run_snapshot(request.resume_run_dir)
    dataset_spec = selection.spec
    dataset_index = DatasetIndex(
        _resolve_train_dir(request, selection),
        hierarchy=dataset_spec.hierarchy,
    )
    level_name = dataset_index.resolve_level_name(request.level_name)
    parent_id = resolve_parent_target(
        dataset_index,
        level_name,
        request.parent_id,
        request.parent_name,
    )
    level_models: dict[str, dict[str, Any]] = {}
    parent_models: dict[str, dict[str, dict[str, Any]]] = {}
    current_meta = build_hierarchy_meta(
        **dataset_index.hierarchy_metadata,
        level_models=level_models,
        parent_models=parent_models,
        label_schema=dataset_spec.hierarchy.label_schema,
        hierarchy=dataset_index.hierarchy.to_payload(),
    )
    # 先检查历史标签空间，避免创建槽位、改写快照后才发现分类头不兼容。
    if existing_meta is not None:
        assert_hierarchy_compatible(existing_meta, current_meta)
    experiment_context = open_experiment_context(experiment_dir, config)
    train_indices, validation_indices = resolve_train_split(
        dataset_index,
        train_spec.train_ratio,
        train_spec.seed,
        experiment_context.experiment_dir,
    )
    save_or_validate_train_archive(experiment_context, dataset_index, train_indices)
    train_task = _build_train_task(
        dataset_index,
        level_name,
        parent_id,
        train_indices,
        validation_indices,
    )

    train_dataset = RamanDataset(
        dataset_index,
        input_spec,
        build_augmentation_spec(config.training),
        augmentation_enable=True,
    )
    validation_dataset = RamanDataset(dataset_index, input_spec, cache_input_enable=True)
    run_context = _open_task_run_context(
        experiment_context,
        train_task,
        request,
        runtime_spec.resume_enable,
    )
    try:
        run_context.write_log(
            f"[{train_task.model_tag}] train={len(train_task.train_indices)} "
            f"validation={len(train_task.val_indices)}"
        )
        _validate_task_samples(train_task)
        result = _run_train_task(
            train_task,
            train_dataset,
            validation_dataset,
            model_spec,
            train_spec,
            runtime_spec,
            run_context,
            request,
        )
        entry = _build_run_entry(experiment_context, run_context, result, train_task)
        _record_train_entry(level_models, parent_models, train_task, entry)
    finally:
        run_context.close()

    merged_meta = merge_hierarchy_meta(
        load_hierarchy_meta(experiment_context.hierarchy_meta_path),
        current_meta,
    )
    write_hierarchy_meta(experiment_context.hierarchy_meta_path, merged_meta)
    return merged_meta


def build_train_artifacts(run_context: RunContext) -> TrainArtifacts:
    """从单 run 上下文提取训练循环所需的固定产物路径。"""
    return TrainArtifacts(
        model_path=run_context.model_path,
        se_stats_path=run_context.se_stats_path,
        checkpoint_path=run_context.checkpoint_path,
        diagnostic_path=run_context.diagnostic_path,
        metrics_path=run_context.metrics_path,
    )


def _resolve_experiment_dir(request: TrainRequest, selection: DatasetSelection) -> Path:
    """解析当前训练使用的实验根目录，不写回配置。"""
    if request.experiment_dir is not None:
        return resolve_path(request.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "output" / selection.profile.dataset_name / timestamp


def _resolve_train_dir(request: TrainRequest, selection: DatasetSelection) -> Path:
    """解析配置数据根目录下的训练目录，也支持直接指定训练目录。"""
    if request.train_dir is not None:
        train_dir = resolve_path(request.train_dir)
        if not train_dir.is_dir():
            raise FileNotFoundError(f"缺少显式训练目录：{train_dir}")
        return train_dir
    return selection.train_dir


def _set_random_seed(seed: int, deterministic_enable: bool) -> None:
    """设置 Python、NumPy 与 PyTorch 的训练随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_enable:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 关闭确定性时允许 cuDNN 自动挑选卷积算法，输入形状固定可稳定复用结果。
        torch.backends.cudnn.benchmark = True


def _build_train_task(
    dataset_index: DatasetIndex,
    level_name: str,
    parent_id: int | None,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> TrainTask:
    """构建本次训练唯一的全局或父类子模型任务。"""
    if parent_id is None:
        return build_global_train_task(
            dataset_index,
            level_name,
            train_indices,
            validation_indices,
        )
    train_task = build_parent_train_task(
        dataset_index,
        level_name,
        parent_id,
        train_indices,
        validation_indices,
    )
    if train_task is None:
        raise ValueError(
            f"{level_name} 的父类 {parent_id} 只有一个子类，推理会直接继承，无需训练"
        )
    return train_task


def _open_task_run_context(
    experiment_context: ExperimentContext,
    train_task: TrainTask,
    request: TrainRequest,
    resume_training_enable: bool,
) -> RunContext:
    """为一个训练任务计算槽位目录并打开对应 run 上下文。"""
    run_dir = resolve_run_dir(
        experiment_context,
        train_task.level_name,
        train_task.parent_id,
        request.resume_run_dir,
    )
    resume_enable = request.resume_run_dir is not None or (
        resume_training_enable and (run_dir / CHECKPOINT_FILE_NAME).is_file()
    )
    return open_run_context(
        experiment_context,
        run_dir,
        resume_enable=resume_enable,
    )


def _validate_task_samples(train_task: TrainTask) -> None:
    """确保每个实际训练任务同时拥有训练和验证样本。"""
    if len(train_task.train_indices) == 0:
        raise ValueError(f"{train_task.model_tag} 没有可用训练样本")
    if len(train_task.val_indices) == 0:
        raise ValueError(f"{train_task.model_tag} 没有可用验证样本")


def _run_train_task(
    train_task: TrainTask,
    train_dataset: RamanDataset,
    validation_dataset: RamanDataset,
    model_spec: ModelSpec,
    train_spec: TrainingSpec,
    runtime_spec: ExecutionSpec,
    run_context: RunContext,
    request: TrainRequest,
) -> TrainResult:
    """构建一个模型及其数据 loader，并执行单任务训练循环。"""
    device = _resolve_loader_device(runtime_spec)
    train_loader = build_loader(
        Subset(train_dataset, train_task.train_indices.tolist()),
        train_spec.train_loader,
        device,
    )
    validation_loader = build_loader(
        Subset(validation_dataset, train_task.val_indices.tolist()),
        train_spec.validation_loader,
        device,
    )
    model = build_model(train_task.num_classes, model_spec)
    if request.initialize_model is not None:
        request.initialize_model(model, train_task)
    return run_train_loop(
        model,
        train_loader,
        validation_loader,
        train_task,
        train_spec,
        runtime_spec,
        build_train_artifacts(run_context),
        run_context.write_log,
        request.apply_training_mode,
    )


def _resolve_loader_device(runtime_spec: ExecutionSpec) -> torch.device:
    """按运行规格选择 DataLoader 的设备相关参数。"""
    use_cuda_enable = runtime_spec.use_gpu_enable and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda_enable else "cpu")


def _build_run_entry(
    experiment_context: ExperimentContext,
    run_context: RunContext,
    train_result: TrainResult,
    train_task: TrainTask,
) -> dict[str, Any]:
    """将单任务训练结果转换为可供后续读取的相对路径元数据条目。"""
    experiment_dir = experiment_context.experiment_dir
    return build_hierarchy_entry(
        relpath(run_context.run_dir, experiment_dir),
        relpath(train_result.model_path, experiment_dir),
        log_path=relpath(run_context.log_path, experiment_dir),
        trained_at=datetime.now().isoformat(timespec="seconds"),
        child_ids=(
            None if train_task.parent_id is None else list(train_task.visible_class_ids)
        ),
    )


def _record_train_entry(
    level_models: dict[str, dict[str, Any]],
    parent_models: dict[str, dict[str, dict[str, Any]]],
    train_task: TrainTask,
    entry: dict[str, Any],
) -> None:
    """将一个训练完成条目写入全局层或父类子模型索引。"""
    if train_task.parent_id is None:
        level_models[train_task.level_name] = entry
        return
    parent_models.setdefault(train_task.level_name, {})[str(train_task.parent_id)] = entry
