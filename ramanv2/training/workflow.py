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
from ramanv2.core.hierarchy_meta import (
    build_hierarchy_entry,
    build_hierarchy_meta,
    compute_split_hash,
    load_hierarchy_meta,
    merge_hierarchy_meta,
    save_hierarchy_meta,
)
from ramanv2.core.input_spec import build_input_spec
from ramanv2.core.paths import PROJECT_ROOT, relpath, resolve_path
from ramanv2.core.run_context import (
    ExperimentContext,
    RunContext,
    open_experiment_context,
    open_run_context,
    resolve_run_dir,
)
from ramanv2.data.augmentation import build_augmentation_spec
from ramanv2.data.dataset import RamanDataset
from ramanv2.data.index import DatasetIndex
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.modeling.factory import build_model, validate_model_input
from ramanv2.modeling.spec import ModelSpec, build_model_spec
from ramanv2.training.loop import TrainArtifacts, TrainResult, run_train_loop
from ramanv2.training.optimizer import build_loader
from ramanv2.training.split import (
    TrainTask,
    apply_train_filter,
    build_global_train_task,
    build_parent_train_task,
    build_train_scope,
    resolve_train_split,
)
from ramanv2.training.spec import (
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
    only_parent: int | None = None
    only_parent_name: str | None = None
    filter_level: str | None = None
    filter_value: object | None = None
    train_per_parent_enable: bool = True
    experiment_dir: Path | str | None = None
    train_dir: Path | str | None = None
    run_name: str | None = None
    resume_run_dir: Path | str | None = None
    initialize_model: Callable[[torch.nn.Module, TrainTask], None] | None = None
    apply_training_mode: Callable[[torch.nn.Module], None] | None = None


def run_training(request: TrainRequest) -> dict[str, Any]:
    """按当前配置枚举任务、训练模型并汇总层级元数据。"""
    config = request.config
    input_spec = build_input_spec(config.input)
    train_spec = build_training_spec(config.training, config.execution)
    runtime_spec = build_execution_spec(config.execution)
    model_spec = build_model_spec(config.model, input_spec)
    validate_model_input(model_spec, input_spec)
    _set_random_seed(train_spec.seed, runtime_spec.deterministic_enable)

    experiment_context = open_experiment_context(
        _resolve_experiment_dir(request, config),
        config,
    )
    dataset_index = DatasetIndex(_resolve_train_dir(request, config))
    level_name = dataset_index.resolve_level_name(request.level_name)
    train_indices, validation_indices = resolve_train_split(
        dataset_index,
        train_spec.train_ratio,
        train_spec.seed,
        experiment_context.experiment_dir,
        split_by_source_prefix_enable=train_spec.split_by_source_prefix_enable,
    )
    train_scope = build_train_scope(
        dataset_index,
        level_name,
        dataset_index.head_name_to_idx,
        only_parent=request.only_parent,
        only_parent_name=request.only_parent_name,
        filter_level=request.filter_level,
        filter_value=request.filter_value,
    )
    train_indices, validation_indices = apply_train_filter(
        dataset_index,
        train_indices,
        validation_indices,
        train_scope,
        dataset_index.head_name_to_idx,
    )
    train_tasks, skipped_entries = _build_train_tasks(
        dataset_index,
        level_name,
        train_indices,
        validation_indices,
        train_scope.only_parent,
        request.train_per_parent_enable,
    )
    _validate_resume_scope(
        request.resume_run_dir,
        train_scope.only_parent,
        request.train_per_parent_enable,
    )

    train_dataset = RamanDataset(
        dataset_index,
        input_spec,
        build_augmentation_spec(config.training),
        augmentation_enable=True,
    )
    validation_dataset = RamanDataset(dataset_index, input_spec)
    level_models: dict[str, dict[str, Any]] = {}
    parent_models: dict[str, dict[str, dict[str, Any]]] = {}
    runs: dict[str, list[dict[str, Any]]] = {}
    if skipped_entries:
        parent_models[level_name] = skipped_entries

    run_name = _build_run_name(request)
    for train_task in train_tasks:
        run_context = _open_task_run_context(
            experiment_context,
            train_task,
            run_name,
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
            runs.setdefault(train_task.model_tag, []).append(entry)
        finally:
            run_context.close()

    current_meta = build_hierarchy_meta(
        head_names=dataset_index.head_names,
        class_names_by_level={
            name: dataset_index.class_names_by_level[
                dataset_index.head_name_to_idx[name]
            ]
            for name in dataset_index.head_names
        },
        parent_to_children={
            name: {str(parent_id): child_ids for parent_id, child_ids in mapping.items()}
            for name, mapping in dataset_index.parent_to_children.items()
        },
        parent_level_name=dataset_index.parent_level_name,
        current_train_level=level_name,
        level_models=level_models,
        parent_models=parent_models,
        runs=runs,
    )
    merged_meta = merge_hierarchy_meta(
        load_hierarchy_meta(experiment_context.hierarchy_meta_path),
        current_meta,
    )
    save_hierarchy_meta(experiment_context.hierarchy_meta_path, merged_meta)
    return merged_meta


def build_train_artifacts(run_context: RunContext) -> TrainArtifacts:
    """从单 run 上下文提取训练循环所需的固定产物路径。"""
    return TrainArtifacts(
        model_path=run_context.model_path,
        se_stats_path=run_context.se_stats_path,
        checkpoint_path=run_context.checkpoint_path,
        diagnostic_path=run_context.diagnostic_path,
    )


def _resolve_experiment_dir(request: TrainRequest, config: Config) -> Path:
    """解析当前训练使用的实验根目录，不写回配置。"""
    if request.experiment_dir is not None:
        return resolve_path(request.experiment_dir)
    profile = get_profile(config.dataset.profile_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "output" / profile.dataset_name / timestamp


def _resolve_train_dir(request: TrainRequest, config: Config) -> Path:
    """解析配置数据根目录下的训练目录，也支持直接指定训练目录。"""
    if request.train_dir is not None:
        train_dir = resolve_path(request.train_dir)
        if not train_dir.is_dir():
            raise FileNotFoundError(f"缺少显式训练目录：{train_dir}")
        return train_dir
    profile = get_profile(config.dataset.profile_id)
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    train_dir = dataset_dir / profile.root_train_clean
    if train_dir.is_dir():
        return train_dir
    init_dir = dataset_dir / profile.root_init
    return init_dir if init_dir.is_dir() else dataset_dir


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


def _build_train_tasks(
    dataset_index: DatasetIndex,
    level_name: str,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    only_parent: int | None,
    train_per_parent_enable: bool,
) -> tuple[list[TrainTask], dict[str, dict[str, Any]]]:
    """构建全局或父类子模型任务，并记录无需训练的单子类父类。"""
    parent_level = dataset_index.get_parent_level(level_name)
    if parent_level is None or not train_per_parent_enable:
        return [
            build_global_train_task(
                dataset_index,
                level_name,
                train_indices,
                validation_indices,
            )
        ], {}

    tasks: list[TrainTask] = []
    skipped_entries: dict[str, dict[str, Any]] = {}
    level_index = dataset_index.head_name_to_idx[level_name]
    for parent_id, child_ids in dataset_index.parent_to_children[level_name].items():
        if only_parent is not None and int(parent_id) != only_parent:
            continue
        child_names = [
            dataset_index.class_names_by_level[level_index][child_id]
            for child_id in child_ids
        ]
        train_task = build_parent_train_task(
            dataset_index,
            level_name,
            parent_id,
            train_indices,
            validation_indices,
        )
        if train_task is None:
            skipped_entries[str(parent_id)] = build_hierarchy_entry(
                None,
                None,
                child_ids=child_ids,
                child_names=child_names,
                status="skipped_single_child",
            )
        else:
            tasks.append(train_task)
    return tasks, skipped_entries


def _validate_resume_scope(
    resume_run_dir: Path | str | None,
    only_parent: int | None,
    train_per_parent_enable: bool,
) -> None:
    """限制父类子模型恢复训练只能针对一个明确父类。"""
    if resume_run_dir is not None and train_per_parent_enable and only_parent is None:
        raise ValueError("恢复父类子模型训练时必须指定 train_only_parent")


def _build_run_name(request: TrainRequest) -> str:
    """为同一次训练生成共享的 run 目录名称。"""
    if request.run_name is not None:
        return request.run_name
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _open_task_run_context(
    experiment_context: ExperimentContext,
    train_task: TrainTask,
    run_name: str,
    request: TrainRequest,
    resume_enable: bool,
) -> RunContext:
    """为一个训练任务计算目录并打开对应 run 上下文。"""
    run_dir = resolve_run_dir(
        experiment_context,
        train_task.level_name,
        train_task.parent_id,
        run_name,
        request.resume_run_dir,
    )
    return open_run_context(
        experiment_context,
        run_dir,
        train_task.model_tag,
        request.config,
        resume_enable=resume_enable,
        task_values=_build_task_values(request, train_task),
    )


def _build_task_values(request: TrainRequest, train_task: TrainTask) -> dict[str, Any]:
    """导出当前任务范围，写入模型和完整快照。"""
    values: dict[str, Any] = {
        "level_name": train_task.level_name,
        "train_per_parent_enable": request.train_per_parent_enable,
    }
    if train_task.parent_id is not None:
        values["only_parent"] = train_task.parent_id
    if request.filter_level is not None:
        values["filter_level"] = request.filter_level
    if request.filter_value is not None:
        values["filter_value"] = request.filter_value
    return values


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
        config_path=relpath(run_context.model_config_path, experiment_dir),
        resolved_config_path=relpath(run_context.resolved_config_path, experiment_dir),
        train_split_path=experiment_context.train_split_path.name,
        val_split_path=experiment_context.validation_split_path.name,
        split_hash=compute_split_hash(
            experiment_context.train_split_path,
            experiment_context.validation_split_path,
        ),
        log_path=relpath(run_context.log_path, experiment_dir),
        trained_at=run_context.run_dir.name.removeprefix("run_"),
        child_ids=(
            None if train_task.parent_id is None else list(train_task.visible_class_ids)
        ),
        status="trained",
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
