"""Raman 层级分类训练脚本"""
from copy import deepcopy
import json
import os
from dataclasses import dataclass
from datetime import datetime
import torch

from raman.config import config as default_config
from raman.data import RamanDataset
from raman.tool.hierarchy import resolve_level_order
from raman.tool.path import relpath
from raman.training.session import (
    prepare_run_runtime,
    prepare_training_runtime,
    save_hierarchy_meta,
    set_seed,
)
from raman.training.model_loop import ModelTrainContext, train_model
from raman.training.split import (
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
    DEFAULT_SPLIT_LEVEL,
    apply_train_filter,
    build_label_map_np,
    load_split_files,
    log_split_summary,
    resolve_levels_to_train,
    resolve_train_scope,
    save_split_files,
    split_by_lowest_level_ratio,
    split_files_hash,
)

@dataclass
class TrainOverrides:
    """训练入口覆盖项，供根目录脚本和 Colab 统一复用"""

    current_train_level: str | None = None
    train_only_parent_name: str | None = None
    train_only_parent: int | None = None
    override_align_loss_weight: float | None = None
    override_supcon_tau: float | None = None
    override_supcon_loss_weight: float | None = None
    override_output_dir: str | None = None


def apply_train_overrides(config, overrides=None):
    """把入口层的手动覆盖项写回配置对象"""
    overrides = overrides or TrainOverrides()

    if overrides.current_train_level is not None:
        config.current_train_level = overrides.current_train_level
    if overrides.train_only_parent_name is not None:
        config.train_only_parent_name = overrides.train_only_parent_name
    if overrides.train_only_parent is not None:
        config.train_only_parent = overrides.train_only_parent

    if overrides.override_align_loss_weight is not None:
        config.align_loss_weight = float(overrides.override_align_loss_weight)
    if overrides.override_supcon_tau is not None:
        config.supcon_tau = float(overrides.override_supcon_tau)
    if overrides.override_supcon_loss_weight is not None:
        config.supcon_loss_weight = float(overrides.override_supcon_loss_weight)
    if overrides.override_output_dir is not None:
        config.output_dir = str(overrides.override_output_dir)
    return config


def _load_existing_hierarchy_meta(exp_dir):
    """读取已有实验目录中的层级训练元数据"""
    meta_path = os.path.join(exp_dir, "hierarchy_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _resolve_saved_model_path(exp_dir, model_path):
    """把层级元数据中的模型路径解析成绝对路径"""
    if isinstance(model_path, dict):
        model_path = model_path.get("model_path")
    if not model_path:
        return None
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(exp_dir, model_path)


def _run_timestamp():
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _model_tag(level_name, parent_idx=None):
    if parent_idx is None:
        return level_name
    return f"{level_name}_{int(parent_idx)}"


def _run_dir(exp_dir, level_name, run_name, parent_idx=None):
    if parent_idx is None:
        return os.path.join(exp_dir, level_name, run_name)
    return os.path.join(exp_dir, level_name, _model_tag(level_name, parent_idx), run_name)


def _parent_name(full_dataset, parent_level_idx, parent_idx):
    if parent_level_idx is None:
        return None
    inv = full_dataset.inv_label_maps_by_level[parent_level_idx]
    return inv.get(int(parent_idx))


def _relative_entry_path(exp_dir, path):
    return relpath(path, exp_dir) if path else None


def _run_entry(exp_dir, run_dir, result, *, child_ids=None, child_names=None, status="trained"):
    model_path = result.get("best_model_path") if result else None
    entry = {
        "run_dir": _relative_entry_path(exp_dir, run_dir),
        "config_path": _relative_entry_path(exp_dir, os.path.join(run_dir, "model_config.yaml")),
        "resolved_config_path": _relative_entry_path(exp_dir, os.path.join(run_dir, "resolved_config.yaml")),
        "model_path": _relative_entry_path(exp_dir, model_path),
        "train_split_path": TRAIN_SPLIT_NAME,
        "val_split_path": VAL_SPLIT_NAME,
        "split_hash": split_files_hash(exp_dir),
        "log_path": _relative_entry_path(exp_dir, os.path.join(run_dir, "logs", "run.log")),
        "trained_at": os.path.basename(os.path.normpath(run_dir)).replace("run_", ""),
        "status": status,
    }
    if child_ids is not None:
        entry["child_ids"] = list(child_ids)
    if child_names is not None:
        entry["child_names"] = list(child_names)
    return entry


def _close_run_files(log_file, config_log_file):
    if config_log_file is not None:
        config_log_file.close()
    if log_file is not None:
        log_file.close()






















def _is_parent_entry_covered(exp_dir, entry):
    """判断某个 parent 条目是否足以支持级联"""
    child_ids = list(entry.get("child_ids", []))
    model_path = entry.get("model_path")
    if model_path:
        full_path = _resolve_saved_model_path(exp_dir, model_path)
        return full_path is not None and os.path.exists(full_path)
    return len(child_ids) <= 1






def _has_level_coverage(exp_dir, full_dataset, level_name):
    """判断实验目录里是否已有某一上级层的完整训练结果"""
    meta = _load_existing_hierarchy_meta(exp_dir)
    if meta is not None:
        level_models = meta.get("level_models", {})
        model_name = level_models.get(level_name)
        model_path = _resolve_saved_model_path(exp_dir, model_name)
        if model_path is not None and os.path.exists(model_path):
            return True

        parent_entries = meta.get("parent_models", {}).get(level_name, {})
        expected_mapping = full_dataset.parent_to_children.get(level_name, {})
        if expected_mapping:
            all_covered = True
            for parent_idx in expected_mapping.keys():
                entry = parent_entries.get(str(parent_idx))
                if entry is None:
                    entry = parent_entries.get(int(parent_idx))
                if entry is None or not _is_parent_entry_covered(exp_dir, entry):
                    all_covered = False
                    break
            if all_covered:
                return True
        elif parent_entries:
            if any(_is_parent_entry_covered(exp_dir, entry) for entry in parent_entries.values()):
                return True

    default_level_model = os.path.join(exp_dir, level_name, f"{level_name}_model.pt")
    if os.path.exists(default_level_model):
        return True

    prefix = f"{level_name}_parent_"
    suffix = "_model.pt"
    level_dir = os.path.join(exp_dir, level_name)
    if not os.path.isdir(level_dir):
        return False
    for name in os.listdir(level_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            return True
    return False


def _log_missing_upper_level_hint(full_dataset, current_train_level, train_per_parent, exp_dir, log):
    """缺少上一级结果时给出非阻断提示"""
    if not train_per_parent:
        return

    upper_level_name = full_dataset.get_parent_level(current_train_level)
    if upper_level_name is None:
        return

    if _has_level_coverage(exp_dir, full_dataset, upper_level_name):
        return

    log(
        f"[Hint] train_per_parent=True，但当前实验目录缺少上一级 "
        f"{upper_level_name} 的完整模型记录"
    )
    log(
        f"[Hint] 若后续需要在同一 EXP_DIR 中继续向下训练、级联预测或验证评估，"
        f"建议先训练 current_train_level={upper_level_name}"
    )


# 逐层训练（全局层或父类子模型）
def run_training(config_obj=None, overrides=None):
    """执行一次完整训练流程"""
    config = deepcopy(config_obj or default_config)
    config = apply_train_overrides(config, overrides)
    current_train_level = getattr(config, "current_train_level", None)
    if current_train_level is None:
        raise ValueError("训练入口必须显式提供 current_train_level")
    TRAIN_PER_PARENT = config.train_per_parent
    USE_ALIGN_LOSS = getattr(config, "use_align_loss", True)
    USE_SUPCON_LOSS = getattr(config, "use_supcon_loss", True)
    ALIGN_LOSS_WEIGHT = getattr(config, "align_loss_weight", 0.05)
    SUPCON_LOSS_WEIGHT = getattr(config, "supcon_loss_weight", 0.03)

    # 读取对齐损失和 SupCon 的衰减起点比例，并写回配置，便于完整记录实验参数
    decay_start_ratio = float(config.decay_start_ratio)
    decay_start_ratio = min(max(decay_start_ratio, 0.0), 1.0)
    config.decay_start_ratio = decay_start_ratio

    runtime_dirs, log, _, log_file, config_log_file = prepare_training_runtime(config)
    exp_dir = config.output_dir

    # 设备与随机种子
    use_cuda = (config.use_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"Using device: {device} (config.use_gpu={config.use_gpu}, cuda_available={torch.cuda.is_available()})")
    set_seed(config.seed, deterministic=config.deterministic)
    log(f"Seed set to {config.seed} (deterministic={config.deterministic})")
    log(f"Decay start ratio set to {decay_start_ratio:.3f}")

    def zero_loss(feat):
        return torch.tensor(0.0, device=feat.device)

    # 构建完整数据集对象，用于读取全局层级和类别信息
    full_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config)

    head_names = full_dataset.head_names
    business_head_names = full_dataset.level_names
    head_name_to_idx = full_dataset.head_name_to_idx
    # 解析当前训练层级
    current_train_level, _ = resolve_level_order(
        full_dataset,
        current_train_level,
        field_name="current_train_level",
    )
    if current_train_level not in head_name_to_idx:
        raise ValueError(f"Unknown current_train_level: {current_train_level}")
    _log_missing_upper_level_hint(
        full_dataset,
        current_train_level,
        TRAIN_PER_PARENT,
        config.output_dir,
        log,
    )
    existing_split = load_split_files(full_dataset, exp_dir)
    if existing_split is not None:
        train_idx, val_idx = existing_split
        log(f"[Split] reuse {TRAIN_SPLIT_NAME}/{VAL_SPLIT_NAME} from {exp_dir}")
    else:
        log(
            "[Split] generate "
            f"split_level={DEFAULT_SPLIT_LEVEL}, "
            f"split_by_source_prefix={getattr(config, 'split_by_source_prefix', False)}"
        )
        train_idx, val_idx = split_by_lowest_level_ratio(
            full_dataset,
            lowest_level=DEFAULT_SPLIT_LEVEL,
            train_ratio=config.train_split,
            seed=config.seed,
            split_by_source_prefix=getattr(config, "split_by_source_prefix", False),
        )
        train_idx = torch.as_tensor(train_idx, dtype=torch.long).numpy()
        val_idx = torch.as_tensor(val_idx, dtype=torch.long).numpy()
        save_split_files(full_dataset, train_idx, val_idx, exp_dir)
        log(f"[Split] saved {TRAIN_SPLIT_NAME}/{VAL_SPLIT_NAME} to {exp_dir}")
    only_parent = resolve_train_scope(
        full_dataset,
        config,
        current_train_level,
        head_name_to_idx,
    )
    train_idx, val_idx = apply_train_filter(
        full_dataset,
        train_idx,
        val_idx,
        config,
        head_name_to_idx,
    )

    log_split_summary(full_dataset, train_idx, val_idx, current_train_level, head_name_to_idx)

    # 构建训练和验证数据集，并保持层级映射一致
    train_dataset = RamanDataset(
        config.dataset_root,
        augment=True,
        config=config
    )
    val_dataset = RamanDataset(
        config.dataset_root,
        augment=False,
        config=config
    )

    # 决定要训练哪些层级
    levels_to_train = resolve_levels_to_train(current_train_level)

    level_models = {}
    parent_models = {}
    run_name = _run_timestamp()

    def build_train_context(run_config, run_dirs, run_log):
        return ModelTrainContext(
            config=run_config,
            log=run_log,
            runtime_dirs=run_dirs,
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            full_dataset=full_dataset,
            head_names=head_names,
            use_align_loss=USE_ALIGN_LOSS,
            use_supcon_loss=USE_SUPCON_LOSS,
            align_loss_weight=ALIGN_LOSS_WEIGHT,
            supcon_loss_weight=SUPCON_LOSS_WEIGHT,
            decay_start_ratio=decay_start_ratio,
            zero_loss=zero_loss,
        )

    for level_name in levels_to_train:
        log(f"[{level_name}] ==================== LEVEL START ====================")
        level_idx = head_name_to_idx[level_name]
        parent_name = full_dataset.get_parent_level(level_name)
        parent_level_idx = head_name_to_idx[parent_name] if parent_name else None
        parent_to_children = full_dataset.parent_to_children.get(level_name, {})
        # 顶层或非按父类训练：训练全局单模型
        if (parent_name is None) or (not TRAIN_PER_PARENT):
            run_dir = _run_dir(exp_dir, level_name, run_name, parent_idx=None)
            run_config = deepcopy(config)
            run_config.output_dir = run_dir
            run_config.experiment_dir = exp_dir
            run_config.run_dir = run_dir
            run_dirs, run_log, _, run_log_file, run_config_log_file = prepare_run_runtime(
                run_config,
                run_dir,
            )
            run_log(f"[{level_name}] run_dir = {run_dir}")
            result = train_model(
                build_train_context(run_config, run_dirs, run_log),
                model_tag=level_name,
                level_name=level_name,
                level_idx=level_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                num_classes=full_dataset.num_classes_by_level[level_name],
                parent_level_idx=parent_level_idx,
                parent_to_children=parent_to_children,
                label_map_np=None,
                use_parent_mask=False,
            )
            _close_run_files(run_log_file, run_config_log_file)
            if result is None:
                continue
            level_models[level_name] = _run_entry(exp_dir, run_dir, result)
        else:
            # 父类内子类独立模型
            parent_models[level_name] = {}
            target_parent_idx = int(only_parent) if only_parent is not None else None

            for parent_idx, child_ids in parent_to_children.items():
                if target_parent_idx is not None and int(parent_idx) != target_parent_idx:
                    continue
                child_ids = list(child_ids)
                if not child_ids:
                    continue

                child_names = [
                    full_dataset.class_names_by_level[level_idx][cid]
                    for cid in child_ids
                ]
                labels_train = full_dataset.level_labels[train_idx]
                labels_val = full_dataset.level_labels[val_idx]
                train_mask = (labels_train[:, parent_level_idx] == parent_idx) & (
                    labels_train[:, level_idx] >= 0
                )
                val_mask = (labels_val[:, parent_level_idx] == parent_idx) & (
                    labels_val[:, level_idx] >= 0
                )
                train_indices = train_idx[train_mask]
                val_indices = val_idx[val_mask]
                parent_label = _parent_name(full_dataset, parent_level_idx, parent_idx)

                # 只有一个子类时不需要训练
                if len(child_ids) <= 1:
                    log(
                        f"[{level_name}] parent={parent_idx} name={parent_label} "
                        f"only one child={child_names}, skip training without run_dir"
                    )
                    parent_models[level_name][parent_idx] = {
                        "model_path": None,
                        "child_ids": child_ids,
                        "child_names": child_names,
                        "status": "skipped_single_child",
                    }
                    continue

                run_dir = _run_dir(exp_dir, level_name, run_name, parent_idx=parent_idx)
                run_config = deepcopy(config)
                run_config.output_dir = run_dir
                run_config.experiment_dir = exp_dir
                run_config.run_dir = run_dir
                run_config.train_only_parent = int(parent_idx)
                run_dirs, run_log, _, run_log_file, run_config_log_file = prepare_run_runtime(
                    run_config,
                    run_dir,
                )

                local_train_labels = full_dataset.level_labels[train_indices, level_idx]
                local_val_labels = full_dataset.level_labels[val_indices, level_idx]
                run_log(
                    f"[{level_name}] parent={parent_idx} name={parent_label} "
                    f"child_names={child_names}"
                )
                run_log(f"[{level_name}] run_dir = {run_dir}")
                run_log(
                    f"[{level_name}] train={len(train_indices)} val={len(val_indices)} "
                    f"child_ids={child_ids}"
                )
                run_log(
                    f"[{level_name}] local train counts = "
                    f"{torch.bincount(torch.as_tensor(local_train_labels), minlength=full_dataset.num_classes_by_level[level_name]).numpy()[child_ids]}"
                )
                run_log(
                    f"[{level_name}] local val counts = "
                    f"{torch.bincount(torch.as_tensor(local_val_labels), minlength=full_dataset.num_classes_by_level[level_name]).numpy()[child_ids]}"
                )

                label_map_np = build_label_map_np(
                    child_ids,
                    full_dataset.num_classes_by_level[level_name]
                )

                result = train_model(
                    build_train_context(run_config, run_dirs, run_log),
                    model_tag=_model_tag(level_name, parent_idx),
                    level_name=level_name,
                    level_idx=level_idx,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    num_classes=len(child_ids),
                    parent_level_idx=None,
                    parent_to_children=None,
                    label_map_np=label_map_np,
                    use_parent_mask=False,
                )
                _close_run_files(run_log_file, run_config_log_file)

                if result is None:
                    continue

                parent_models[level_name][parent_idx] = _run_entry(
                    exp_dir,
                    run_dir,
                    result,
                    child_ids=child_ids,
                    child_names=child_names,
                )

    save_hierarchy_meta(
        config,
        full_dataset,
        business_head_names,
        current_train_level,
        level_models,
        parent_models,
    )

    _close_run_files(log_file, config_log_file)

    return {
        "output_dir": exp_dir,
        "current_train_level": current_train_level,
        "levels_to_train": levels_to_train,
        "run_dirs": [
            entry.get("run_dir")
            for entry in level_models.values()
            if isinstance(entry, dict) and entry.get("run_dir")
        ] + [
            entry.get("run_dir")
            for mapping in parent_models.values()
            for entry in mapping.values()
            if isinstance(entry, dict) and entry.get("run_dir")
        ],
    }


def main():
    """提示用户改用根目录训练入口"""
    raise SystemExit("请使用根目录 train.py，并在入口里显式设置 CURRENT_TRAIN_LEVEL")


if __name__ == "__main__":
    main()
