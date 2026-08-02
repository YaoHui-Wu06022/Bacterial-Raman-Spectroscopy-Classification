"""Stanford 预训练模型迁移到目标拉曼数据集的微调流程。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from raman.config_io import load_run_config
from raman.trainer import TrainOverrides, run_training
from raman_fine_tuning.config import FineTuneConfig


TRANSFER_INPUT_FIELDS = (
    "cut_min",
    "cut_max",
    "target_points",
    "input_grid_mode",
    "bad_bands",
    "norm_method",
    "smooth_use",
    "d1_use",
    "win_smooth",
    "win1",
)

TRANSFER_MODEL_FIELDS = (
    "backbone_type",
    "cnn_block_type",
    "cardinality",
    "base_width",
    "resnet_bottleneck_ratio",
    "identity_pool_kernel",
    "se_use",
    "reduction",
    "backbone_activation_negative_slope",
    "encoder_type",
    "transformer_nhead",
    "transformer_dim",
    "transformer_ffn_dim",
    "transformer_layers",
    "transformer_dropout",
    "lstm_hidden",
    "lstm_layers",
    "lstm_dropout",
    "lstm_bidirectional",
    "pooling_type",
    "cosine_head",
    "cosine_scale",
    "stem_kernel_sizes",
)

TAIL_MODULES = ("layer4", "proj", "transformer", "lstm", "att_pool")


def _yaml_equal(left, right):
    """把列表和元组统一后比较配置值。"""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(_yaml_equal_value(item) for item in left) == tuple(
            _yaml_equal_value(item) for item in right
        )
    return left == right


def _yaml_equal_value(value):
    if isinstance(value, (list, tuple)):
        return tuple(_yaml_equal_value(item) for item in value)
    return value


def _assert_config_compatible(source_config, target_config):
    """确保来源模型和目标数据的输入及网络结构能够逐项对齐。"""
    mismatches = []
    for field in (*TRANSFER_INPUT_FIELDS, *TRANSFER_MODEL_FIELDS):
        source_value = getattr(source_config, field)
        target_value = getattr(target_config, field)
        if not _yaml_equal(source_value, target_value):
            mismatches.append((field, source_value, target_value))
    if mismatches:
        detail = "; ".join(
            f"{field}: source={source_value!r}, target={target_value!r}"
            for field, source_value, target_value in mismatches
        )
        raise ValueError(f"预训练模型与目标配置不兼容：{detail}")

    if target_config.input_grid_mode != "stanford_transfer":
        raise ValueError("跨 Stanford 微调必须使用 input_grid_mode='stanford_transfer'")
    if target_config.norm_method != "minmax":
        raise ValueError("跨 Stanford 微调必须使用 norm_method='minmax'")


def _load_model_state(model_path):
    """读取最佳模型权重，兼容纯 state_dict 与 checkpoint 两种文件。"""
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    if not isinstance(payload, dict):
        raise TypeError(f"预训练权重不是 state_dict：{model_path}")
    return payload


class _TransferModelInitializer:
    """在模型创建后加载骨干权重，并控制可训练模块与运行模式。"""

    def __init__(self, source_model_path, unfreeze_tail, load_head=False):
        self.source_model_path = Path(source_model_path)
        self.unfreeze_tail = bool(unfreeze_tail)
        self.load_head = bool(load_head)
        self.reports = {}

    @property
    def trainable_roots(self):
        roots = {"head"}
        if self.unfreeze_tail:
            roots.update(TAIL_MODULES)
        return roots

    def initialize(self, model, level_name, model_tag, num_classes):
        """加载所有形状匹配的非分类头参数，并严格检查骨干完整性。"""
        source_state = _load_model_state(self.source_model_path)
        target_state = model.state_dict()
        loaded_state = {}
        skipped_head = []
        incompatible = []
        unexpected = []

        for key, value in source_state.items():
            if key.startswith("head.") and not self.load_head:
                skipped_head.append(key)
                continue
            target_value = target_state.get(key)
            if target_value is None:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                incompatible.append(key)
                continue
            loaded_state[key] = value

        missing_keys, unexpected_keys = model.load_state_dict(loaded_state, strict=False)
        missing_required = list(missing_keys) if self.load_head else [
            key for key in missing_keys if not key.startswith("head.")
        ]
        if missing_required or incompatible or unexpected or unexpected_keys:
            detail = {
                "missing_required": missing_required,
                "incompatible": incompatible,
                "unexpected": unexpected,
                "unexpected_after_load": list(unexpected_keys),
            }
            raise ValueError(f"预训练骨干未能完整加载：{detail}")

        for name, parameter in model.named_parameters():
            root = name.split(".", 1)[0]
            parameter.requires_grad = root in self.trainable_roots

        self.reports[model_tag] = {
            "source_model_path": str(self.source_model_path),
            "level_name": level_name,
            "target_num_classes": int(num_classes),
            "loaded_parameter_keys": len(loaded_state),
            "skipped_source_head_keys": skipped_head,
            "loaded_source_head": self.load_head,
            "trainable_modules": sorted(self.trainable_roots),
            "frozen_modules": sorted(
                name for name, _ in model.named_children() if name not in self.trainable_roots
            ),
        }

    def apply_training_mode(self, model):
        """冻结模块保持 eval，防止 BatchNorm 和 Dropout 在仅训头时继续变化。"""
        for name, module in model.named_children():
            module.train(name in self.trainable_roots)


def _source_model_path(source_run_dir, source_level, parent_idx=None):
    """根据指定 run 目录定位层级或 parent 子模型的最佳权重。"""
    model_tag = source_level if parent_idx is None else f"{source_level}_{int(parent_idx)}"
    path = Path(source_run_dir).resolve() / f"{model_tag}_model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"未找到预训练最佳模型：{path}")
    return path


def _write_transfer_reports(result, initializer, fine_tune_config, source_config, target_config):
    """将来源、加载结果和冻结范围分别写入实验根与实际 run。"""
    reports = initializer.reports
    root_payload = {
        "fine_tune_config": fine_tune_config.to_dict(),
        "source_run_dir": str(Path(fine_tune_config.source_run_dir).resolve()),
        "source_level": fine_tune_config.source_level,
        "warm_start_run_dir": (
            str(Path(fine_tune_config.warm_start_run_dir).resolve())
            if fine_tune_config.warm_start_run_dir
            else None
        ),
        "target_dataset": target_config.dataset_name,
        "target_level": fine_tune_config.current_train_level,
        "unfreeze_tail": fine_tune_config.unfreeze_tail,
        "source_norm_method": source_config.norm_method,
        "target_norm_method": target_config.norm_method,
        "input_grid_mode": target_config.input_grid_mode,
        "models": reports,
    }
    root_path = Path(result["output_dir"]) / "transfer_report.json"
    root_path.write_text(json.dumps(root_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for run_dir in result.get("run_dirs", []):
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = Path(result["output_dir"]) / run_path
        model_tag = run_path.parent.name
        payload = reports.get(model_tag)
        if payload is not None:
            (run_path / "transfer_report.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def run_fine_tuning(config, fine_tune_config: FineTuneConfig):
    """从 Stanford 最佳模型初始化目标任务，并执行一次独立的微调训练。"""
    if not isinstance(fine_tune_config, FineTuneConfig):
        raise TypeError("fine_tune_config 必须是 FineTuneConfig")

    target_config = fine_tune_config.build_target_config(config)
    source_run_dir = Path(fine_tune_config.source_run_dir).resolve()
    source_config = load_run_config(source_run_dir)
    _assert_config_compatible(source_config, target_config)
    source_model_path = _source_model_path(source_run_dir, fine_tune_config.source_level)

    load_head = False
    if fine_tune_config.warm_start_run_dir:
        warm_start_run_dir = Path(fine_tune_config.warm_start_run_dir).resolve()
        warm_config = load_run_config(warm_start_run_dir)
        _assert_config_compatible(warm_config, target_config)
        source_model_path = _source_model_path(
            warm_start_run_dir,
            fine_tune_config.current_train_level,
            parent_idx=fine_tune_config.train_only_parent,
        )
        load_head = True

    initializer = _TransferModelInitializer(
        source_model_path=source_model_path,
        unfreeze_tail=fine_tune_config.unfreeze_tail,
        load_head=load_head,
    )
    result = run_training(
        target_config,
        overrides=TrainOverrides(
            current_train_level=fine_tune_config.current_train_level,
            train_only_parent_name=fine_tune_config.train_only_parent_name,
            train_only_parent=fine_tune_config.train_only_parent,
            override_output_dir=fine_tune_config.output_dir,
        ),
        model_initializer=initializer,
    )
    _write_transfer_reports(
        result,
        initializer,
        fine_tune_config,
        source_config,
        target_config,
    )
    return result
