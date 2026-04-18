import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import torch

from raman.config_io import load_experiment
from raman.model import RamanClassifier1D

from .experiment import (
    load_hierarchy_meta,
    resolve_level_model_path,
    resolve_project_path,
    scan_parent_model_files,
)


@dataclass
class ExperimentRuntime:
    """统一管理实验目录下模型的懒加载与缓存"""

    exp_dir: str
    config: object
    device: torch.device
    meta: dict
    level_model_paths: dict[str, str] = field(default_factory=dict)
    parent_models: dict[str, dict] = field(default_factory=dict)
    class_names_by_level: dict[str, list] = field(default_factory=dict)
    parent_to_children: dict[str, dict] = field(default_factory=dict)
    level_model_cache: dict[str, object] = field(default_factory=dict)
    parent_model_cache: dict[tuple[str, int], object] = field(default_factory=dict)

    def _load_model(self, model_path, num_classes):
        model = RamanClassifier1D(num_classes=num_classes, config=self.config).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def build_level_model_paths(self, level_order):
        """为给定层级顺序补齐全局模型路径"""
        level_models_meta = self.meta.get("level_models", {})
        for level_name in level_order:
            self.level_model_paths[level_name] = resolve_level_model_path(
                self.exp_dir,
                level_name,
                level_models_meta,
            )
        return {level_name: self.level_model_paths[level_name] for level_name in level_order}

    def ensure_parent_models(self, level_name, fallback_parent_to_children=None):
        """确保某层 parent 模型映射完整，并在缺失时扫描实验目录补齐"""
        current = self.parent_models.setdefault(level_name, {})
        fallback = fallback_parent_to_children if fallback_parent_to_children is not None else self.parent_to_children
        scanned = scan_parent_model_files(self.exp_dir, level_name, fallback)

        class_names = self.class_names_by_level.get(level_name, [])
        for parent_idx, scanned_entry in scanned.items():
            entry = dict(current.get(parent_idx, {}))
            if not entry.get("child_ids"):
                entry["child_ids"] = list(scanned_entry.get("child_ids", []))
            if entry.get("model_path") is None:
                entry["model_path"] = scanned_entry.get("model_path")
            if "child_names" not in entry and class_names and entry.get("child_ids"):
                entry["child_names"] = [
                    class_names[child_id]
                    for child_id in entry["child_ids"]
                    if 0 <= int(child_id) < len(class_names)
                ]
            current[int(parent_idx)] = entry

        return current

    def get_level_model(self, level_name, num_classes=None):
        """懒加载某层全局模型"""
        if level_name in self.level_model_cache:
            return self.level_model_cache[level_name]

        if num_classes is None:
            num_classes = len(self.class_names_by_level.get(level_name, []))
        if not num_classes:
            raise ValueError(f"Cannot infer num_classes for level '{level_name}'.")

        model_path = self.level_model_paths.get(level_name)
        if not model_path:
            model_path = resolve_level_model_path(
                self.exp_dir,
                level_name,
                self.meta.get("level_models", {}),
            )
            self.level_model_paths[level_name] = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for level '{level_name}': {model_path}")

        model = self._load_model(model_path, int(num_classes))
        self.level_model_cache[level_name] = model
        return model

    def load_single_level_model(self, level_name, num_classes=None):
        """给单层分析/诊断场景提供统一入口"""
        return self.get_level_model(level_name, num_classes=num_classes)

    def get_parent_model(self, level_name, parent_idx, child_ids=None, model_path=None):
        """懒加载某个 parent 子模型"""
        key = (level_name, int(parent_idx))
        if key in self.parent_model_cache:
            return self.parent_model_cache[key]

        entry = self.ensure_parent_models(level_name).get(int(parent_idx), {})
        if child_ids is None:
            child_ids = entry.get("child_ids", [])
        if model_path is None:
            model_path = entry.get("model_path")

        if not child_ids:
            raise ValueError(
                f"Missing child_ids for level='{level_name}', parent={parent_idx}."
            )
        if model_path is None:
            raise FileNotFoundError(
                f"Missing parent model path for level='{level_name}', parent={parent_idx}."
            )

        full_path = Path(model_path)
        if not full_path.is_absolute():
            full_path = Path(self.exp_dir) / full_path
        if not full_path.exists():
            raise FileNotFoundError(f"Parent model not found: {full_path}")

        model = self._load_model(os.fspath(full_path), len(child_ids))
        self.parent_model_cache[key] = model
        return model


def build_experiment_runtime(exp_dir, device, config=None, meta=None):
    """构建统一的实验运行时上下文"""
    exp_dir = os.fspath(resolve_project_path(exp_dir))
    if config is None:
        config = load_experiment(exp_dir)
    if meta is None:
        meta = load_hierarchy_meta(exp_dir) or {}

    return ExperimentRuntime(
        exp_dir=exp_dir,
        config=config,
        device=device,
        meta=meta,
        level_model_paths={},
        parent_models=deepcopy(meta.get("parent_models", {})),
        class_names_by_level=deepcopy(meta.get("class_names_by_level", {})),
        parent_to_children=deepcopy(meta.get("parent_to_children", {})),
        level_model_cache={},
        parent_model_cache={},
    )
