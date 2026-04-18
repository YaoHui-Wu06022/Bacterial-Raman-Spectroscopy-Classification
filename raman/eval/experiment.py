import json
import os
import re
from pathlib import Path

from raman.config_io import load_experiment
from raman.data import resolve_dataset_stage


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path):
    """将相对路径解析到项目根目录，绝对路径保持不变"""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_experiment_with_dataset(exp_dir):
    """加载实验配置，并把 dataset_root 对齐到训练阶段目录"""
    exp_dir = resolve_project_path(exp_dir)
    config = load_experiment(os.fspath(exp_dir))
    dataset_root = resolve_dataset_stage(
        os.fspath(resolve_project_path(config.dataset_root)),
        stage="train",
        project_root=os.fspath(PROJECT_ROOT),
        must_exist=True,
    )
    config.dataset_root = os.fspath(dataset_root)
    return os.fspath(exp_dir), config


# 配置/字典结构的标准化
def _normalize_parent_to_children(parent_to_children):
    normalized = {}
    for level_name, mapping in (parent_to_children or {}).items():
        normalized[level_name] = {
            int(parent_idx): [int(child_id) for child_id in child_ids]
            for parent_idx, child_ids in mapping.items()
        }
    return normalized


def _normalize_parent_models(parent_models):
    normalized = {}
    for level_name, mapping in (parent_models or {}).items():
        normalized[level_name] = {}
        for parent_idx, entry in mapping.items():
            item = dict(entry)
            item["child_ids"] = [int(child_id) for child_id in item.get("child_ids", [])]
            normalized[level_name][int(parent_idx)] = item
    return normalized


def load_hierarchy_meta(exp_dir):
    """读取并规范化 hierarchy_meta.json"""
    exp_dir = resolve_project_path(exp_dir)
    meta_path = Path(exp_dir) / "hierarchy_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)

    meta["parent_to_children"] = _normalize_parent_to_children(meta.get("parent_to_children", {}))
    meta["parent_models"] = _normalize_parent_models(meta.get("parent_models", {}))
    meta["level_models"] = dict(meta.get("level_models", {}))
    meta["class_names_by_level"] = dict(meta.get("class_names_by_level", {}))
    meta["head_names"] = list(meta.get("head_names", []))
    return meta


def resolve_head_level_name(dataset, level_name):
    """解析并校验业务层级名"""
    return dataset._resolve_level_name(level_name, field_name="level_name")


def resolve_level_model_path(exp_dir, level_name, level_models_meta):
    """解析某一层全局模型文件路径"""
    exp_dir = Path(resolve_project_path(exp_dir))
    model_name = (level_models_meta or {}).get(level_name, f"{level_name}_model.pt")
    model_path = Path(model_name)
    if not model_path.is_absolute():
        model_path = exp_dir / model_path
    return os.fspath(model_path)


def scan_parent_model_files(exp_dir, level_name, parent_to_children):
    """扫描某一层 parent 子模型文件，并补齐 child_ids"""
    exp_dir = Path(resolve_project_path(exp_dir))
    if isinstance(parent_to_children, dict) and level_name in parent_to_children:
        level_mapping = parent_to_children.get(level_name, {})
    else:
        level_mapping = parent_to_children or {}

    mapping = {
        int(parent_idx): {
            "model_path": None,
            "child_ids": [int(child_id) for child_id in child_ids],
        }
        for parent_idx, child_ids in level_mapping.items()
    }

    pattern = re.compile(rf"^{re.escape(level_name)}_parent_(\d+)_model\.pt$")
    for name in os.listdir(exp_dir):
        match = pattern.match(name)
        if not match:
            continue
        parent_idx = int(match.group(1))
        entry = mapping.get(parent_idx, {"child_ids": []})
        entry["model_path"] = name
        mapping[parent_idx] = entry

    return mapping
