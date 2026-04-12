import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dataset_process.profiles import get_profile
from raman.config_io import dump_config_to_yaml


def _sanitize_log_name(name):
    """将日志名裁剪到适合文件名的形式。"""
    text = str(name).strip()
    if not text:
        return "unnamed"
    return text.replace("\\", "_").replace("/", "_").replace(" ", "_")


def set_seed(seed, deterministic=True):
    """
    统一设置 Python、NumPy 和 PyTorch 的随机种子。

    当 `deterministic=True` 时，同时关闭 cuDNN 的自动 benchmark，
    优先保证训练过程可复现。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_output_dirs(config):
    """
    创建训练输出目录及其日志子目录。
    """
    base = config.output_dir
    dirs = {
        "base": base,
        "logs": os.path.join(base, "logs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def prepare_training_runtime(config):
    """
    准备实验目录、日志文件和配置快照。
    """
    if config.timestamp is None:
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_dir is None:
        dataset_name = get_profile(config.dataset_name).dataset_name
        config.output_dir = os.fspath(Path("output") / dataset_name / config.timestamp)

    dirs = prepare_output_dirs(config=config)
    dump_config_to_yaml(config, os.path.join(config.output_dir, "config.yaml"))

    run_tag = _sanitize_log_name(config.timestamp)
    log_file = open(
        os.path.join(dirs["logs"], f"run_{run_tag}.log"),
        "w",
        buffering=1,
    )
    config_log_file = open(
        os.path.join(dirs["logs"], f"config_{run_tag}.txt"),
        "w",
        buffering=1,
    )

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    def config_log(msg):
        config_log_file.write(msg + "\n")

    config_log("===== Run Meta =====")
    config_log(f"Experiment timestamp: {config.timestamp}")
    config_log(f"Output dir: {config.output_dir}")
    config_log("=====================\n")
    config_log("===== Full Config Dump =====")

    for key in sorted(dir(config)):
        if key.startswith("_"):
            continue
        try:
            value = getattr(config, key)
        except Exception:
            continue
        if callable(value):
            continue
        config_log(f"{key}: {value}")

    return dirs, log, config_log, log_file, config_log_file


def create_model_logger(logs_dir, model_tag, run_tag, shared_log):
    """为单个模型创建独立日志，并同步写入总日志。"""
    safe_tag = _sanitize_log_name(model_tag)
    safe_run_tag = _sanitize_log_name(run_tag)
    log_path = os.path.join(logs_dir, f"{safe_tag}_{safe_run_tag}.log")
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")

    def model_log(msg):
        shared_log(msg)
        log_file.write(msg + "\n")

    return log_path, model_log, log_file


def save_hierarchy_meta(
    config,
    full_dataset,
    head_names,
    current_train_level,
    level_models,
    parent_models,
):
    """
    保存层级类别名、父子关系和训练得到的模型索引信息。
    """
    class_names_path = os.path.join(config.output_dir, "class_names.json")
    class_names_by_level = {
        name: full_dataset.class_names_by_level[i]
        for i, name in enumerate(head_names)
    }
    with open(class_names_path, "w", encoding="utf-8") as file:
        json.dump(class_names_by_level, file, indent=2, ensure_ascii=False)

    parent_to_children_json = {
        level: {str(key): value for key, value in mapping.items()}
        for level, mapping in full_dataset.parent_to_children.items()
    }
    level_models_json = dict(level_models)
    parent_models_json = {
        level: {str(key): value for key, value in mapping.items()}
        for level, mapping in parent_models.items()
    }

    hier_meta_path = os.path.join(config.output_dir, "hierarchy_meta.json")
    if os.path.exists(hier_meta_path):
        try:
            with open(hier_meta_path, "r", encoding="utf-8") as file:
                old_meta = json.load(file)
        except Exception:
            old_meta = {}

        old_level_models = (
            old_meta.get("level_models", {}) if isinstance(old_meta, dict) else {}
        )
        old_parent_models = (
            old_meta.get("parent_models", {}) if isinstance(old_meta, dict) else {}
        )

        merged_level_models = dict(old_level_models)
        merged_level_models.update(level_models_json)
        level_models_json = merged_level_models

        merged_parent_models = {}
        for level, mapping in old_parent_models.items():
            if isinstance(mapping, dict):
                merged_parent_models[level] = dict(mapping)
        for level, mapping in parent_models_json.items():
            merged_parent_models.setdefault(level, {})
            merged_parent_models[level].update(mapping)
        parent_models_json = merged_parent_models

    with open(hier_meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "head_names": head_names,
                "level_names": full_dataset.level_names,
                "class_names_by_level": class_names_by_level,
                "parent_to_children": parent_to_children_json,
                "parent_level_name": full_dataset.parent_level_name,
                "current_train_level": current_train_level,
                "level_models": level_models_json,
                "parent_models": parent_models_json,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
