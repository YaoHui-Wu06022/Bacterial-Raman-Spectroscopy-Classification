import json
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from raman.data.profiles import get_profile
from raman.config_io import (
    SHARED_CONFIG_NAME,
    assert_shared_compatible,
    dump_model_config_to_yaml,
    dump_resolved_config_to_yaml,
    dump_shared_config_to_yaml,
    load_config_from_yaml,
)


def _sanitize_log_name(name):
    """将日志名裁剪到适合文件名的形式"""
    text = str(name).strip()
    if not text:
        return "unnamed"
    return text.replace("\\", "_").replace("/", "_").replace(" ", "_")


def _config_group_to_dict(group):
    """把配置分组转换成逐字段字典"""
    if group is None:
        return {}
    if is_dataclass(group):
        return asdict(group)
    if hasattr(group, "to_dict"):
        return group.to_dict()
    return {
        key: value
        for key, value in vars(group).items()
        if not key.startswith("_")
    }


def _write_config_section(config_log, title, data):
    """写入一个配置分组，避免 dataclass 挤成一行"""
    config_log(f"\n===== {title} =====")
    if not data:
        config_log("  <empty>")
        return
    for key in data:
        config_log(f"  {key}: {data[key]}")


def _write_readable_config_dump(config_log, config):
    """按 shared/model/runtime 分组写入配置"""
    _write_config_section(
        config_log,
        "Derived Values",
        {
            "dataset_root": getattr(config, "dataset_root", None),
            "in_channels": getattr(config, "in_channels", None),
            "delta": getattr(config, "delta", None),
        },
    )
    _write_config_section(config_log, "Shared Input Config", _config_group_to_dict(getattr(config, "shared", None)))
    _write_config_section(config_log, "Model Run Config", _config_group_to_dict(getattr(config, "model", None)))
    _write_config_section(config_log, "Runtime Config", _config_group_to_dict(getattr(config, "runtime", None)))


def set_seed(seed, deterministic=True):
    """
    统一设置 Python、NumPy 和 PyTorch 的随机种子

    当 `deterministic=True` 时，同时关闭 cuDNN 的自动 benchmark，
    优先保证训练过程可复现
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
    创建实验根目录

    具体训练日志由每个 run_xxx/logs 独立保存，实验根不再创建 logs
    """
    base = config.output_dir
    dirs = {
        "base": base,
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def prepare_training_runtime(config):
    """准备实验根目录和 shared_config.yaml"""
    if config.timestamp is None:
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.output_dir is None:
        dataset_name = get_profile(config.dataset_name).dataset_name
        config.output_dir = os.fspath(Path("output") / dataset_name / config.timestamp)
    config.experiment_dir = config.output_dir

    dirs = prepare_output_dirs(config=config)
    shared_path = os.path.join(config.output_dir, SHARED_CONFIG_NAME)
    if os.path.exists(shared_path):
        existing_shared = load_config_from_yaml(shared_path)
        assert_shared_compatible(existing_shared, config, context="current_config")
    else:
        dump_shared_config_to_yaml(config, shared_path)

    def log(msg):
        print(msg)

    def config_log(_msg):
        return None

    return dirs, log, config_log, None, None


def prepare_run_runtime(config, run_dir):
    """准备单次模型训练 run 的目录、日志和配置快照"""
    run_dir = os.fspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    config.run_dir = run_dir

    log_file = open(
        os.path.join(logs_dir, "run.log"),
        "w",
        buffering=1,
        encoding="utf-8",
    )
    config_log_file = open(
        os.path.join(logs_dir, "config.txt"),
        "w",
        buffering=1,
        encoding="utf-8",
    )
    dump_model_config_to_yaml(config, os.path.join(run_dir, "model_config.yaml"))
    dump_resolved_config_to_yaml(config, os.path.join(run_dir, "resolved_config.yaml"))

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    def config_log(msg):
        config_log_file.write(msg + "\n")

    config_log("===== Run Meta =====")
    config_log(f"Experiment dir: {getattr(config, 'experiment_dir', config.output_dir)}")
    config_log(f"Run dir: {run_dir}")
    config_log("=====================\n")
    _write_readable_config_dump(config_log, config)

    return {"base": run_dir, "logs": logs_dir}, log, config_log, log_file, config_log_file


def create_model_logger(logs_dir, model_tag, shared_log):
    """为单个模型创建独立日志，并同步写入总日志"""
    safe_tag = _sanitize_log_name(model_tag)
    log_path = os.path.join(logs_dir, f"{safe_tag}.log")
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
    保存层级类别名、父子关系和训练得到的模型索引信息
    """
    class_names_path = os.path.join(config.output_dir, "class_names.json")
    class_names_by_level = {
        name: full_dataset.class_names_by_level[full_dataset.head_name_to_idx[name]]
        for name in head_names
    }
    with open(class_names_path, "w", encoding="utf-8") as file:
        json.dump(class_names_by_level, file, indent=2, ensure_ascii=False)

    parent_to_children_json = {
        level: {str(key): value for key, value in mapping.items()}
        for level, mapping in full_dataset.parent_to_children.items()
        if level in head_names
    }
    level_models_json = dict(level_models)
    parent_models_json = {
        level: {str(key): value for key, value in mapping.items()}
        for level, mapping in parent_models.items()
        if level in head_names
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
        old_runs = old_meta.get("runs", {}) if isinstance(old_meta, dict) else {}

        merged_level_models = {
            level: model_path
            for level, model_path in old_level_models.items()
            if level in head_names
        }
        merged_level_models.update(level_models_json)
        level_models_json = merged_level_models

        merged_parent_models = {}
        for level, mapping in old_parent_models.items():
            if level in head_names and isinstance(mapping, dict):
                merged_parent_models[level] = dict(mapping)
        for level, mapping in parent_models_json.items():
            merged_parent_models.setdefault(level, {})
            merged_parent_models[level].update(mapping)
        parent_models_json = merged_parent_models
    else:
        old_runs = {}

    runs_json = dict(old_runs)
    for level, entry in level_models_json.items():
        if isinstance(entry, dict) and entry.get("run_dir"):
            runs_json.setdefault(level, [])
            if entry not in runs_json[level]:
                runs_json[level].append(entry)
    for level, mapping in parent_models_json.items():
        for parent_idx, entry in mapping.items():
            if isinstance(entry, dict) and entry.get("run_dir"):
                key = f"{level}_{parent_idx}"
                runs_json.setdefault(key, [])
                if entry not in runs_json[key]:
                    runs_json[key].append(entry)

    with open(hier_meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "head_names": head_names,
                "level_names": head_names,
                "class_names_by_level": class_names_by_level,
                "parent_to_children": parent_to_children_json,
                "parent_level_name": {
                    level: parent
                    for level, parent in full_dataset.parent_level_name.items()
                    if level in head_names
                },
                "current_train_level": current_train_level,
                "level_models": level_models_json,
                "parent_models": parent_models_json,
                "runs": runs_json,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
