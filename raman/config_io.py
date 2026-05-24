import os
from pathlib import Path

import yaml

from raman.config import (
    INPUT_COMPAT_FIELDS,
    MODEL_CONFIG_FIELDS,
    SHARED_CONFIG_FIELDS,
    Config,
)


# ==================== 配置文件名 ====================
# shared_config.yaml 放在实验根，保存同一实验必须一致的输入/预处理/增强配置
SHARED_CONFIG_NAME = "shared_config.yaml"
# model_config.yaml 放在每个 run_*，保存该模型可独立调整的训练和结构配置
MODEL_CONFIG_NAME = "model_config.yaml"
# resolved_config.yaml 放在每个 run_*，保存 shared + model + runtime 合成后的完整快照
RESOLVED_CONFIG_NAME = "resolved_config.yaml"


# ==================== 字典导出 ====================
def _yaml_ready(value):
    """把 tuple/list/dict 递归转换成适合 yaml 保存的结构"""
    if isinstance(value, tuple):
        return [_yaml_ready(item) for item in value]
    if isinstance(value, list):
        return [_yaml_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _yaml_ready(item) for key, item in value.items()}
    return value


def config_to_dict(config):
    """把配置对象导出为完整扁平字典"""
    if hasattr(config, "to_dict"):
        return {key: _yaml_ready(value) for key, value in config.to_dict().items()}

    cfg = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        try:
            value = getattr(config, key)
        except Exception:
            continue
        if callable(value):
            continue
        cfg[key] = _yaml_ready(value)
    return cfg


def select_config_fields(config, fields):
    """按字段列表从配置对象中取值"""
    data = {}
    for key in fields:
        try:
            data[key] = _yaml_ready(getattr(config, key))
        except Exception:
            continue
    return data


def shared_config_dict(config):
    """导出实验根 shared_config.yaml 的字段"""
    return select_config_fields(config, SHARED_CONFIG_FIELDS)


def model_config_dict(config):
    """导出单次训练 run 的 model_config.yaml 字段"""
    return select_config_fields(config, MODEL_CONFIG_FIELDS)


# ==================== YAML 写入 ====================
def dump_yaml_dict(data, yaml_path, label="Config"):
    """保存 yaml 字典，并确保父目录存在"""
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    print(f"[{label}] saved to {path}")


def dump_config_to_yaml(config, yaml_path):
    """导出完整配置"""
    dump_yaml_dict(config_to_dict(config), yaml_path, label="Config")


def dump_shared_config_to_yaml(config, yaml_path):
    """导出实验根共享配置"""
    dump_yaml_dict(shared_config_dict(config), yaml_path, label="SharedConfig")


def dump_model_config_to_yaml(config, yaml_path):
    """导出单模型训练配置"""
    dump_yaml_dict(model_config_dict(config), yaml_path, label="ModelConfig")


def dump_resolved_config_to_yaml(config, yaml_path):
    """导出 shared + model + runtime 合成后的完整配置快照"""
    dump_yaml_dict(config_to_dict(config), yaml_path, label="ResolvedConfig")


# ==================== YAML 读取和配置合成 ====================
def load_yaml_dict(yaml_path):
    """读取 yaml 文件，空文件按空字典处理"""
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config_from_yaml(yaml_path):
    """从单个 yaml 恢复 Config"""
    return Config.from_dict(load_yaml_dict(yaml_path))


def compose_config(shared=None, model=None, runtime=None, base_config=None):
    """按默认值 -> shared -> model -> runtime 的顺序合成配置"""
    base = Config.from_dict(config_to_dict(base_config)) if base_config is not None else Config()
    for payload in (shared or {}, model or {}, runtime or {}):
        for key, value in payload.items():
            setattr(base, key, value)
    if getattr(base, "scheduler_Tmax", None) is None:
        base.scheduler_Tmax = int(base.epochs)
    return base


# ==================== 实验目录解析 ====================
def find_experiment_root(path):
    """从实验根、best 或 run 目录向上寻找包含共享配置的根目录"""
    current = Path(path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / SHARED_CONFIG_NAME).exists() or (candidate / "hierarchy_meta.json").exists():
            return candidate
    return current


def load_run_config(run_dir, exp_dir=None):
    """读取单个 run 配置，并叠加实验根 shared_config.yaml"""
    run_dir = Path(run_dir).resolve()
    exp_root = Path(exp_dir).resolve() if exp_dir is not None else find_experiment_root(run_dir)
    shared_path = exp_root / SHARED_CONFIG_NAME
    model_path = run_dir / MODEL_CONFIG_NAME
    resolved_path = run_dir / RESOLVED_CONFIG_NAME

    # 新结构优先用 shared + model 合成，缺少时使用完整快照兜底
    if shared_path.exists() and model_path.exists():
        cfg = compose_config(
            shared=load_yaml_dict(shared_path),
            model=load_yaml_dict(model_path),
        )
    elif resolved_path.exists():
        cfg = load_config_from_yaml(resolved_path)
    else:
        raise FileNotFoundError(f"No model_config.yaml/resolved_config.yaml found in {run_dir}")

    # 运行期路径只在内存中补齐，不写回 model_config.yaml
    cfg.output_dir = os.fspath(exp_root)
    cfg.experiment_dir = os.fspath(exp_root)
    cfg.run_dir = os.fspath(run_dir)
    return cfg


def load_experiment(exp_dir):
    """
    加载实验根或具体 run 的配置
    - 实验根：读取 shared_config.yaml，并叠加默认模型配置
    - run 目录：读取 shared_config.yaml + model_config.yaml
    """
    path = Path(exp_dir).resolve()

    if (path / MODEL_CONFIG_NAME).exists() or (path / RESOLVED_CONFIG_NAME).exists():
        return load_run_config(path)

    # 新实验根只保存 shared_config.yaml，模型字段使用默认值等待 run 覆盖
    shared_path = path / SHARED_CONFIG_NAME
    if shared_path.exists():
        config = compose_config(shared=load_yaml_dict(shared_path))
        config.output_dir = os.fspath(path)
        config.experiment_dir = os.fspath(path)
        config.model_dir = os.fspath(path)
        return config

    raise FileNotFoundError(f"No shared_config.yaml/model_config.yaml found in {path}")


# ==================== 配置一致性校验 ====================
def input_compat_dict(config):
    """导出评估/推理时必须一致的输入兼容字段"""
    return select_config_fields(config, INPUT_COMPAT_FIELDS)


def assert_input_compatible(base_config, model_config, context="model"):
    """校验模型 run 的输入预处理配置是否匹配实验根"""
    base = input_compat_dict(base_config)
    other = input_compat_dict(model_config)
    mismatches = []
    for key in INPUT_COMPAT_FIELDS:
        if _yaml_ready(base.get(key)) != _yaml_ready(other.get(key)):
            mismatches.append((key, base.get(key), other.get(key)))
    if mismatches:
        detail = "; ".join(
            f"{key}: experiment={left!r}, {context}={right!r}"
            for key, left, right in mismatches
        )
        raise ValueError(f"输入预处理配置不一致，不能在同一实验根内混用：{detail}")


def assert_shared_compatible(base_config, other_config, context="current_config"):
    """校验继续写入同一实验根时 shared_config.yaml 是否一致"""
    base = shared_config_dict(base_config)
    other = shared_config_dict(other_config)
    mismatches = []
    for key in SHARED_CONFIG_FIELDS:
        if _yaml_ready(base.get(key)) != _yaml_ready(other.get(key)):
            mismatches.append((key, base.get(key), other.get(key)))
    if mismatches:
        detail = "; ".join(
            f"{key}: experiment={left!r}, {context}={right!r}"
            for key, left, right in mismatches
        )
        raise ValueError(f"实验根共享配置不一致，不能继续写入同一实验根：{detail}")
