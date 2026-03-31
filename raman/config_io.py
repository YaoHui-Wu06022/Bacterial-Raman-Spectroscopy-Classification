import yaml
from types import SimpleNamespace
import os


def dump_config_to_yaml(config, yaml_path):
    """
    把当前 config 导出成 yaml
    """
    cfg = {}

    # dump config 本体
    for k in dir(config):
        if k.startswith("_"):
            continue

        try:
            v = getattr(config, k)
        except Exception:
            continue

        if callable(v):
            continue

        cfg[k] = v

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[Config] saved to {yaml_path}")


def load_config_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)


    # 用 namespace 构造 config
    config = SimpleNamespace(**cfg)

    return config

def load_experiment(exp_dir):
    """
    从一个实验目录加载完整配置（推荐 eval 使用）
    """
    cfg_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.yaml found in {exp_dir}")
    config = load_config_from_yaml(cfg_path)

    # 绑定实验目录
    config.output_dir = exp_dir

    # 模型路径：默认指向当前训练层级对应模型
    config.model_dir = exp_dir
    current_train_level = getattr(config, "current_train_level", None)
    if current_train_level:
        model_name = f"{current_train_level}_model.pt"
    else:
        model_name = "model.pt"
    config.model_path = os.path.join(exp_dir, model_name)

    return config
