import os
from pathlib import Path

from raman.config_io import load_experiment
from raman.data import resolve_dataset_stage


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path):
    """将相对路径解析到项目根目录下。"""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_experiment_with_train_dataset(exp_dir):
    """加载实验配置，并把 dataset_root 对齐到训练阶段目录。"""
    exp_dir = _resolve_project_path(exp_dir)
    config = load_experiment(os.fspath(exp_dir))
    dataset_root = resolve_dataset_stage(
        os.fspath(_resolve_project_path(config.dataset_root)),
        stage="train",
        project_root=os.fspath(PROJECT_ROOT),
        must_exist=True,
    )
    config.dataset_root = os.fspath(dataset_root)
    return os.fspath(exp_dir), config


def resolve_head_level_name(dataset, level_name, fallback_level=None):
    """解析并校验 head 层级名。"""
    level_name = level_name or fallback_level or "leaf"
    if hasattr(dataset, "_resolve_level_name"):
        level_name = dataset._resolve_level_name(level_name)
    if level_name not in dataset.head_names:
        raise ValueError(
            f"未知层级：{level_name}，可选层级：{dataset.head_names}"
        )
    return level_name
