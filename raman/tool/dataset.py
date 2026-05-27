"""数据集阶段识别和 .arc_data 遍历工具"""

import os
from pathlib import Path

from raman.tool.path import PROJECT_ROOT


DATASET_BUNDLE_STAGE_MAP = {
    "train": ("train",),
    "test": ("test",),
    "init": ("init",),
    "init_test": ("init_test",),
    "predict_input": ("test",),
    "train_fig": ("fig_train",),
    "test_fig": ("fig_test",),
}


def resolve_dataset(dataset, project_root=PROJECT_ROOT, create=False):
    """按 profile id 或数据集名称解析 profile 和 dataset 目录"""
    from raman.data.profiles import get_dataset_dir, get_profile

    profile = get_profile(dataset)
    dataset_dir = get_dataset_dir(profile, project_root)
    if create:
        dataset_dir.mkdir(parents=True, exist_ok=True)
    return profile, dataset_dir


def iter_arc_dirs(root_dir):
    """递归遍历目录树，只返回包含 .arc_data 文件的叶子目录"""
    root_dir = os.fspath(root_dir)
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [name for name in files if name.lower().endswith(".arc_data")]
        if arc_files:
            yield Path(root), arc_files


def _coerce_path(path_value):
    if isinstance(path_value, Path):
        return path_value
    return Path(path_value)


def _is_dataset_bundle_dir(path):
    """判断路径是否是包含 train/test/init 等阶段目录的数据集根"""
    if not path.is_dir():
        return False
    return any(
        (path / child).exists()
        for children in DATASET_BUNDLE_STAGE_MAP.values()
        for child in children
    )


def dataset_bundle_root(path):
    """把 train/test/init/init_test 等阶段目录还原到数据集根目录"""
    path = Path(path)
    if path.name in {"train", "test", "init", "init_test"}:
        return path.parent
    return path


def resolve_dataset_stage(path_value, stage="train", project_root=None, must_exist=False):
    """解析数据集根或阶段目录，返回指定阶段的实际目录"""
    path = _coerce_path(path_value)
    root = Path(project_root) if project_root is not None else Path.cwd()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()

    if _is_dataset_bundle_dir(path):
        candidate_names = DATASET_BUNDLE_STAGE_MAP.get(stage, (stage,))
        for child_name in candidate_names:
            candidate = path / child_name
            if candidate.exists():
                return candidate
        candidate = path / candidate_names[0]
        if must_exist:
            raise FileNotFoundError(f"Dataset stage not found: {candidate}")
        return candidate

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path
