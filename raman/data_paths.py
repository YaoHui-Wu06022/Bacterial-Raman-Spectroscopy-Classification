from pathlib import Path

LEGACY_DATASET_DIR_MAP = {
    "dataset_train_细菌": Path("dataset") / "细菌" / "dataset_train",
    "dataset_test_细菌": Path("dataset") / "细菌" / "dataset_test",
    "dataset_train_耐药菌": Path("dataset") / "耐药菌" / "dataset_train",
    "dataset_test_耐药菌": Path("dataset") / "耐药菌" / "dataset_test",
    "dataset_train_厌氧菌": Path("dataset") / "厌氧菌" / "dataset_train",
    "dataset_test_厌氧菌": Path("dataset") / "厌氧菌" / "dataset_test",
}

DATASET_BUNDLE_STAGE_MAP = {
    "train": ("dataset_train",),
    "test": ("dataset_test",),
    "raw": ("dataset_raw",),
    "init": ("dataset_init",),
    "predict_input": ("测试菌", "dataset_test"),
    "train_fig": ("dataset_train_fig",),
    "test_fig": ("dataset_test_fig",),
}


def _coerce_path(path_value):
    if isinstance(path_value, Path):
        return path_value
    return Path(path_value)


def _is_dataset_bundle_dir(path):
    if not path.is_dir():
        return False
    return any((path / child).exists() for children in DATASET_BUNDLE_STAGE_MAP.values() for child in children)


def resolve_legacy_dataset_path(path_value, project_root=None):
    path = _coerce_path(path_value)
    if project_root is not None:
        root = Path(project_root)
    elif path.is_absolute():
        root = path.parent
    else:
        root = Path.cwd()
    if path.is_absolute():
        name = path.name
    else:
        name = path.as_posix().rstrip("/")
    mapped = LEGACY_DATASET_DIR_MAP.get(name)
    if mapped is None:
        return path
    return (root / mapped).resolve()


def resolve_dataset_stage(path_value, stage="train", project_root=None, must_exist=False):
    path = resolve_legacy_dataset_path(path_value, project_root=project_root)
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
