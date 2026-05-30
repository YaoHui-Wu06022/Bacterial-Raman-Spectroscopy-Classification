import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from raman.config_io import (
    MODEL_CONFIG_NAME,
    RESOLVED_CONFIG_NAME,
    find_experiment_root,
    load_experiment,
)
from raman.tool.dataset import dataset_bundle_root, resolve_dataset_stage
from raman.tool.path import (
    PROJECT_ROOT,
    exp_abspath,
    exp_relpath,
    normalize_relpath,
    resolve_project_path,
)
from raman.training.split import TRAIN_SPLIT_NAME, VAL_SPLIT_NAME, split_files_hash


RUN_SELECTION_ENV = "RAMAN_RUN_SELECTION"
RESULT_DIR_NAMES = {
    "val": "val_result",
    "analysis": "analysis_result",
    "baseline": "baseline_val_result",
    "infer": "test_result",
}
LEVEL_ONLY_RESULT_DIR = "level_only_result"
CASCADE_RESULT_DIR = "cascade_result"


@dataclass
class ExperimentInputContext:
    """描述用户传入的是实验根还是某个 run 目录"""

    input_path: str
    exp_dir: str
    input_run_dir: str | None = None
    input_slot_dir: str | None = None
    input_level: str | None = None
    input_parent_idx: int | None = None
    run_selection: dict[str, str] | None = None

    @property
    def is_single_run(self):
        return self.input_run_dir is not None


def _parse_slot_identity(slot_dir):
    """从模型槽位目录推断 level 和 parent"""
    name = Path(slot_dir).name
    if re.fullmatch(r"level_\d+", name):
        return name, None

    match = re.fullmatch(r"(level_\d+)_(\d+)", name)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def resolve_experiment_input(path):
    """统一解析实验根或 run 目录输入"""
    input_path = Path(resolve_project_path(path)).resolve()
    if input_path.name.startswith("run_"):
        run_dir = input_path
        slot_dir = run_dir.parent
        if slot_dir.name == "best":
            slot_dir = slot_dir.parent
        exp_dir = find_experiment_root(run_dir).resolve()
        level_name, parent_idx = _parse_slot_identity(slot_dir)
        slot_key = exp_relpath(exp_dir, slot_dir)
        selected = os.path.relpath(run_dir, slot_dir)
        return ExperimentInputContext(
            input_path=os.fspath(input_path),
            exp_dir=os.fspath(exp_dir),
            input_run_dir=os.fspath(run_dir),
            input_slot_dir=os.fspath(slot_dir),
            input_level=level_name,
            input_parent_idx=parent_idx,
            run_selection={normalize_relpath(slot_key).strip("/"): selected},
        )

    exp_dir = input_path
    return ExperimentInputContext(
        input_path=os.fspath(input_path),
        exp_dir=os.fspath(exp_dir),
        run_selection=None,
    )


def load_experiment_context_with_dataset(exp_dir, dataset_stage="train", must_exist=True):
    """加载实验配置，并按需要把 dataset_root 对齐到数据集阶段目录"""
    input_context = resolve_experiment_input(exp_dir)
    config_path = input_context.input_run_dir or input_context.exp_dir
    config = load_experiment(os.fspath(config_path))
    raw_dataset_root = resolve_project_path(config.dataset_root)
    if dataset_stage is None:
        dataset_root = dataset_bundle_root(raw_dataset_root)
    else:
        dataset_root = resolve_dataset_stage(
            os.fspath(raw_dataset_root),
            stage=dataset_stage,
            project_root=os.fspath(PROJECT_ROOT),
            must_exist=must_exist,
        )
    config.dataset_root = os.fspath(dataset_root)
    config.output_dir = input_context.exp_dir
    config.experiment_dir = input_context.exp_dir
    if input_context.input_run_dir:
        config.run_dir = input_context.input_run_dir
    return input_context, config


def _load_run_selection():
    """从环境变量读取临时 run 选择配置"""
    raw = os.environ.get(RUN_SELECTION_ENV, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{RUN_SELECTION_ENV} must be a JSON object") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{RUN_SELECTION_ENV} must be a JSON object")
    return {
        normalize_relpath(key).strip("/"): str(value)
        for key, value in data.items()
        if value is not None and str(value).strip()
    }


def _selection_keys_for_slot(slot_dir):
    """为同一个模型槽位生成多种可匹配的 RUN_SELECTION key"""
    slot_dir = Path(slot_dir)
    keys = [
        normalize_relpath(slot_dir.name).strip("/"),
        normalize_relpath(slot_dir).strip("/"),
    ]
    parts = slot_dir.parts
    if len(parts) >= 2:
        keys.append(normalize_relpath(Path(parts[-2]) / parts[-1]).strip("/"))
    try:
        keys.append(normalize_relpath(slot_dir.resolve()).strip("/"))
    except OSError:
        pass
    return keys


def _selected_run_dir(slot_dir, run_selection=None):
    """按 RUN_SELECTION 在指定模型槽位里选择 run"""
    selection = run_selection or _load_run_selection()
    if not selection:
        return None
    selection = {
            normalize_relpath(key).strip("/"): str(value)
        for key, value in selection.items()
        if value is not None and str(value).strip()
    }
    for key in _selection_keys_for_slot(slot_dir):
        selected = selection.get(key)
        if not selected:
            continue
        run_dir = Path(selected)
        if not run_dir.is_absolute():
            run_dir = Path(slot_dir) / selected
        if not (run_dir.is_dir() and run_dir.name.startswith("run_")):
            raise ValueError(f"{RUN_SELECTION_ENV} selected path is not a run_* dir: {run_dir}")
        return run_dir
    return None


def select_run_dir(slot_dir, run_selection=None):
    """
    在某个模型槽位目录中选择要使用的 run
    best/ 里有唯一 run 时优先使用，否则回退到外层最新 run
    """
    slot_dir = Path(slot_dir)
    selected_run = _selected_run_dir(slot_dir, run_selection=run_selection)
    if selected_run is not None:
        return selected_run, "selected"

    best_dir = slot_dir / "best"
    if best_dir.exists():
        best_runs = sorted(path for path in best_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
        if len(best_runs) > 1:
            raise ValueError(f"{best_dir} 内有多个 run_*，best 目录只允许保留一个")
        if len(best_runs) == 1:
            return best_runs[0], "best"

    if not slot_dir.exists():
        return None, None

    runs = sorted(path for path in slot_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if runs:
        return runs[-1], "latest"
    return None, None


def _find_model_file(run_dir, model_names):
    """在 run 目录中查找模型文件，优先使用明确文件名"""
    for name in model_names:
        path = Path(run_dir) / name
        if path.exists():
            return path
    candidates = sorted(Path(run_dir).glob("*_model.pt"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _run_entry(exp_dir, run_dir, model_path, source):
    """生成记录某个 run 模型、配置和 split 信息的条目"""
    run_dir = Path(run_dir)
    entry = {
        "run_dir": exp_relpath(exp_dir, run_dir),
        "config_path": exp_relpath(exp_dir, run_dir / MODEL_CONFIG_NAME),
        "resolved_config_path": exp_relpath(exp_dir, run_dir / RESOLVED_CONFIG_NAME),
        "model_path": exp_relpath(exp_dir, model_path),
        "train_split_path": TRAIN_SPLIT_NAME,
        "val_split_path": VAL_SPLIT_NAME,
        "split_hash": split_files_hash(exp_dir),
        "source": source,
    }
    return entry


def _entry_from_meta(exp_dir, entry):
    """把旧 meta 条目补齐成运行时可用的模型条目"""
    if entry is None:
        return None
    if isinstance(entry, dict):
        item = dict(entry)
    else:
        item = {"model_path": entry}
    if not item.get("split_hash"):
        item["split_hash"] = split_files_hash(exp_dir)
    return item


def resolve_level_model_entry(exp_dir, level_name, level_models_meta=None, run_selection=None):
    """解析某一层全局模型条目，优先使用 best/latest run"""
    exp_dir = Path(resolve_project_path(exp_dir))
    run_dir, source = select_run_dir(exp_dir / level_name, run_selection=run_selection)
    if run_dir is not None:
        model_path = _find_model_file(run_dir, [f"{level_name}_model.pt"])
        if model_path is None:
            raise FileNotFoundError(f"Run 中缺少模型文件：{run_dir}")
        return _run_entry(exp_dir, run_dir, model_path, source)

    meta_entry = _entry_from_meta(exp_dir, (level_models_meta or {}).get(level_name))
    if meta_entry is not None:
        return meta_entry

    expected_path = exp_dir / level_name / f"{level_name}_model.pt"
    return {"model_path": exp_relpath(exp_dir, expected_path)}


def resolve_level_model_path(exp_dir, level_name, level_models_meta, run_selection=None):
    """解析某一层全局模型文件绝对路径"""
    exp_dir = Path(resolve_project_path(exp_dir))
    entry = resolve_level_model_entry(
        exp_dir,
        level_name,
        level_models_meta,
        run_selection=run_selection,
    )
    return os.fspath(exp_abspath(exp_dir, entry.get("model_path")))


def resolve_model_sidecar_path(model_path, sidecar_suffix=".se_stats.pt"):
    """根据模型文件路径推导配套 sidecar 路径"""
    model_path = Path(model_path)
    if model_path.suffix == ".pt":
        return os.fspath(model_path.with_suffix(sidecar_suffix))
    return os.fspath(Path(f"{model_path}{sidecar_suffix}"))


def _parent_model_names(level_name, parent_idx):
    """列出 parent 子模型可能使用的文件名"""
    return [
        f"{level_name}_{int(parent_idx)}_model.pt",
        f"{level_name}_parent_{int(parent_idx)}_model.pt",
    ]


def scan_parent_model_files(exp_dir, level_name, parent_to_children, run_selection=None):
    """扫描某一层 parent 子模型，优先识别新 best/latest run 结构"""
    exp_dir = Path(resolve_project_path(exp_dir))
    level_dir = exp_dir / level_name
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

    for parent_idx in list(mapping.keys()):
        slot_dir = level_dir / f"{level_name}_{int(parent_idx)}"
        run_dir, source = select_run_dir(slot_dir, run_selection=run_selection)
        if run_dir is None:
            continue
        model_path = _find_model_file(run_dir, _parent_model_names(level_name, parent_idx))
        entry = mapping[parent_idx]
        entry.update(_run_entry(exp_dir, run_dir, model_path, source))
        if model_path is None and len(entry.get("child_ids", [])) > 1:
            raise FileNotFoundError(f"Run 中缺少 parent 模型文件：{run_dir}")

    pattern = re.compile(rf"^{re.escape(level_name)}_parent_(\d+)_model\.pt$")
    if not level_dir.exists():
        return mapping

    for name in os.listdir(level_dir):
        match = pattern.match(name)
        if not match:
            continue
        parent_idx = int(match.group(1))
        entry = mapping.get(parent_idx, {"child_ids": []})
        if entry.get("model_path") is None:
            entry["model_path"] = exp_relpath(exp_dir, level_dir / name)
        mapping[parent_idx] = entry

    return mapping


def resolve_run_dir(exp_dir, entry):
    """从模型条目中解析实际 run 目录"""
    run_dir = (entry or {}).get("run_dir")
    if not run_dir:
        return None
    path = exp_abspath(resolve_project_path(exp_dir), run_dir)
    return path if path is not None and path.exists() else None


def _used_run_item(exp_dir, run_dir):
    """把 run 目录转换成 used_runs.json 中的简洁记录"""
    run_dir = Path(run_dir)
    return {
        "run": run_dir.name,
        "path": exp_relpath(exp_dir, run_dir),
    }


def collect_used_runs(exp_dir, runtime, level_order=None, target_level=None, target_parent_idx=None):
    """收集当前 runtime 已解析到的 run 目录"""
    exp_dir = Path(resolve_project_path(exp_dir))
    used = {}
    level_names = list(level_order or [])
    if target_level and target_level not in level_names:
        level_names.append(target_level)

    for level_name in level_names:
        entry = runtime.level_model_entries.get(level_name)
        run_dir = resolve_run_dir(exp_dir, entry)
        if run_dir is not None:
            used[level_name] = _used_run_item(exp_dir, run_dir)

    for level_name in level_names:
        parent_entries = runtime.parent_models.get(level_name, {})
        for parent_idx, entry in sorted(parent_entries.items()):
            if target_parent_idx is not None and level_name == target_level and int(parent_idx) != int(target_parent_idx):
                continue
            run_dir = resolve_run_dir(exp_dir, entry)
            if run_dir is None:
                continue
            used[f"{level_name}_{int(parent_idx)}"] = _used_run_item(exp_dir, run_dir)
    return used


def resolve_single_result_run_dir(exp_dir, runtime, target_level, target_parent_idx=None):
    """解析单模型任务应该写入的 run 目录"""
    if runtime is None:
        return None

    if target_parent_idx is not None:
        entry = runtime.parent_models.get(target_level, {}).get(int(target_parent_idx))
        run_dir = resolve_run_dir(exp_dir, entry)
        if run_dir is not None:
            return run_dir

    entry = runtime.level_model_entries.get(target_level)
    return resolve_run_dir(exp_dir, entry)


def resolve_result_dir(
    exp_dir,
    kind,
    target_level,
    *,
    input_context=None,
    runtime=None,
    target_parent_idx=None,
    prefer_run_dir=False,
):
    """按新目录规则解析结果输出目录"""
    name = RESULT_DIR_NAMES[kind]
    if input_context is not None and input_context.input_run_dir:
        return Path(input_context.input_run_dir) / name
    if prefer_run_dir:
        run_dir = resolve_single_result_run_dir(
            exp_dir,
            runtime,
            target_level,
            target_parent_idx=target_parent_idx,
        )
        if run_dir is not None:
            return Path(run_dir) / name
    return Path(resolve_project_path(exp_dir)) / target_level / name


def resolve_mode_result_root(exp_dir, target_level, mode):
    """解析多模型模式的结果根目录"""
    if mode == "level_only":
        result_name = LEVEL_ONLY_RESULT_DIR
    elif mode == "cascade":
        result_name = CASCADE_RESULT_DIR
    else:
        raise ValueError(f"Unknown result mode: {mode}")
    return Path(resolve_project_path(exp_dir)) / target_level / result_name


def resolve_mode_result_dir(exp_dir, kind, target_level, mode):
    """解析多模型模式下某类结果目录"""
    return resolve_mode_result_root(exp_dir, target_level, mode) / RESULT_DIR_NAMES[kind]


def write_used_runs(
    result_dir,
    *,
    mode,
    target_level,
    target_parent=None,
    runs=None,
):
    """把本次实际使用的模型 run 写入结果目录"""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "target_level": target_level,
        "target_parent": target_parent,
        "runs": runs or {},
    }
    path = result_dir / "used_runs.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def validate_parent_split_hashes(exp_dir, level_name, parent_entries):
    """同级 parent 级联评估时，要求所有选中 run 使用实验根同一套切分"""
    root_hash = split_files_hash(exp_dir)
    hashes = {}
    for parent_idx, entry in (parent_entries or {}).items():
        if not isinstance(entry, dict):
            continue
        split_hash = entry.get("split_hash") or root_hash
        entry["split_hash"] = split_hash
        if split_hash:
            hashes[int(parent_idx)] = split_hash

    unique_hashes = set(hashes.values())
    if len(unique_hashes) > 1:
        detail = ", ".join(f"parent={idx}: {value[:12]}" for idx, value in sorted(hashes.items()))
        raise ValueError(f"{level_name} 同级 parent 的 split 不一致，不能级联评估：{detail}")


def resolve_split_dir(exp_dir):
    """返回实验根 split 目录；新结构只允许实验根保存 train/val split"""
    split_dir = Path(resolve_project_path(exp_dir))
    if (split_dir / TRAIN_SPLIT_NAME).exists() and (split_dir / VAL_SPLIT_NAME).exists():
        return os.fspath(split_dir)
    return None
