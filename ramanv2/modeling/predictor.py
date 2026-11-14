"""层级模型加载、缓存与单条张量预测。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from ramanv2.core.run_snapshot import RunSnapshot, load_run_snapshot
from ramanv2.core.config import InputConfig
from ramanv2.core.run_context import HIERARCHY_META_FILE_NAME, MODEL_FILE_NAME
from ramanv2.common.hierarchy import load_hierarchy_meta, normalize_level_name, resolve_allowed_indices
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.common.paths import find_ancestor_dir
from ramanv2.modeling.factory import build_model, validate_model_input
from ramanv2.modeling.model_spec import build_model_spec
from ramanv2.modeling.run_manifest import has_model_path, resolve_entry_path


@dataclass(frozen=True)
class Prediction:
    """单层预测的类别名称、概率和全局类别索引。"""

    label: str
    probability: float
    class_id: int


@dataclass
class Predictor:
    """持有实验元数据、输入规格和按需加载的分类模型。"""

    experiment_dir: Path
    run_dir: Path | None
    meta: dict[str, Any]
    profile_id: str
    dataset_id: str
    input_config: InputConfig
    input_spec: InputSpec
    device: torch.device
    predict_level: str
    level_order: tuple[str, ...]
    _models: dict[Path, torch.nn.Module] = field(default_factory=dict)
    _snapshots: dict[Path, RunSnapshot] = field(default_factory=dict)

    @property
    def class_names_by_level(self) -> dict[str, list[str]]:
        """返回各分类层的完整类别名称。"""
        return {
            str(level_name): list(names)
            for level_name, names in (self.meta.get("class_names_by_level") or {}).items()
        }

    def resolve_target_class_names(self) -> list[str]:
        """返回当前预测入口实际可输出的类别名称。"""
        if self.run_dir is None:
            return self.class_names_by_level[self.predict_level]
        entry, parent_id = _find_run_entry(
            self.meta,
            self.experiment_dir,
            self.run_dir,
        )
        _class_ids, names = _resolve_entry_classes(
            self.meta,
            self.predict_level,
            parent_id,
            entry,
        )
        return names

    def predict_tensor(
        self,
        values: torch.Tensor,
        top_k: int = 3,
        allowed_values: Mapping[str, list[int | str]] | None = None,
    ) -> list[Prediction]:
        """对单条输入张量执行层级级联或指定 run 预测。"""
        if self.run_dir is not None:
            entry, parent_id = _find_run_entry(
                self.meta,
                self.experiment_dir,
                self.run_dir,
            )
            return self._predict_entry(
                values,
                self.predict_level,
                entry,
                parent_id,
                top_k,
                (allowed_values or {}).get(self.predict_level),
            )

        parent_id = None
        for level_name in self.level_order:
            direct_class_id = _resolve_single_child_class_id(
                self.meta,
                level_name,
                parent_id,
            )
            if direct_class_id is None:
                entry, model_parent_id = _resolve_level_entry(
                    self.meta,
                    level_name,
                    parent_id,
                )
                predictions = self._predict_entry(
                    values,
                    level_name,
                    entry,
                    model_parent_id,
                    top_k if level_name == self.predict_level else 1,
                    (allowed_values or {}).get(level_name),
                )
            else:
                class_names = self.class_names_by_level[level_name]
                predictions = [
                    Prediction(
                        label=class_names[direct_class_id],
                        probability=1.0,
                        class_id=direct_class_id,
                    )
                ]
            if level_name == self.predict_level:
                return predictions
            parent_id = predictions[0].class_id
        raise RuntimeError(f"未找到目标层级：{self.predict_level}")

    def load_model(self, level_name: str, entry: Mapping[str, Any], parent_id: int | None):
        """按元数据条目严格加载模型，并缓存已加载权重。"""
        model_path = resolve_entry_path(self.experiment_dir, entry, "model_path")
        if model_path in self._models:
            return self._models[model_path]
        run_path = resolve_entry_path(self.experiment_dir, entry, "run_dir")
        snapshot = self._snapshots.get(run_path)
        if snapshot is None:
            snapshot = load_run_snapshot(run_path, self.experiment_dir)
            self._snapshots[run_path] = snapshot
        model_spec = build_model_spec(snapshot.config.model, self.input_spec)
        validate_model_input(model_spec, self.input_spec)
        class_count = _resolve_class_count(self.meta, level_name, parent_id, entry)
        model = build_model(class_count, model_spec).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        self._models[model_path] = model
        return model

    def _predict_entry(
        self,
        values: torch.Tensor,
        level_name: str,
        entry: Mapping[str, Any],
        parent_id: int | None,
        top_k: int,
        allowed_values: list[int | str] | None,
    ) -> list[Prediction]:
        """执行一个全局层模型或父类子模型，并转换为全局类别标识。"""
        model = self.load_model(level_name, entry, parent_id)
        with torch.no_grad():
            probabilities = torch.softmax(model(values.to(self.device)), dim=1)[0]
        class_ids, class_names = _resolve_entry_classes(
            self.meta,
            level_name,
            parent_id,
            entry,
        )
        selected = resolve_allowed_indices(class_names, allowed_values)
        if selected:
            mask = torch.full_like(probabilities, float("-inf"))
            mask[selected] = probabilities[selected].log()
            probabilities = torch.softmax(mask, dim=0)
        count = min(max(int(top_k), 1), len(class_names))
        indices = torch.argsort(probabilities, descending=True)[:count].tolist()
        return [
            Prediction(
                label=class_names[index],
                probability=float(probabilities[index].item()),
                class_id=class_ids[index],
            )
            for index in indices
        ]

    def predict_batch_class_ids(self, values: torch.Tensor) -> list[int]:
        """单 run 路径下对一批 ``[B, C, L]`` 输入返回逐行 top-1 全局类别 id。"""
        if self.run_dir is None:
            raise ValueError("批量预测仅支持指定 run_dir 的单模型路径")
        entry, parent_id = _find_run_entry(self.meta, self.experiment_dir, self.run_dir)
        model = self.load_model(self.predict_level, entry, parent_id)
        with torch.no_grad():
            probabilities = torch.softmax(model(values.to(self.device)), dim=1)
        class_ids, _names = _resolve_entry_classes(
            self.meta, self.predict_level, parent_id, entry
        )
        return [int(class_ids[index]) for index in probabilities.argmax(dim=1).tolist()]

def load_predictor(
    source_dir: Path | str,
    device: torch.device | str,
    predict_level: int | str,
) -> Predictor:
    """从实验目录或模型槽位构建推理预测器。"""
    source_path = Path(source_dir).resolve()
    experiment_dir = find_ancestor_dir(source_path, HIERARCHY_META_FILE_NAME)
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    if meta is None:
        raise FileNotFoundError(f"缺少 hierarchy_meta.json：{experiment_dir}")
    level_name = normalize_level_name(predict_level)
    head_names = tuple(meta.get("head_names") or [])
    if level_name not in head_names:
        raise ValueError(f"未知预测层级：{level_name}；可选值：{list(head_names)}")
    run_dir = _resolve_source_run_dir(source_path, experiment_dir)
    entry = _resolve_input_entry(meta, experiment_dir, run_dir, level_name)
    snapshot = load_run_snapshot(
        resolve_entry_path(experiment_dir, entry, "run_dir"),
        experiment_dir,
    )
    input_spec = build_input_spec(snapshot.config.input)
    if run_dir is None:
        global_entry = (meta.get("level_models") or {}).get(level_name)
        # 目标层有全局模型时可直接预测，不需要先走上级级联。
        level_order = (
            (level_name,)
            if has_model_path(global_entry)
            else head_names[: head_names.index(level_name) + 1]
        )
    else:
        level_order = (level_name,)
    return Predictor(
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        meta=meta,
        profile_id=snapshot.config.dataset.profile_id,
        dataset_id=snapshot.config.dataset.dataset_id,
        input_config=snapshot.config.input,
        input_spec=input_spec,
        device=torch.device(device),
        predict_level=level_name,
        level_order=tuple(level_order),
    )


def _resolve_input_entry(
    meta: Mapping[str, Any],
    experiment_dir: Path,
    run_dir: Path | None,
    level_name: str,
) -> Mapping[str, Any]:
    """选择用于确定输入规格的模型条目。"""
    if run_dir is not None:
        entry, _ = _find_run_entry(meta, experiment_dir, run_dir)
        return entry
    for current_level in meta.get("head_names") or []:
        global_entry = (meta.get("level_models") or {}).get(current_level)
        if global_entry and has_model_path(global_entry):
            return global_entry
        parent_entries = (meta.get("parent_models") or {}).get(current_level) or {}
        for parent_id in sorted(parent_entries, key=int):
            entry = parent_entries[parent_id]
            if has_model_path(entry):
                return entry
    raise RuntimeError(f"未找到可读取输入规格的模型：{level_name}")


def _resolve_level_entry(
    meta: Mapping[str, Any],
    level_name: str,
    parent_id: int | None,
) -> tuple[Mapping[str, Any], int | None]:
    """按上一级预测结果选择全局模型或对应父类子模型。"""
    global_entry = (meta.get("level_models") or {}).get(level_name)
    if global_entry and has_model_path(global_entry):
        return global_entry, None
    if parent_id is None:
        raise FileNotFoundError(f"{level_name} 缺少全局模型，无法开始层级预测")
    entry = ((meta.get("parent_models") or {}).get(level_name) or {}).get(str(parent_id))
    if not has_model_path(entry):
        raise FileNotFoundError(f"{level_name} 缺少 parent={parent_id} 的子模型")
    return entry, int(parent_id)


def _resolve_single_child_class_id(
    meta: Mapping[str, Any],
    level_name: str,
    parent_id: int | None,
) -> int | None:
    """解析无模型的唯一类别/唯一子类直通结果。"""
    global_entry = (meta.get("level_models") or {}).get(level_name)
    if global_entry and has_model_path(global_entry):
        return None
    if parent_id is None:
        class_names = (meta.get("class_names_by_level") or {}).get(level_name) or []
        return 0 if len(class_names) == 1 else None
    entry = (
        ((meta.get("parent_models") or {}).get(level_name) or {}).get(str(parent_id))
    )
    if entry and has_model_path(entry):
        return None
    child_ids = [
        int(item)
        for item in (
            ((meta.get("parent_to_children") or {}).get(level_name) or {})
            .get(str(parent_id))
            or []
        )
    ]
    return child_ids[0] if len(child_ids) == 1 else None


def _resolve_source_run_dir(
    source_path: Path,
    experiment_dir: Path,
) -> Path | None:
    """解析实验根或模型槽位目录，拒绝无法识别的目录。"""
    if source_path == experiment_dir:
        return None
    if (source_path / MODEL_FILE_NAME).is_file():
        return source_path
    raise ValueError(f"不是有效的模型槽位目录：{source_path}")


def _find_run_entry(
    meta: Mapping[str, Any],
    experiment_dir: Path,
    run_dir: Path,
) -> tuple[Mapping[str, Any], int | None]:
    """在元数据中定位模型槽位条目。"""
    target = run_dir.resolve()
    for _level_name, entry in (meta.get("level_models") or {}).items():
        if resolve_entry_path(experiment_dir, entry, "run_dir") == target:
            return entry, None
    for _level_name, entries in (meta.get("parent_models") or {}).items():
        for parent_text, entry in entries.items():
            if resolve_entry_path(experiment_dir, entry, "run_dir") == target:
                return entry, int(parent_text)
    raise FileNotFoundError(f"{HIERARCHY_META_FILE_NAME} 未记录模型槽位：{run_dir}")


def _resolve_class_count(
    meta: Mapping[str, Any],
    level_name: str,
    parent_id: int | None,
    entry: Mapping[str, Any],
) -> int:
    """计算模型分类头输出数量。"""
    if parent_id is not None:
        return len(entry.get("child_ids") or [])
    return len((meta.get("class_names_by_level") or {}).get(level_name) or [])


def _resolve_entry_classes(
    meta: Mapping[str, Any],
    level_name: str,
    parent_id: int | None,
    entry: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    """返回当前模型输出位置对应的全局类别标识和名称。"""
    names = list((meta.get("class_names_by_level") or {}).get(level_name) or [])
    if parent_id is None:
        return list(range(len(names))), names
    child_ids = [int(item) for item in (entry.get("child_ids") or [])]
    return child_ids, [names[index] for index in child_ids]


