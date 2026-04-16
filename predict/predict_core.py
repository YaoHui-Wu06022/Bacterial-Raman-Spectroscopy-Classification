# 预测核心（级联推理）

import os
import json
import torch
import torch.nn.functional as F

from raman.config_io import load_experiment
from raman.data import InputPreprocessor
from raman.model import RamanClassifier1D
from raman.training import mask_logits_by_parent

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path):
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def _load_hierarchy_meta(exp_dir):
    meta_path = os.path.join(exp_dir, "hierarchy_meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    parent_to_children_raw = meta.get("parent_to_children", {})
    parent_to_children = {}
    for level, mapping in parent_to_children_raw.items():
        parent_to_children[level] = {
            int(k): list(v) for k, v in mapping.items()
        }

    parent_models_raw = meta.get("parent_models", {})
    parent_models = {}
    for level, mapping in parent_models_raw.items():
        parent_models[level] = {}
        for k, v in mapping.items():
            entry = dict(v)
            entry["child_ids"] = [int(c) for c in entry.get("child_ids", [])]
            parent_models[level][int(k)] = entry

    meta["parent_to_children"] = parent_to_children
    meta["parent_models"] = parent_models
    meta["level_models"] = meta.get("level_models", {})
    return meta


def _infer_parent_models(exp_dir, head_names, parent_to_children, class_names_by_level):
    parent_models = {}
    files = set(os.listdir(exp_dir))
    for level in head_names:
        if level == head_names[0]:
            continue
        if level not in parent_to_children:
            continue
        parent_models[level] = {}
        for parent_idx, child_ids in parent_to_children[level].items():
            model_name = f"{level}_parent_{parent_idx}_model.pt"
            if model_name in files:
                entry = {
                    "model_path": model_name,
                    "child_ids": list(child_ids),
                }
                if level in class_names_by_level:
                    entry["child_names"] = [
                        class_names_by_level[level][cid] for cid in child_ids
                    ]
                parent_models[level][int(parent_idx)] = entry
            else:
                parent_models[level][int(parent_idx)] = {
                    "model_path": None,
                    "child_ids": list(child_ids),
                }
    return parent_models


# 加载模型与层级元数据

def load_predictor(exp_dir, device, predict_level=None):
    """
    构建预测器（级联推理）
    语义来源：
        - config.yaml           : 模型结构 & 预处理
        - {level}_model.pt      : 各层级模型
        - hierarchy_meta.json   : 层级映射与类别顺序
    """

    exp_dir = resolve_path(exp_dir)
    config = load_experiment(exp_dir)
    if not predict_level:
        raise ValueError("predict_level 必须显式提供，且只能是业务层级。")
    if not isinstance(predict_level, str) or not predict_level.startswith("level_"):
        raise ValueError(
            f"predict_level 只能是形如 level_n 的业务层级，当前为：{predict_level}"
        )

    meta = _load_hierarchy_meta(exp_dir)
    if meta is None:
        # 兼容旧模式：仅加载单层模型
        class_path = os.path.join(exp_dir, "class_names.json")
        if not os.path.exists(class_path):
            raise FileNotFoundError(
                f"[Predict] class_names.json not found in {exp_dir}."
            )

        with open(class_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)

        # 兼容新格式：class_names.json 可能是 {level: [names]}
        if isinstance(class_names, dict):
            level = str(predict_level)
            if level not in class_names:
                raise ValueError(
                    f"Unknown predict_level: {level}. Available: {list(class_names.keys())}"
                )
            class_names = class_names.get(level, [])

        num_classes = len(class_names)
        model = RamanClassifier1D(
            num_classes=num_classes,
            config=config
        ).to(device)

        model_path = config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[Predict] Model not found: {model_path}"
            )

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        preprocessor = InputPreprocessor(config, device)

        return {
            "mode": "single",
            "model": model,
            "class_names": class_names,
            "device": device,
            "preprocessor": preprocessor,
            "config": config,
        }

    head_names = meta.get("head_names", [])
    class_names_by_level = meta.get("class_names_by_level", {})
    parent_to_children = meta.get("parent_to_children", {})
    parent_models = meta.get("parent_models", {})
    level_models_meta = meta.get("level_models", {})

    if predict_level not in head_names:
        raise ValueError(
            f"Unknown predict_level: {predict_level}. Available: {head_names}"
        )

    level_order = []
    for name in head_names:
        level_order.append(name)
        if name == predict_level:
            break

    if not parent_models or all(not v for v in parent_models.values()):
        inferred = _infer_parent_models(
            exp_dir, head_names, parent_to_children, class_names_by_level
        )
        if inferred:
            parent_models = inferred

    level_model_paths = {}
    for level in level_order:
        model_name = level_models_meta.get(level, f"{level}_model.pt")
        level_model_paths[level] = os.path.join(exp_dir, model_name)

    preprocessor = InputPreprocessor(config, device)

    return {
        "mode": "cascade",
        "level_model_paths": level_model_paths,
        "level_model_cache": {},
        "parent_model_cache": {},
        "class_names_by_level": class_names_by_level,
        "parent_to_children": parent_to_children,
        "parent_models": parent_models,
        "level_order": level_order,
        "predict_level": predict_level,
        "device": device,
        "preprocessor": preprocessor,
        "config": config,
        "exp_dir": exp_dir,
    }


# 单样本预测（核心接口）

def _mask_logits_by_allowed(logits, allowed_indices):
    # 仅保留允许的类别索引
    if not allowed_indices:
        return logits, None
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, allowed_indices] = True
    masked = logits.masked_fill(~mask, float("-inf"))
    valid = mask.any(dim=1)
    if (~valid).any():
        masked[~valid] = 0.0
    return masked, valid


def _resolve_allowed_indices(class_names, allowed):
    # 允许传入名称或索引
    if not allowed:
        return []
    indices = []
    if isinstance(allowed, (list, tuple, set)):
        items = list(allowed)
    else:
        items = [allowed]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for item in items:
        if isinstance(item, int):
            indices.append(int(item))
        else:
            idx = name_to_idx.get(str(item))
            if idx is not None:
                indices.append(int(idx))
    return sorted(set(indices))


def _compute_probs(logits):
    """将单头分类 logits 转成概率。"""
    return F.softmax(logits, dim=1)


def predict_one(path, predictor, top_k=3, parent_mask=None):
    """
    对单个 .arc_data 文件进行预测
    返回 top-k 结果
    """
    x = predictor["preprocessor"](path)

    if predictor.get("mode") == "single":
        model = predictor["model"]
        class_names = predictor["class_names"]
        with torch.no_grad():
            logits = model(x)
            probs = _compute_probs(logits).cpu().numpy().reshape(-1)
        idx = probs.argsort()[::-1][:top_k]
        return [
            {
                "label": class_names[i],
                "prob": float(probs[i])
            }
            for i in idx
        ]

    level_model_paths = predictor["level_model_paths"]
    level_model_cache = predictor["level_model_cache"]
    parent_model_cache = predictor["parent_model_cache"]
    class_names_by_level = predictor["class_names_by_level"]
    parent_to_children = predictor["parent_to_children"]
    parent_models = predictor["parent_models"]
    level_order = predictor["level_order"]
    predict_level = predictor["predict_level"]
    device = predictor["device"]
    config = predictor["config"]
    exp_dir = predictor["exp_dir"]

    def get_level_model(level):
        if level in level_model_cache:
            return level_model_cache[level]

        level_classes = class_names_by_level.get(level, [])
        if not level_classes:
            raise ValueError(f"Missing class names for level '{level}'")

        model_path = level_model_paths.get(level)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[Predict] Model not found for level '{level}': {model_path}"
            )

        model = RamanClassifier1D(
            num_classes=len(level_classes),
            config=config
        ).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        level_model_cache[level] = model
        return model

    def get_parent_model(level, parent_idx, child_ids, model_path):
        key = (level, parent_idx)
        if key in parent_model_cache:
            return parent_model_cache[key]

        full_path = model_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(exp_dir, model_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"[Predict] Parent model not found: {full_path}"
            )

        model = RamanClassifier1D(
            num_classes=len(child_ids),
            config=config
        ).to(device)

        state = torch.load(full_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        parent_model_cache[key] = model
        return model

    def forward_level(model):
        logits = model(x)
        probs = _compute_probs(logits)
        return logits, probs

    parent_pred = None
    probs_at_target = None
    target_child_ids = None
    target_class_names = None
    last_probs = None
    last_class_names = None

    with torch.no_grad():
        for level in level_order:
            if parent_pred is None:
                logits, probs = forward_level(get_level_model(level))
                last_probs = probs
                last_class_names = class_names_by_level.get(level, [])

                if parent_mask and level in parent_mask:
                    allowed_global = _resolve_allowed_indices(
                        class_names_by_level.get(level, []),
                        parent_mask[level]
                    )
                    logits, valid = _mask_logits_by_allowed(logits, allowed_global)
                    probs = _compute_probs(logits)
                    if valid is not None and not valid.any():
                        if last_probs is not None and last_class_names:
                            probs_at_target = last_probs
                            target_class_names = last_class_names
                            break
                        return [{"label": "unknown", "prob": 0.0}]

                if level == predict_level:
                    probs_at_target = probs
                    target_class_names = class_names_by_level[level]
                    break

                parent_pred = probs.argmax(1)
                continue

            parent_idx = int(parent_pred.item())
            if level in parent_models and parent_models[level]:
                entry = parent_models[level].get(parent_idx)
                if entry is None:
                    if last_probs is not None and last_class_names:
                        probs_at_target = last_probs
                        target_class_names = last_class_names
                        break
                    return [{"label": "unknown", "prob": 0.0}]

                child_ids = entry.get("child_ids", [])
                model_path = entry.get("model_path")

                if model_path is None:
                    if len(child_ids) == 1:
                        pred_global = child_ids[0]
                        if level == predict_level:
                            return [{
                                "label": class_names_by_level[level][pred_global],
                                "prob": 1.0
                            }]
                        parent_pred = torch.tensor([pred_global], device=device)
                        continue
                    if last_probs is not None and last_class_names:
                        probs_at_target = last_probs
                        target_class_names = last_class_names
                        break
                    return [{"label": "unknown", "prob": 0.0}]

                logits, probs = forward_level(
                    get_parent_model(level, parent_idx, child_ids, model_path)
                )
                last_probs = probs
                last_class_names = [
                    class_names_by_level[level][cid] for cid in child_ids
                ]

                if parent_mask and level in parent_mask:
                    allowed_global = _resolve_allowed_indices(
                        class_names_by_level.get(level, []),
                        parent_mask[level]
                    )
                    allowed_local = [i for i, cid in enumerate(child_ids) if cid in allowed_global]
                    logits, valid = _mask_logits_by_allowed(logits, allowed_local)
                    probs = _compute_probs(logits)
                    if valid is not None and not valid.any():
                        if last_probs is not None and last_class_names:
                            probs_at_target = last_probs
                            target_class_names = last_class_names
                            break
                        return [{"label": "unknown", "prob": 0.0}]

                if level == predict_level:
                    probs_at_target = probs
                    target_child_ids = child_ids
                    target_class_names = [
                        class_names_by_level[level][cid] for cid in child_ids
                    ]
                    break

                pred_local = probs.argmax(1).item()
                pred_global = child_ids[pred_local]
                parent_pred = torch.tensor([pred_global], device=device)
                continue

            logits, probs = forward_level(get_level_model(level))
            last_probs = probs
            last_class_names = class_names_by_level.get(level, [])
            if parent_pred is not None and level in parent_to_children:
                logits, valid_parent = mask_logits_by_parent(
                        logits, parent_pred, parent_to_children[level]
                )
                probs = _compute_probs(logits)
                if not valid_parent.any():
                    if last_probs is not None and last_class_names:
                        probs_at_target = last_probs
                        target_class_names = last_class_names
                        break
                    return [{"label": "unknown", "prob": 0.0}]

            if parent_mask and level in parent_mask:
                allowed_global = _resolve_allowed_indices(
                    class_names_by_level.get(level, []),
                    parent_mask[level]
                )
                logits, valid = _mask_logits_by_allowed(logits, allowed_global)
                probs = _compute_probs(logits)
                if valid is not None and not valid.any():
                    if last_probs is not None and last_class_names:
                        probs_at_target = last_probs
                        target_class_names = last_class_names
                        break
                    return [{"label": "unknown", "prob": 0.0}]

            if level == predict_level:
                probs_at_target = probs
                target_class_names = class_names_by_level[level]
                break

            parent_pred = probs.argmax(1)

    if probs_at_target is None:
        if last_probs is not None and last_class_names:
            probs_at_target = last_probs
            target_class_names = last_class_names
        else:
            return [{"label": "unknown", "prob": 0.0}]

    probs = probs_at_target.cpu().numpy().reshape(-1)
    k = min(top_k, len(probs))
    idx = probs.argsort()[::-1][:k]

    if target_child_ids is not None:
        return [
            {
                "label": target_class_names[i],
                "prob": float(probs[i])
            }
            for i in idx
        ]

    return [
        {
            "label": target_class_names[i],
            "prob": float(probs[i])
        }
        for i in idx
    ]
