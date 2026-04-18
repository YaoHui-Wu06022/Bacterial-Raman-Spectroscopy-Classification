import json
import os

import torch

from raman.config_io import load_experiment
from raman.data import InputPreprocessor
from raman.eval.common import run_cascade_inference, select_logits
from raman.eval.experiment import load_hierarchy_meta, resolve_project_path
from raman.eval.runtime import build_experiment_runtime


def load_predictor(exp_dir, device, predict_level=None):
    exp_dir = os.fspath(resolve_project_path(exp_dir))
    config = load_experiment(exp_dir)
    if not predict_level:
        raise ValueError("predict_level must be provided explicitly.")
    if not isinstance(predict_level, str) or not predict_level.startswith("level_"):
        raise ValueError(
            f"predict_level must be a business level like level_n, got: {predict_level}"
        )

    meta = load_hierarchy_meta(exp_dir)
    preprocessor = InputPreprocessor(config, device)

    if meta is None:
        class_path = os.path.join(exp_dir, "class_names.json")
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"[Predict] class_names.json not found in {exp_dir}.")

        with open(class_path, "r", encoding="utf-8") as file:
            class_names = json.load(file)

        if isinstance(class_names, dict):
            if predict_level not in class_names:
                raise ValueError(
                    f"Unknown predict_level: {predict_level}. Available: {list(class_names.keys())}"
                )
            class_names = class_names[predict_level]

        compat_meta = {
            "head_names": [predict_level],
            "level_models": {predict_level: config.model_path},
            "class_names_by_level": {predict_level: class_names},
            "parent_models": {},
            "parent_to_children": {},
        }
        runtime = build_experiment_runtime(exp_dir, device, config=config, meta=compat_meta)
        model = runtime.load_single_level_model(
            predict_level,
            num_classes=len(class_names),
        )

        return {
            "mode": "single",
            "model": model,
            "class_names": class_names,
            "device": device,
            "preprocessor": preprocessor,
            "config": config,
            "runtime": runtime,
        }

    head_names = meta.get("head_names", [])
    if predict_level not in head_names:
        raise ValueError(
            f"Unknown predict_level: {predict_level}. Available: {head_names}"
        )

    level_order = []
    for level_name in head_names:
        level_order.append(level_name)
        if level_name == predict_level:
            break

    runtime = build_experiment_runtime(exp_dir, device, config=config, meta=meta)
    runtime.build_level_model_paths(level_order)
    for level_name in level_order:
        runtime.ensure_parent_models(level_name)

    return {
        "mode": "cascade",
        "runtime": runtime,
        "level_order": level_order,
        "predict_level": predict_level,
        "device": device,
        "preprocessor": preprocessor,
        "config": config,
        "exp_dir": exp_dir,
    }


def predict_one(path, predictor, top_k=3, parent_mask=None):
    x = predictor["preprocessor"](path)

    if predictor.get("mode") == "single":
        model = predictor["model"]
        class_names = predictor["class_names"]
        with torch.no_grad():
            logits = select_logits(model(x))
            probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        idx = probs.argsort()[::-1][:top_k]
        return [{"label": class_names[i], "prob": float(probs[i])} for i in idx]

    runtime = predictor["runtime"]
    class_names_by_level = runtime.class_names_by_level
    level_order = predictor["level_order"]
    predict_level = predictor["predict_level"]

    num_classes_by_level = {
        level_name: len(class_names_by_level.get(level_name, []))
        for level_name in level_order
    }

    with torch.no_grad():
        result = run_cascade_inference(
            runtime,
            x,
            level_order=level_order,
            target_level=predict_level,
            num_classes_by_level=num_classes_by_level,
            class_names_by_level=class_names_by_level,
            parent_to_children=runtime.parent_to_children,
            allowed_names_by_level=parent_mask,
            fallback_to_previous=True,
        )

    if result is None:
        return [{"label": "unknown", "prob": 0.0}]

    probs = result["probs"].cpu().numpy().reshape(-1)
    class_names = result["class_names"]
    k = min(top_k, len(probs))
    idx = probs.argsort()[::-1][:k]
    return [{"label": class_names[i], "prob": float(probs[i])} for i in idx]
