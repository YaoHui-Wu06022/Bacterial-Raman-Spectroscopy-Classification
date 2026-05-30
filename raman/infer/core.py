import torch

from raman.data import InputPreprocessor
from raman.eval.common import run_cascade_inference
from raman.eval.experiment import load_experiment_context_with_dataset
from raman.eval.runtime import build_experiment_runtime
from raman.tool.hierarchy import load_hierarchy_meta, normalize_level_name
from raman.tool.model import select_logits


def load_predictor(exp_dir, device, predict_level=None):
    """读取实验目录并构建单条光谱预测所需的运行时上下文"""
    input_context, config = load_experiment_context_with_dataset(
        exp_dir,
        dataset_stage=None,
        must_exist=False,
    )
    exp_dir = input_context.exp_dir
    if not predict_level:
        raise ValueError("predict_level must be provided explicitly.")
    predict_level = normalize_level_name(predict_level)

    meta = load_hierarchy_meta(exp_dir)
    if meta is None:
        raise FileNotFoundError(f"[Predict] hierarchy_meta.json not found in {exp_dir}.")

    preprocessor = InputPreprocessor(config, device)

    head_names = meta.get("head_names", [])
    if predict_level not in head_names:
        raise ValueError(
            f"Unknown predict_level: {predict_level}. Available: {head_names}"
        )

    runtime = build_experiment_runtime(
        exp_dir,
        device,
        config=config,
        meta=meta,
        run_selection=input_context.run_selection,
    )

    mode = "cascade"
    parent_idx = None
    child_ids = None
    selected_class_names = runtime.class_names_by_level.get(predict_level, [])
    if input_context.is_single_run:
        level_order = [predict_level]
        if input_context.input_parent_idx is not None:
            mode = "single_parent"
            parent_idx = int(input_context.input_parent_idx)
            runtime.ensure_parent_models(predict_level)
            entry = runtime.parent_models.get(predict_level, {}).get(parent_idx)
            if entry is None or entry.get("model_path") is None:
                raise FileNotFoundError(
                    f"No parent model for {predict_level}, parent={parent_idx}"
                )
            child_ids = [int(item) for item in entry.get("child_ids", [])]
            selected_class_names = [
                selected_class_names[child_id]
                for child_id in child_ids
            ]
        else:
            mode = "single_global"
            runtime.build_level_model_paths(level_order)
    else:
        level_order = []
        for level_name in head_names:
            level_order.append(level_name)
            if level_name == predict_level:
                break
        runtime.build_level_model_paths(level_order)

    for level_name in level_order:
        runtime.ensure_parent_models(level_name)

    return {
        "mode": mode,
        "runtime": runtime,
        "level_order": level_order,
        "predict_level": predict_level,
        "parent_idx": parent_idx,
        "child_ids": child_ids,
        "class_names": selected_class_names,
        "device": device,
        "preprocessor": preprocessor,
        "config": config,
        "exp_dir": exp_dir,
        "meta": meta,
        "input_context": input_context,
    }


def predict_tensor(x, predictor, top_k=3, parent_mask=None):
    """对已经构建好的模型输入执行级联预测"""
    runtime = predictor["runtime"]
    class_names_by_level = runtime.class_names_by_level
    level_order = predictor["level_order"]
    predict_level = predictor["predict_level"]

    if predictor.get("mode") == "single_parent":
        child_ids = [int(item) for item in predictor["child_ids"]]
        parent_idx = int(predictor["parent_idx"])
        entry = runtime.parent_models.get(predict_level, {}).get(parent_idx, {})
        with torch.no_grad():
            logits = runtime.get_parent_model(
                predict_level,
                parent_idx,
                child_ids=child_ids,
                model_path=entry.get("model_path"),
            )(x)
            probs = torch.softmax(select_logits(logits), dim=1)
        labels = predictor["class_names"]
        k = min(top_k, len(labels))
        idx = probs.cpu().numpy().reshape(-1).argsort()[::-1][:k]
        return [{"label": labels[i], "prob": float(probs[0, i].item())} for i in idx]

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


def predict_one(path, predictor, top_k=3, parent_mask=None):
    """对单条光谱执行级联预测并返回 top-k 类别概率"""
    x = predictor["preprocessor"](path)
    return predict_tensor(x, predictor, top_k=top_k, parent_mask=parent_mask)
