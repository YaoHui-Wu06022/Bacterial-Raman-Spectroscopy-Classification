__all__ = [
    "load_predictor",
    "normalize_level_name",
    "predict_one",
    "predict_tensor",
]


def __getattr__(name):
    """按需加载推理核心，避免 help 阶段提前依赖 torch"""
    if name not in __all__:
        raise AttributeError(f"module 'raman.infer' has no attribute {name!r}")
    from raman.infer.core import load_predictor, normalize_level_name, predict_one, predict_tensor

    values = {
        "load_predictor": load_predictor,
        "normalize_level_name": normalize_level_name,
        "predict_one": predict_one,
        "predict_tensor": predict_tensor,
    }
    globals().update(values)
    return values[name]
