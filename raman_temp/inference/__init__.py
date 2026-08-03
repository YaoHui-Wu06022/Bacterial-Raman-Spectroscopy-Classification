"""独立文件夹推理接口。"""

__all__ = [
    "Prediction",
    "Predictor",
    "load_predictor",
    "run_independent_inference",
]


def __getattr__(name: str):
    """按需暴露推理接口，避免命令帮助加载模型依赖。"""
    if name in {"Prediction", "Predictor", "load_predictor"}:
        from . import predictor

        return getattr(predictor, name)
    if name == "run_independent_inference":
        from .runner import run_independent_inference

        return run_independent_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
