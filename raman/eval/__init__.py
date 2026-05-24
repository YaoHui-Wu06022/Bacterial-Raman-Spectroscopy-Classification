"""评估与基线入口统一导出层"""

from .common import compute_classification_metrics

__all__ = [
    "compute_classification_metrics",
    "run_baseline_cascade",
    "run_baseline_level_only",
    "run_baseline_single_model",
    "run_eval_cascade",
    "run_eval_level_only",
    "run_eval_single_model",
]


def __getattr__(name):
    if name in {
        "run_baseline_single_model",
        "run_baseline_level_only",
        "run_baseline_cascade",
    }:
        from .baseline import (
            run_baseline_cascade,
            run_baseline_level_only,
            run_baseline_single_model,
        )

        exports = {
            "run_baseline_single_model": run_baseline_single_model,
            "run_baseline_level_only": run_baseline_level_only,
            "run_baseline_cascade": run_baseline_cascade,
        }
        return exports[name]

    if name in {"run_eval_single_model", "run_eval_level_only", "run_eval_cascade"}:
        from .evaluator import (
            run_eval_cascade,
            run_eval_level_only,
            run_eval_single_model,
        )

        exports = {
            "run_eval_single_model": run_eval_single_model,
            "run_eval_level_only": run_eval_level_only,
            "run_eval_cascade": run_eval_cascade,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
