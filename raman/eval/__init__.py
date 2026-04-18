"""评估与基线入口统一导出层"""

from .common import compute_classification_metrics

__all__ = [
    "BaselineOverrides",
    "EvaluateOverrides",
    "compute_classification_metrics",
    "run_evaluate_test_set",
    "run_pca_svm_baseline",
]


def __getattr__(name):
    if name in {"BaselineOverrides", "run_pca_svm_baseline"}:
        from .baseline import BaselineOverrides, run_pca_svm_baseline

        exports = {
            "BaselineOverrides": BaselineOverrides,
            "run_pca_svm_baseline": run_pca_svm_baseline,
        }
        return exports[name]

    if name in {"EvaluateOverrides", "run_evaluate_test_set"}:
        from .evaluator import EvaluateOverrides, run_evaluate_test_set

        exports = {
            "EvaluateOverrides": EvaluateOverrides,
            "run_evaluate_test_set": run_evaluate_test_set,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
