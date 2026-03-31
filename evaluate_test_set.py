"""测试集评估入口。"""

from raman.eval import EvaluateOverrides, run_evaluate_test_set

# 手动覆盖
EXP_DIR = "output/ding/20260318_055705"
EVAL_LEVEL = "level_1"
INHERIT_MISSING_LEVELS = True
EVAL_ONLY_LEVEL = None
EVAL_ONLY_PARENT = None


def main():
    overrides = EvaluateOverrides(
        exp_dir=EXP_DIR,
        eval_level=EVAL_LEVEL,
        inherit_missing_levels=INHERIT_MISSING_LEVELS,
        eval_only_level=EVAL_ONLY_LEVEL,
        eval_only_parent=EVAL_ONLY_PARENT,
    )
    run_evaluate_test_set(overrides)


if __name__ == "__main__":
    main()
