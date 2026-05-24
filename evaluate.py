"""验证集评估入口"""

from raman.eval import (
    run_eval_cascade,
    run_eval_level_only,
    run_eval_single_model,
)


# 可选模式：single_model / level_only / cascade
EVAL_MODE = "single_model"

# single_model 模式填具体 run_* 或 best/run_* 目录
RUN_DIR = ""

# level_only / cascade 模式填实验根目录
EXP_DIR = ""

TARGET_LEVEL = "level_1"


def main():
    if EVAL_MODE == "single_model":
        if not RUN_DIR:
            raise ValueError("single_model 模式请先填写 RUN_DIR")
        result_dir = run_eval_single_model(RUN_DIR, level=TARGET_LEVEL)
    elif EVAL_MODE == "level_only":
        if not EXP_DIR:
            raise ValueError("level_only 模式请先填写 EXP_DIR")
        result_dir = run_eval_level_only(EXP_DIR, TARGET_LEVEL)
    elif EVAL_MODE == "cascade":
        if not EXP_DIR:
            raise ValueError("cascade 模式请先填写 EXP_DIR")
        result_dir = run_eval_cascade(EXP_DIR, TARGET_LEVEL)
    else:
        raise ValueError("EVAL_MODE 只能是 single_model / level_only / cascade")

    print("val_result_dir =", result_dir)


if __name__ == "__main__":
    main()
