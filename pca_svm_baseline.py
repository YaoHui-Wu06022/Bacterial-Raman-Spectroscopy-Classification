"""PCA+SVM 验证集基线入口"""

from raman.eval import (
    run_baseline_cascade,
    run_baseline_level_only,
    run_baseline_single_model,
)


# 可选模式：single_model / level_only / cascade
BASELINE_MODE = "single_model"

# single_model 模式填具体 run_* 或 best/run_* 目录
RUN_DIR = ""

# level_only / cascade 模式填实验根目录
EXP_DIR = ""

TARGET_LEVEL = "level_1"
USE_ALL_CHANNELS = False
PCA_N_COMPONENTS = 5
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
RANDOM_STATE = 42


def _baseline_kwargs():
    return {
        "use_all_channels": USE_ALL_CHANNELS,
        "pca_n_components": PCA_N_COMPONENTS,
        "svm_c": SVM_C,
        "svm_kernel": SVM_KERNEL,
        "svm_gamma": SVM_GAMMA,
        "random_state": RANDOM_STATE,
    }


def main():
    if BASELINE_MODE == "single_model":
        if not RUN_DIR:
            raise ValueError("single_model 模式请先填写 RUN_DIR")
        result_dir = run_baseline_single_model(
            RUN_DIR,
            level=TARGET_LEVEL,
            **_baseline_kwargs(),
        )
    elif BASELINE_MODE == "level_only":
        if not EXP_DIR:
            raise ValueError("level_only 模式请先填写 EXP_DIR")
        result_dir = run_baseline_level_only(
            EXP_DIR,
            TARGET_LEVEL,
            **_baseline_kwargs(),
        )
    elif BASELINE_MODE == "cascade":
        if not EXP_DIR:
            raise ValueError("cascade 模式请先填写 EXP_DIR")
        result_dir = run_baseline_cascade(
            EXP_DIR,
            TARGET_LEVEL,
            **_baseline_kwargs(),
        )
    else:
        raise ValueError("BASELINE_MODE 只能是 single_model / level_only / cascade")

    print("baseline_val_result_dir =", result_dir)


if __name__ == "__main__":
    main()
