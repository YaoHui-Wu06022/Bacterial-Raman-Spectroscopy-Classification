"""在任意 Jupyter / Python 环境中直接运行的评估入口。"""

from pathlib import Path

from ramanv2.evaluation.baseline import (
    BaselineSpec,
    evaluate_baseline_parent_routed,
    evaluate_baseline_run,
)
from ramanv2.evaluation.model_eval import (
    evaluate_model_parent_routed,
    evaluate_model_run,
)
from ramanv2.inference.runner import evaluate_test_set


# TARGET 取 model、baseline 或 test_set；MODEL/BASELINE 下 MODE 取 run 或 parent_routed。
TARGET = "model"
MODE = "run"
# model/baseline 时填 run 目录；test_set 时填实验目录。
SOURCE_DIR = ""
LEVEL_NAME = "level_1"
CPU_ENABLE = False
# 仅 test_set 使用；留空时读取 dataset/<profile>/test。
TEST_DIR = ""

# PCA-SVM 仅在 TARGET = "baseline" 时使用。
ALL_CHANNELS_ENABLE = False
PCA_COMPONENTS = 0.95
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
RANDOM_STATE = 42


def main():
    """按目标类型和路由方式调用对应评估服务。"""
    if not SOURCE_DIR:
        raise ValueError("请先在 evaluate.py 里填写 SOURCE_DIR")

    if TARGET == "test_set":
        report = evaluate_test_set(
            SOURCE_DIR,
            "cpu" if CPU_ENABLE else None,
            TEST_DIR or None,
            level_name=LEVEL_NAME,
        )
        overall = report["overall"]
        if overall is not None:
            print(
                f"{overall['level_name']} / {overall['mode']}："
                f"Spectrum Accuracy = {overall['spectrum_accuracy'] * 100:.4f}%，"
                f"Folder Accuracy = {overall['folder_accuracy'] * 100:.4f}% "
                f"({overall['folder_correct']}/{overall['folder_count']})"
            )
        print(Path(report["summary_path"]).read_text(encoding="utf-8"))
        return

    if TARGET == "model" and MODE == "run":
        result_dir = evaluate_model_run(
            SOURCE_DIR,
            LEVEL_NAME,
            "cpu" if CPU_ENABLE else None,
        )
    elif TARGET == "model" and MODE == "parent_routed":
        result_dir = evaluate_model_parent_routed(
            SOURCE_DIR,
            LEVEL_NAME,
            "cpu" if CPU_ENABLE else None,
        )
    elif TARGET == "baseline" and MODE == "run":
        result_dir = evaluate_baseline_run(
            SOURCE_DIR,
            LEVEL_NAME,
            _build_baseline_spec(),
        )
    elif TARGET == "baseline" and MODE == "parent_routed":
        result_dir = evaluate_baseline_parent_routed(
            SOURCE_DIR,
            LEVEL_NAME,
            _build_baseline_spec(),
        )
    else:
        raise ValueError("TARGET 只能是 model / baseline / test_set，MODE 只能是 run / parent_routed")

    print("result_dir =", result_dir)


def _build_baseline_spec():
    """将脚本中的 PCA-SVM 参数转换为冻结的评估规格。"""
    return BaselineSpec(
        all_channels_enable=ALL_CHANNELS_ENABLE,
        pca_components=PCA_COMPONENTS,
        svm_c=SVM_C,
        svm_kernel=SVM_KERNEL,
        svm_gamma=SVM_GAMMA,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
