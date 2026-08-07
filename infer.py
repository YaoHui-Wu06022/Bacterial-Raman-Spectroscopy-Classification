"""在 PyCharm 或 Colab 中直接运行的独立测试集推理入口。"""

from dataclasses import replace

from ramanv2.data.build import build_test
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.inference.runner import run_independent_inference


# 本次推理任务范围；测试集重建沿用常规 data 服务。
SOURCE_DIR = "output/GN/20260807_013457_90%"
# 填写实验目录内的历史 run 时，推理仅加载该 run，不改写 hierarchy_meta.json。
MODEL_RUN_DIR = "output/GN/20260807_013457_90%/level_1/run_20260807_013458"
LEVEL_NAME = "level_1"
BUILD_TEST_ENABLE = False
PROFILE_ID = "test"
TEST_DIR = "dataset/CSdata/test"
ONE_DIR = None
TOP_K = 3
CPU_ENABLE = False
EVALUATE_ENABLE = True
PLOT_TRAIN_MEAN_ENABLE = False


def main():
    """按顶部任务范围重建测试集并执行独立推理。"""
    if not SOURCE_DIR:
        raise ValueError("请先在 infer.py 里填写 SOURCE_DIR")

    if BUILD_TEST_ENABLE:
        profile = replace(get_profile(PROFILE_ID), root_init_test="init")
        dataset_dir = get_dataset_dir(profile)
        build_test(profile, dataset_dir)

    run_independent_inference(
        SOURCE_DIR,
        LEVEL_NAME,
        model_run_dir=MODEL_RUN_DIR,
        input_dir=TEST_DIR,
        one_dir=ONE_DIR,
        top_k=TOP_K,
        device="cpu" if CPU_ENABLE else None,
        evaluate_enable=EVALUATE_ENABLE,
        plot_train_mean_enable=PLOT_TRAIN_MEAN_ENABLE,
    )


if __name__ == "__main__":
    main()
