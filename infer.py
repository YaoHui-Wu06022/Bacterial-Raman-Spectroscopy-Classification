"""在 PyCharm 或 Colab 中直接运行的独立测试集推理入口。"""

from dataclasses import replace

from ramanv2.data.build import build_test
from ramanv2.data.profiles import get_dataset_dir, get_profile
from ramanv2.inference.runner import run_independent_inference


# 本次推理任务范围；测试集重建沿用常规 data 服务。
SOURCE_DIR = "output/GN/20260722_051934_div4_89%"
LEVEL_NAME = "level_2"
BUILD_TEST_ENABLE = True
PROFILE_ID = "test"
TEST_DIR = "dataset/测试菌/test"
FOLDER = None
TOP_K = 3
CPU_ENABLE = False
EVALUATE_ENABLE = True
PLOT_TRAIN_MEAN_ENABLE = False
SKIP_TRANSFERRED_ENABLE = True
TRANSFER_MANIFEST_PATH = "dataset/测试菌/test_transfer_manifest.csv"


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
        input_dir=TEST_DIR,
        folder=FOLDER,
        top_k=TOP_K,
        device="cpu" if CPU_ENABLE else None,
        evaluate_enable=EVALUATE_ENABLE,
        plot_train_mean_enable=PLOT_TRAIN_MEAN_ENABLE,
        skip_transferred_enable=SKIP_TRANSFERRED_ENABLE,
        transfer_manifest_path=TRANSFER_MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
