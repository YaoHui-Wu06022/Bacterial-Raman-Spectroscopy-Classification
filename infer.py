"""在 PyCharm 或 Colab 中直接运行的独立测试集推理入口。"""

from ramanv2.inference.runner import run_independent_inference


# 本次独立推理任务范围；输入目录固定使用 CSdata。
SOURCE_DIR = "output/ramanv2/GP/20260809_103407"
# 填写实验目录内的历史 run 时，推理仅加载该 run，不改写 hierarchy_meta.json。
MODEL_RUN_DIR = None
LEVEL_NAME = "level_2"
PROFILE_ID = "GN"
TEST_DIR = "dataset/CSdata"
ONE_DIR = None
TOP_K = 3
CPU_ENABLE = False
EVALUATE_ENABLE = True
PLOT_TRAIN_MEAN_ENABLE = False


def main():
    """按顶部任务范围直接使用 CSdata 执行独立推理。"""
    if not SOURCE_DIR:
        raise ValueError("请先在 infer.py 里填写 SOURCE_DIR")

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
