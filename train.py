"""在 PyCharm 或 Colab 中直接运行的层级训练入口。"""

from ramanv2.core.config import build_config
from ramanv2.training.workflow import TrainRequest, run_training


# 本次训练任务范围；模型与训练默认参数在 ramanv2/core/config.py 中维护。
PROFILE_ID = "GN"
LEVEL_NAME = "level_1"
PARENT_NAME = None
PARENT_INDEX = None
TRAIN_PER_PARENT_ENABLE = True
EXPERIMENT_DIR = None
RUN_NAME = None
RESUME_RUN_DIR = None


def main():
    """构建一次训练请求，并交由训练工作流完成编排。"""
    request = TrainRequest(
        config=build_config({"profile_id": PROFILE_ID}),
        level_name=LEVEL_NAME,
        only_parent=PARENT_INDEX,
        only_parent_name=PARENT_NAME,
        train_per_parent_enable=TRAIN_PER_PARENT_ENABLE,
        experiment_dir=EXPERIMENT_DIR,
        run_name=RUN_NAME,
        resume_run_dir=RESUME_RUN_DIR,
    )
    run_training(request)


if __name__ == "__main__":
    main()
