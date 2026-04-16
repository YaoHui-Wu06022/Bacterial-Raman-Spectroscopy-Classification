"""Raman 层级分类训练入口。"""

from raman.config import config
from raman.trainer import TrainOverrides, run_training

# 手动覆盖
# 适合在 Colab 里快速单独训练某个层级/父类
CURRENT_TRAIN_LEVEL = "level_1"
TRAIN_ONLY_PARENT_NAME = None
TRAIN_ONLY_PARENT = None
# 可选：覆盖损失参数（单独训练时可能不同）
OVERRIDE_ALIGN_LOSS_WEIGHT = None
OVERRIDE_SUPCON_TAU = None
OVERRIDE_SUPCON_LOSS_WEIGHT = None

# 可选：固定输出目录，避免切换 config 导致输出分散
OVERRIDE_OUTPUT_DIR = None


def main():
    overrides = TrainOverrides(
        current_train_level=CURRENT_TRAIN_LEVEL,
        train_only_parent_name=TRAIN_ONLY_PARENT_NAME,
        train_only_parent=TRAIN_ONLY_PARENT,
        override_align_loss_weight=OVERRIDE_ALIGN_LOSS_WEIGHT,
        override_supcon_tau=OVERRIDE_SUPCON_TAU,
        override_supcon_loss_weight=OVERRIDE_SUPCON_LOSS_WEIGHT,
        override_output_dir=OVERRIDE_OUTPUT_DIR,
    )
    run_training(config, overrides=overrides)


if __name__ == "__main__":
    main()
