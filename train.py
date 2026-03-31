"""Raman 层级分类训练入口。"""

from raman.config import config
from raman.trainer import TrainOverrides, run_training

# 手动覆盖
# 适合在 Colab 里快速单独训练某个层级/父类
CURRENT_TRAIN_LEVEL = "level_1"  # 例如 "level_2"
TRAIN_ONLY_PARENT_NAME = None  # 例如 "dachang"
TRAIN_ONLY_PARENT = None  # 例如 2（可选，优先级高于名称）
OVERRIDE_DECAY_START_RATIO = None  # None 表示直接使用 config.decay_start_ratio

# 可选：覆盖损失参数（单独训练时可能不同）
OVERRIDE_ALIGN_LOSS_WEIGHT = None
OVERRIDE_SUPCON_TAU = None
OVERRIDE_SUPCON_LOSS_WEIGHT = None

# 可选：固定输出目录/时间戳，避免切换 config 导致输出分散
OVERRIDE_TIMESTAMP = None
OVERRIDE_OUTPUT_DIR = None


def main():
    overrides = TrainOverrides(
        current_train_level=CURRENT_TRAIN_LEVEL,
        train_only_parent_name=TRAIN_ONLY_PARENT_NAME,
        train_only_parent=TRAIN_ONLY_PARENT,
        override_decay_start_ratio=OVERRIDE_DECAY_START_RATIO,
        override_align_loss_weight=OVERRIDE_ALIGN_LOSS_WEIGHT,
        override_supcon_tau=OVERRIDE_SUPCON_TAU,
        override_supcon_loss_weight=OVERRIDE_SUPCON_LOSS_WEIGHT,
        override_timestamp=OVERRIDE_TIMESTAMP,
        override_output_dir=OVERRIDE_OUTPUT_DIR,
    )
    run_training(config, overrides=overrides)


if __name__ == "__main__":
    main()
