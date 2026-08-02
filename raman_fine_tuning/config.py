"""跨数据集微调专属配置。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass


DEFAULT_TRANSFER_INPUT_GRID_MODE = "stanford_transfer"
DEFAULT_TRANSFER_NORM_METHOD = "minmax"


@dataclass
class FineTuneConfig:
    """描述来源模型、目标任务及其微调策略，不污染通用 raman 配置。"""

    source_run_dir: str
    source_level: str = "level_1"
    target_dataset_name: str = "GN"
    current_train_level: str = "level_1"
    train_only_parent_name: str | None = None
    train_only_parent: int | None = None
    output_dir: str | None = None
    unfreeze_tail: bool = True
    warm_start_run_dir: str | None = None
    learning_rate: float | None = 5e-5
    input_grid_mode: str = DEFAULT_TRANSFER_INPUT_GRID_MODE
    norm_method: str = DEFAULT_TRANSFER_NORM_METHOD

    def build_target_config(self, base_config):
        """基于通用配置生成一次性的目标微调配置，不修改调用方对象。"""
        target_config = deepcopy(base_config)
        target_config.dataset_name = self.target_dataset_name
        target_config.input_grid_mode = self.input_grid_mode
        target_config.norm_method = self.norm_method
        if self.learning_rate is not None:
            target_config.learning_rate = float(self.learning_rate)
        target_config.resume_training = False
        return target_config

    def to_dict(self):
        """用于 notebook 展示和实验记录。"""
        return asdict(self)
