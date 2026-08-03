"""基于数据索引构建 PyTorch 训练数据视图。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from raman_temp.core.input_spec import InputSpec
from raman_temp.data.augmentation import AugmentationSpec
from raman_temp.data.index import DatasetIndex
from raman_temp.data.input import build_model_input, build_sg_kernels


class RamanDataset(Dataset):
    """按需将 DatasetIndex 样本转换为模型输入张量。"""

    def __init__(
        self,
        dataset_index: DatasetIndex,
        input_spec: InputSpec,
        augmentation_spec: AugmentationSpec | None = None,
        augmentation_enable: bool = False,
    ) -> None:
        if augmentation_enable and augmentation_spec is None:
            raise ValueError("启用数据增强时必须提供 AugmentationSpec")
        self.dataset_index = dataset_index
        self.input_spec = input_spec
        self.augmentation_spec = augmentation_spec
        self.augmentation_enable = augmentation_enable
        self.smooth_kernel, self.d1_kernel = build_sg_kernels(input_spec, "cpu")

    def __len__(self) -> int:
        """返回索引中的样本数量。"""
        return len(self.dataset_index.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray, str]:
        """返回模型输入、多层数值标签和样本路径。"""
        model_input = build_model_input(
            self.dataset_index.get_raw_intensity(index),
            self.input_spec,
            self.smooth_kernel,
            self.d1_kernel,
            "cpu",
            self.augmentation_spec,
            self.augmentation_enable,
        )
        return (
            model_input,
            self.dataset_index.level_labels[index],
            str(self.dataset_index.samples[index]),
        )
