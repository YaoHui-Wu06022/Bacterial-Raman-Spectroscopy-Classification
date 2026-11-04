"""基于数据索引构建 PyTorch 训练数据视图。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ramanv2.core.input_spec import InputSpec
from ramanv2.data.runtime.augmentation import AugmentationSpec
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.data.runtime.input import build_model_input, build_sg_kernels


class RamanDataset(Dataset):
    """按需将 DatasetIndex 样本转换为模型输入张量。"""

    def __init__(
        self,
        dataset_index: DatasetIndex,
        input_spec: InputSpec,
        augmentation_spec: AugmentationSpec | None = None,
        augmentation_enable: bool = False,
        cache_input_enable: bool = False,
    ) -> None:
        if augmentation_enable and augmentation_spec is None:
            raise ValueError("启用数据增强时必须提供 AugmentationSpec")
        if cache_input_enable and augmentation_enable:
            raise ValueError("缓存模型输入时不能同时启用数据增强")
        self.dataset_index = dataset_index
        self.input_spec = input_spec
        self.augmentation_spec = augmentation_spec
        self.augmentation_enable = augmentation_enable
        self.cache_input_enable = cache_input_enable
        self.smooth_kernel, self.d1_kernel = build_sg_kernels(input_spec, "cpu")
        # 验证集不增强，各 epoch 结果相同；进程内首次构建后复用，避免反复预处理。
        self._input_cache: list[torch.Tensor | None] = (
            [None] * len(dataset_index.samples) if cache_input_enable else []
        )

    def __len__(self) -> int:
        """返回索引中的样本数量。"""
        return len(self.dataset_index.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray, str]:
        """返回模型输入、多层数值标签和样本路径。"""
        model_input = self._input_cache[index] if self.cache_input_enable else None
        if model_input is None:
            model_input = build_model_input(
                self.dataset_index.get_raw_intensity(index),
                self.input_spec,
                self.smooth_kernel,
                self.d1_kernel,
                "cpu",
                self.augmentation_spec,
                self.augmentation_enable,
            )
            if self.cache_input_enable:
                self._input_cache[index] = model_input
        return (
            model_input,
            self.dataset_index.level_labels[index],
            str(self.dataset_index.samples[index]),
        )
