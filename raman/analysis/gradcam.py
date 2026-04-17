import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .ig import _ensure_dir, _needs_cudnn_rnn_guard, _select_logits

def collect_analyzable_layers(model):
    """
    自动收集：
    - conv1 / input_proj
    - 所有 ResidualBottleneck1D
    - 所有 TransformerEncoderLayer
    - LSTM
    返回:
        analyzable: { "layer1.0": module, ... }
        groups:     { "layer1.0": "layer1", "layer1.1": "layer1", ... }
    """
    from raman.model import ResidualBottleneck1D

    analyzable = {}
    groups = {}

    def recursive_find(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # ResNet 风格的瓶颈块，按所在 stage 归组
            if isinstance(child, ResidualBottleneck1D):
                analyzable[full_name] = child
                groups[full_name] = full_name.split(".")[0]  # layer1.0 -> layer1
                continue

            # Transformer 编码层统一归到 transformer 组
            if isinstance(child, nn.TransformerEncoderLayer):
                analyzable[full_name] = child
                groups[full_name] = "transformer"
                continue

            # 循环层统一归到 lstm 组
            if isinstance(child, nn.LSTM):
                analyzable[full_name] = child
                groups[full_name] = "lstm"
                continue

            # 输入投影层单独保留，方便看最前端特征提取的贡献
            if full_name in ("conv1", "input_proj"):
                analyzable[full_name] = child
                groups[full_name] = full_name
                continue

            # 其他模块继续递归向下查找
            recursive_find(child, full_name)

    recursive_find(model)
    return analyzable, groups

class LayerGradCAMAnalyzer:
    """
    通过 forward / backward hook 收集中间层的激活与梯度，
    再用 mean(|A * G|) 计算每一层的重要性。
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.activations = {}
        self.gradients = {}
        self.hooks = []

    # 注册前向与反向 hook，分别缓存激活和梯度
    def register_layer(self, name, layer):
        # 前向 hook：保存该层输出激活
        def f_hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.activations[name] = out.detach()

        # 反向 hook：保存该层输出对应的梯度
        def b_hook(module, grad_in, grad_out):
            g = grad_out[0]
            self.gradients[name] = g.detach()

        self.hooks.append(layer.register_forward_hook(f_hook))
        self.hooks.append(layer.register_full_backward_hook(b_hook))

    # 分析结束后释放所有 hook，避免重复注册或内存残留
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_importance(self):
        importance = OrderedDict()

        for name, A in self.activations.items():
            if name not in self.gradients:
                continue
            G = self.gradients[name]

            # 激活和梯度应当同形，才能逐元素组合计算重要性
            score = torch.mean(torch.abs(A * G))
            importance[name] = float(score.item())

        # 归一化后更方便在不同层之间直接比较
        total = sum(importance.values())
        for k in list(importance.keys()):
            importance[k] = importance[k] / (total + 1e-8)

        return importance

    def run(self, loader, save_dir=None, num_batches=3, head_name=None, head_index=None):
        """
        从前 num_batches 个 batch 计算平均 layer importance（更稳定）。
        """
        _ensure_dir(save_dir)

        self.model.eval()
        disable_cudnn = _needs_cudnn_rnn_guard(self.model)
        prev_cudnn = torch.backends.cudnn.enabled
        if disable_cudnn:
            torch.backends.cudnn.enabled = False

        merged = None
        used = 0

        it = iter(loader)
        for _ in range(num_batches):
            try:
                x, y, _ = next(it)
            except StopIteration:
                break
            used += 1
            x, y = x.to(self.device), y.to(self.device)

            if y.ndim == 2:
                if head_index is None:
                    head_index = y.size(1) - 1
                y = y[:, head_index]

            logits = _select_logits(self.model(x), head_name=head_name)

            target = y

            # 用目标类别的 logit 之和作为反传目标
            score = logits.gather(1, target.view(-1, 1)).sum()

            self.model.zero_grad(set_to_none=True)
            score.backward()

            scores = self.compute_importance()

            if merged is None:
                merged = OrderedDict(scores)
            else:
                for k, v in scores.items():
                    merged[k] = merged.get(k, 0.0) + v

            # 清空本轮缓存，避免不同批次的激活和梯度混在一起
            self.activations.clear()
            self.gradients.clear()

        if used == 0:
            self.clear_hooks()
            if disable_cudnn:
                torch.backends.cudnn.enabled = prev_cudnn
            raise RuntimeError("No batch available to run LayerGradCAMAnalyzer.")

        # 多个批次先取平均，再统一做一次归一化
        for k in list(merged.keys()):
            merged[k] /= float(used)
        total = sum(merged.values())
        for k in list(merged.keys()):
            merged[k] /= (total + 1e-8)

        if save_dir is not None:
            self.plot(merged, save_dir)

        self.clear_hooks()
        if disable_cudnn:
            torch.backends.cudnn.enabled = prev_cudnn
        return merged

    def plot(self, scores, save_dir):
        names = list(scores.keys())
        vals = list(scores.values())
        plt.figure(figsize=(10, 5))
        plt.bar(names, vals)
        plt.xticks(rotation=60)
        plt.ylabel("Layer Importance (Normalized |A × G|)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "layer_importance.png"), dpi=300)
        plt.close()

def merge_scores_by_group(layer_scores, groups):
    """
    将 layer_scores 合并到 stage 粒度（conv1 / layer1 / ... / transformer）
    """
    merged = {}
    for name, score in layer_scores.items():
        g = groups.get(name, name)
        merged[g] = merged.get(g, 0.0) + float(score)

    total = sum(merged.values())
    for k in list(merged.keys()):
        merged[k] /= (total + 1e-8)
    return merged


def _plot_layer_importance(scores, save_path):
    """聚合模式下的层级重要性柱状图。"""
    names = list(scores.keys())
    vals = list(scores.values())
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.xticks(rotation=60)
    plt.ylabel("Layer Importance (Aggregated |A x G|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
