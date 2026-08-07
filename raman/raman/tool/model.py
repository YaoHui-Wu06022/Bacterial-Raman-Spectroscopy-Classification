"""模型输出解析工具"""

import torch
import torch.nn as nn


def select_logits(pred, head_name=None):
    """从模型输出中取出当前需要使用的 logits"""
    if isinstance(pred, dict):
        if head_name is None:
            head_name = list(pred.keys())[-1]
        return pred[head_name]
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred


def needs_cudnn_rnn_guard(model):
    """判断评估态 RNN 反传是否需要临时关闭 cuDNN"""
    if model.training:
        return False
    if not torch.backends.cudnn.enabled:
        return False
    for module in model.modules():
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            return True
    return False
