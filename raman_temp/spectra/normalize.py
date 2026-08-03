"""光谱归一化，支持 NumPy 与可选的 PyTorch 张量。"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


EPS = 1e-8
SUPPORTED_NORM_METHODS = ("none", "snv", "minmax", "l2")


def _normalize_method(method: str) -> str:
    """校验并规范化归一化方法名称。"""
    normalized = str(method).lower()
    if normalized not in SUPPORTED_NORM_METHODS:
        supported = ", ".join(SUPPORTED_NORM_METHODS)
        raise ValueError(f"未知的 norm_method: {normalized}，可选值: {supported}")
    return normalized


def _normalize_numpy(values, method: str, eps: float):
    """沿最后一维完成 NumPy 归一化。"""
    if method == "none":
        return values
    if method == "snv":
        mean = np.mean(values, axis=-1, keepdims=True)
        std = np.std(values, axis=-1, keepdims=True)
        return (values - mean) / (std + eps)
    if method == "minmax":
        minimum = np.min(values, axis=-1, keepdims=True)
        maximum = np.max(values, axis=-1, keepdims=True)
        return (values - minimum) / np.maximum(maximum - minimum, eps)
    if method == "l2":
        norm = np.sqrt(np.sum(values * values, axis=-1, keepdims=True))
        return values / np.maximum(norm, eps)
    raise AssertionError(method)


def _normalize_numpy_preserve_nan(values, method: str, eps: float):
    """仅对每行有限位置归一化，并保留原有 NaN。"""
    output = np.full(values.shape, np.nan, dtype=np.float32)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_output = output.reshape(-1, output.shape[-1])
    for index, row in enumerate(flat_values):
        finite = np.isfinite(row)
        if finite.any():
            flat_output[index, finite] = _normalize_numpy(
                row[finite].astype(np.float32, copy=False), method, eps
            ).astype(np.float32, copy=False)
    return output


def _normalize_tensor(values, method: str, eps: float):
    """沿最后一维完成 PyTorch 张量归一化。"""
    if method == "none":
        return values
    if method == "snv":
        return (values - values.mean(dim=-1, keepdim=True)) / (
            values.std(dim=-1, unbiased=False, keepdim=True) + eps
        )
    if method == "minmax":
        minimum = values.amin(dim=-1, keepdim=True)
        maximum = values.amax(dim=-1, keepdim=True)
        return (values - minimum) / (maximum - minimum).clamp_min(eps)
    if method == "l2":
        norm = torch.sqrt(torch.sum(values * values, dim=-1, keepdim=True))
        return values / norm.clamp_min(eps)
    raise AssertionError(method)


def _normalize_tensor_preserve_nan(values, method: str, eps: float):
    """在张量上按有限位置归一化，同时保留 NaN 掩码。"""
    output = torch.full_like(values, torch.nan)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_output = output.reshape(-1, output.shape[-1])
    for index, row in enumerate(flat_values):
        finite = torch.isfinite(row)
        if bool(finite.any()):
            flat_output[index, finite] = _normalize_tensor(row[finite], method, eps)
    return output


def normalize_spectrum(data, method: str, eps: float = EPS, preserve_nan_enable: bool = False):
    """按最后一维归一化单条或多条光谱，支持 ``none/snv/minmax/l2``。"""
    method = _normalize_method(method)
    if torch is not None and isinstance(data, torch.Tensor):
        values = data.to(dtype=torch.float32)
        if values.ndim == 0:
            raise ValueError("normalize_spectrum 需要至少一维输入")
        if preserve_nan_enable:
            return _normalize_tensor_preserve_nan(values, method, eps)
        return _normalize_tensor(values, method, eps)

    values = np.asarray(data, dtype=np.float32)
    if values.ndim == 0:
        raise ValueError("normalize_spectrum 需要至少一维输入")
    if preserve_nan_enable:
        return _normalize_numpy_preserve_nan(values, method, eps)
    return _normalize_numpy(values, method, eps).astype(np.float32, copy=False)
