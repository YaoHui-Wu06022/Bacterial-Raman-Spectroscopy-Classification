"""训练集专用的随机光谱增强。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from ramanv2.core.config import TrainingConfig


# 分段增益：每段乘性系数以 1 为中心的标准差。
PIECEWISE_GAIN_STD = 0.12

# 高斯噪声：基础噪声相对谱振幅的范围，以及随局部强度增长的斜率范围。
NOISE_BASE_REL_MIN = 0.005
NOISE_BASE_REL_MAX = 0.02
NOISE_SLOPE_REL_MIN = 0.0
NOISE_SLOPE_REL_MAX = 0.015

# 轴扰动：线性扭曲系数范围与正弦扭曲最大点位偏移。
AXIS_WARP_ALPHA = 0.002
AXIS_WARP_BETA = 1.0

# 弱基线：线性项、正弦项相对振幅范围及正弦频率范围。
BASELINE_LIN_MIN = 0.0
BASELINE_LIN_MAX = 0.02
BASELINE_SIN_MIN = 0.0
BASELINE_SIN_MAX = 0.01
BASELINE_FREQ_MIN = 0.5
BASELINE_FREQ_MAX = 2.0

# 强基线：插值节点振幅相对范围和随机节点数量范围。
BASELINE_STRONG_AMP_MIN = 0.05
BASELINE_STRONG_AMP_MAX = 0.15
BASELINE_STRONG_KNOTS_MIN = 5
BASELINE_STRONG_KNOTS_MAX = 9

# 归一化后形状扰动：峰位最大平移点数、展宽核范围和局部遮罩参数。
SHIFT_MAX = 3
BROAD_SIGMA_MIN = 0.6
BROAD_SIGMA_MAX = 1.2
BROAD_TRUNCATE = 3.0
MASK_WIDTH_MIN = 20
MASK_WIDTH_MAX = 50
MASK_ATTEN_MIN = 0.5
MASK_ATTEN_MAX = 0.8


@dataclass(frozen=True)
class AugmentationSpec:
    """控制训练样本随机增强的概率与每段最大叠加次数。"""

    p_piecewise_gain: float
    p_noise: float
    p_axis: float
    p_baseline_weak: float
    p_baseline_strong: float
    p_shift: float
    p_broadening: float
    p_cut: float
    max_pre_augs: int
    max_post_augs: int


def build_augmentation_spec(training_config: TrainingConfig) -> AugmentationSpec:
    """从训练配置提取随机增强规格。"""
    return AugmentationSpec(
        p_piecewise_gain=float(training_config.p_piecewise_gain),
        p_noise=float(training_config.p_noise),
        p_axis=float(training_config.p_axis),
        p_baseline_weak=float(training_config.p_baseline_weak),
        p_baseline_strong=float(training_config.p_baseline_strong),
        p_shift=float(training_config.p_shift),
        p_broadening=float(training_config.p_broadening),
        p_cut=float(training_config.p_cut),
        max_pre_augs=int(training_config.max_pre_augs),
        max_post_augs=int(training_config.max_post_augs),
    )


def augment_raw_spectrum(values: np.ndarray, augmentation_spec: AugmentationSpec) -> np.ndarray:
    """在归一化前随机组合强度、噪声、轴与基线扰动。"""
    augmented = np.asarray(values, dtype=np.float32)
    operations = []
    if np.random.rand() < augmentation_spec.p_piecewise_gain:
        segments = _build_random_segments(len(augmented), 60, 180)
        operations.append(lambda items: augment_piecewise_gain(items, segments, PIECEWISE_GAIN_STD))
    if np.random.rand() < augmentation_spec.p_noise:
        operations.append(augment_gaussian_noise)
    if np.random.rand() < augmentation_spec.p_axis:
        operations.append(augment_axis_warp)
    baseline_random = np.random.rand()
    if baseline_random < augmentation_spec.p_baseline_weak:
        operations.append(augment_weak_baseline)
    elif baseline_random < augmentation_spec.p_baseline_weak + augmentation_spec.p_baseline_strong:
        operations.append(augment_strong_baseline)

    np.random.shuffle(operations)
    for operation in operations[: augmentation_spec.max_pre_augs]:
        augmented = operation(augmented)
    return augmented.astype(np.float32, copy=False)


def augment_normalized_spectrum(
    values: np.ndarray,
    augmentation_spec: AugmentationSpec,
) -> np.ndarray:
    """在归一化后随机组合轻微形状扰动。"""
    augmented = np.asarray(values, dtype=np.float32)
    operations = []
    if np.random.rand() < augmentation_spec.p_shift:
        operations.append(augment_shift)
    if np.random.rand() < augmentation_spec.p_broadening:
        operations.append(augment_broadening)
    if np.random.rand() < augmentation_spec.p_cut:
        operations.append(augment_mask_attenuation)
    np.random.shuffle(operations)
    for operation in operations[: augmentation_spec.max_post_augs]:
        augmented = operation(augmented)
    return augmented.astype(np.float32, copy=False)


def _build_random_segments(length: int, minimum: int, maximum: int) -> list[tuple[int, int]]:
    """按随机长度覆盖整条谱，生成不重叠增益片段。"""
    segments = []
    left = 0
    while left < length:
        right = min(length, left + np.random.randint(minimum, maximum + 1))
        segments.append((left, right))
        left = right
    return segments


def _robust_amplitude(values: np.ndarray) -> float:
    """使用 1% 与 99% 分位差估计对异常峰稳定的振幅。"""
    lower, upper = np.percentile(np.asarray(values, dtype=np.float32), [1, 99])
    return max(float(upper - lower), 1e-6)


def augment_piecewise_gain(
    values: np.ndarray,
    segments: list[tuple[int, int]],
    gain_std: float,
) -> np.ndarray:
    """为每个连续片段施加独立随机增益。"""
    augmented = np.asarray(values, dtype=np.float32).copy()
    for left, right in segments:
        augmented[left:right] *= np.random.normal(1.0, gain_std)
    return augmented


def augment_gaussian_noise(values: np.ndarray) -> np.ndarray:
    """加入与谱振幅和当前位置强度相关的高斯噪声。"""
    values = np.asarray(values, dtype=np.float32)
    amplitude = _robust_amplitude(values)
    base = np.random.uniform(NOISE_BASE_REL_MIN, NOISE_BASE_REL_MAX) * amplitude
    slope = np.random.uniform(NOISE_SLOPE_REL_MIN, NOISE_SLOPE_REL_MAX)
    sigma = base + slope * np.abs(values)
    return values + np.random.randn(*values.shape).astype(np.float32) * sigma.astype(np.float32)


def augment_weak_baseline(values: np.ndarray) -> np.ndarray:
    """加入线性和正弦构成的弱低频背景扰动。"""
    values = np.asarray(values, dtype=np.float32)
    amplitude = _robust_amplitude(values)
    positions = np.linspace(-0.5, 0.5, len(values), dtype=np.float32)
    linear = np.random.uniform(BASELINE_LIN_MIN, BASELINE_LIN_MAX) * amplitude
    sinusoid = np.random.uniform(BASELINE_SIN_MIN, BASELINE_SIN_MAX) * amplitude
    frequency = np.random.uniform(BASELINE_FREQ_MIN, BASELINE_FREQ_MAX)
    phase = np.random.uniform(0, 2 * np.pi)
    baseline = linear * positions + sinusoid * np.sin(2 * np.pi * frequency * positions + phase)
    return values + baseline.astype(np.float32)


def augment_strong_baseline(values: np.ndarray) -> np.ndarray:
    """使用单调分段三次插值生成更明显的平滑低频背景。"""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2:
        return values.copy()
    knot_minimum = max(2, min(BASELINE_STRONG_KNOTS_MIN, len(values)))
    knot_maximum = max(knot_minimum, min(BASELINE_STRONG_KNOTS_MAX, len(values)))
    knot_count = np.random.randint(knot_minimum, knot_maximum + 1)
    knots = np.linspace(0, len(values) - 1, knot_count, dtype=np.float32)
    knot_values = np.random.uniform(-1.0, 1.0, knot_count).astype(np.float32)
    knot_values *= (
        np.random.uniform(BASELINE_STRONG_AMP_MIN, BASELINE_STRONG_AMP_MAX)
        * _robust_amplitude(values)
    )
    positions = np.arange(len(values), dtype=np.float32)
    baseline = PchipInterpolator(knots, knot_values)(positions).astype(np.float32)
    return values + baseline


def augment_axis_warp(values: np.ndarray) -> np.ndarray:
    """轻微扭曲采样位置，模拟坐标标定偏差。"""
    values = np.asarray(values, dtype=np.float32)
    length = len(values)
    positions = np.arange(length, dtype=np.float32)
    centered = positions - (length - 1) / 2.0
    linear = np.random.uniform(-AXIS_WARP_ALPHA, AXIS_WARP_ALPHA) * centered
    nonlinear = np.random.uniform(-AXIS_WARP_BETA, AXIS_WARP_BETA) * np.sin(
        2 * np.pi * positions / length + np.random.uniform(0, 2 * np.pi)
    )
    warped = np.clip(positions + linear + nonlinear, 0, length - 1)
    return np.interp(positions, warped, values).astype(np.float32)


def augment_shift(values: np.ndarray) -> np.ndarray:
    """随机平移峰位，并以边界值填补空白位置。"""
    values = np.asarray(values, dtype=np.float32)
    shift = int(np.random.randint(-SHIFT_MAX, SHIFT_MAX + 1))
    if shift == 0:
        return values
    augmented = np.empty_like(values)
    if shift > 0:
        augmented[:shift] = values[0]
        augmented[shift:] = values[:-shift]
    else:
        shift = -shift
        augmented[-shift:] = values[-1]
        augmented[:-shift] = values[shift:]
    return augmented


def augment_broadening(values: np.ndarray) -> np.ndarray:
    """通过小高斯核卷积模拟峰展宽。"""
    values = np.asarray(values, dtype=np.float32)
    sigma = float(np.random.uniform(BROAD_SIGMA_MIN, BROAD_SIGMA_MAX))
    radius = max(int(BROAD_TRUNCATE * sigma + 0.5), 1)
    positions = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel /= kernel.sum() + 1e-8
    padded = np.pad(values, (radius, radius), mode="reflect")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def augment_mask_attenuation(values: np.ndarray) -> np.ndarray:
    """随机衰减一段带平滑边缘的局部谱段。"""
    values = np.asarray(values, dtype=np.float32)
    if MASK_WIDTH_MIN <= 0 or MASK_WIDTH_MIN >= len(values):
        return values
    width = int(np.random.randint(MASK_WIDTH_MIN, min(MASK_WIDTH_MAX, len(values) - 1) + 1))
    start = int(np.random.randint(0, len(values) - width))
    attenuation = float(np.random.uniform(MASK_ATTEN_MIN, MASK_ATTEN_MAX))
    window = np.ones(width, dtype=np.float32)
    edge = int(0.2 * width)
    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]
    augmented = values.copy()
    augmented[start : start + width] *= attenuation * window + (1.0 - window)
    return augmented
