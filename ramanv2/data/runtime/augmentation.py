"""训练集专用的随机光谱增强。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramanv2.core.config import TrainingConfig


# 分段增益：每段乘性系数以 1 为中心的标准差，段长按输入点计（约 80–240 cm^-1）。
PIECEWISE_GAIN_STD = 0.12
PIECEWISE_SEGMENT_MIN = 60
PIECEWISE_SEGMENT_MAX = 180

# 轴扰动：线性项与正弦项直接以 cm^-1 定义，位移最终硬截断到 ±5 cm^-1。
AXIS_WARP_LINEAR_CM = 2.0
AXIS_WARP_SINE_MIN_CM = 0.5
AXIS_WARP_SINE_MAX_CM = 2.5
AXIS_WARP_LIMIT_CM = 5.0

# 基线：线性倾斜、宽域高斯混合与 1200–1400 区域单峰的振幅均相对稳健振幅。
BASELINE_TILT_MIN = 0.02
BASELINE_TILT_MAX = 0.05

BASELINE_BROAD_COUNT_MIN = 1
BASELINE_BROAD_COUNT_MAX = 3
BASELINE_BROAD_CENTER_MIN = 700.0
BASELINE_BROAD_CENTER_MAX = 1750.0
BASELINE_BROAD_SIGMA_MIN = 40.0
BASELINE_BROAD_SIGMA_MAX = 150.0
BASELINE_BROAD_AMP_MIN = 0.05
BASELINE_BROAD_AMP_MAX = 0.15
BASELINE_BROAD_CAP = 0.30

BASELINE_REGION_CENTER_MIN = 1150.0
BASELINE_REGION_CENTER_MAX = 1450.0
BASELINE_REGION_SIGMA_MIN = 50.0
BASELINE_REGION_SIGMA_MAX = 120.0
BASELINE_REGION_AMP_MIN = 0.08
BASELINE_REGION_AMP_MAX = 0.18
BASELINE_REGION_CAP = 0.25


@dataclass(frozen=True)
class AugmentationSpec:
    """控制训练样本随机增强的概率与单样本最大叠加次数。"""

    p_piecewise_gain: float
    p_axis: float
    p_baseline_broad: float
    p_baseline_region: float
    max_pre_augs: int


def build_augmentation_spec(training_config: TrainingConfig) -> AugmentationSpec:
    """从训练配置提取随机增强规格。"""
    return AugmentationSpec(
        p_piecewise_gain=float(training_config.p_piecewise_gain),
        p_axis=float(training_config.p_axis),
        p_baseline_broad=float(training_config.p_baseline_broad),
        p_baseline_region=float(training_config.p_baseline_region),
        max_pre_augs=int(training_config.max_pre_augs),
    )


def augment_raw_spectrum(
    values: np.ndarray,
    augmentation_spec: AugmentationSpec,
    axis_cm: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """在归一化前随机组合区域增益、轴偏移与基线扰动。"""
    augmented = np.asarray(values, dtype=np.float32)
    axis = np.asarray(axis_cm, dtype=np.float32)
    operations = []
    if np.random.rand() < augmentation_spec.p_piecewise_gain:
        segments = _build_random_segments(
            len(augmented), PIECEWISE_SEGMENT_MIN, PIECEWISE_SEGMENT_MAX
        )
        operations.append(
            lambda items: augment_piecewise_gain(items, segments, PIECEWISE_GAIN_STD)
        )
    if np.random.rand() < augmentation_spec.p_axis:
        operations.append(lambda items: augment_axis_warp(items, axis))
    baseline_random = np.random.rand()
    if baseline_random < augmentation_spec.p_baseline_broad:
        operations.append(lambda items: augment_broad_baseline(items, axis))
    elif baseline_random < (
        augmentation_spec.p_baseline_broad + augmentation_spec.p_baseline_region
    ):
        operations.append(lambda items: augment_region_baseline(items, axis))

    np.random.shuffle(operations)
    for operation in operations[: augmentation_spec.max_pre_augs]:
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


def augment_axis_warp(values: np.ndarray, axis_cm: np.ndarray) -> np.ndarray:
    """轻微扭曲采样位置，位移以 cm^-1 定义并硬截断到 ±5 cm^-1。"""
    values = np.asarray(values, dtype=np.float32)
    length = values.size
    step_cm = float(axis_cm[1] - axis_cm[0])
    positions = np.arange(length, dtype=np.float32)
    centered = positions - (length - 1) / 2.0
    linear_cm = np.random.uniform(-AXIS_WARP_LINEAR_CM, AXIS_WARP_LINEAR_CM)
    linear = linear_cm / step_cm * centered / ((length - 1) / 2.0)
    sine_cm = np.random.uniform(AXIS_WARP_SINE_MIN_CM, AXIS_WARP_SINE_MAX_CM)
    nonlinear = sine_cm / step_cm * np.sin(
        2 * np.pi * positions / length + np.random.uniform(0, 2 * np.pi)
    )
    limit = AXIS_WARP_LIMIT_CM / step_cm
    displacement = np.clip(linear + nonlinear, -limit, limit)
    warped = np.clip(positions + displacement, 0, length - 1)
    return np.interp(positions, warped, values).astype(np.float32)


def augment_broad_baseline(values: np.ndarray, axis_cm: np.ndarray) -> np.ndarray:
    """叠加 1–3 个宽高斯背景与线性倾斜，模拟整段基线残差。"""
    values = np.asarray(values, dtype=np.float32)
    amplitude = _robust_amplitude(values)
    count = np.random.randint(BASELINE_BROAD_COUNT_MIN, BASELINE_BROAD_COUNT_MAX + 1)
    centers = np.random.uniform(BASELINE_BROAD_CENTER_MIN, BASELINE_BROAD_CENTER_MAX, count)
    sigmas = np.exp(
        np.random.uniform(
            np.log(BASELINE_BROAD_SIGMA_MIN), np.log(BASELINE_BROAD_SIGMA_MAX), count
        )
    )
    weights = np.random.uniform(
        BASELINE_BROAD_AMP_MIN, BASELINE_BROAD_AMP_MAX, count
    ) * amplitude
    curve = _linear_tilt(values.size, amplitude)
    for center, sigma, weight in zip(centers, sigmas, weights):
        curve += weight * np.exp(-0.5 * ((axis_cm - center) / sigma) ** 2)
    curve = np.clip(curve, -BASELINE_BROAD_CAP * amplitude, BASELINE_BROAD_CAP * amplitude)
    return _apply_baseline(values, curve)


def augment_region_baseline(values: np.ndarray, axis_cm: np.ndarray) -> np.ndarray:
    """叠加 1150–1450 cm^-1 附近的宽单峰背景与线性倾斜。"""
    values = np.asarray(values, dtype=np.float32)
    amplitude = _robust_amplitude(values)
    center = np.random.uniform(BASELINE_REGION_CENTER_MIN, BASELINE_REGION_CENTER_MAX)
    sigma = np.random.uniform(BASELINE_REGION_SIGMA_MIN, BASELINE_REGION_SIGMA_MAX)
    weight = (
        np.random.uniform(BASELINE_REGION_AMP_MIN, BASELINE_REGION_AMP_MAX)
        * amplitude
        * np.random.choice([-1.0, 1.0])
    )
    curve = _linear_tilt(values.size, amplitude)
    curve += weight * np.exp(-0.5 * ((axis_cm - center) / sigma) ** 2)
    curve = np.clip(curve, -BASELINE_REGION_CAP * amplitude, BASELINE_REGION_CAP * amplitude)
    return _apply_baseline(values, curve)


def _linear_tilt(length: int, amplitude: float) -> np.ndarray:
    """生成相对振幅 2–5% 的随机线性倾斜。"""
    positions = np.linspace(-0.5, 0.5, length, dtype=np.float32)
    scale = np.random.uniform(BASELINE_TILT_MIN, BASELINE_TILT_MAX) * amplitude
    return scale * np.random.choice([-1.0, 1.0]) * positions


def _apply_baseline(values: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """去均值后叠加基线曲线，避免引入整体偏移。"""
    return (values + (curve - curve.mean())).astype(np.float32)
