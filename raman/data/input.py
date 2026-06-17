"""模型在线输入链路

集中放置训练、评估、推理会直接用到的输入构造逻辑：
归一化、随机增强、SG 卷积核和最终 [C, L] 张量构造
"""

import math

import numpy as np
from scipy.interpolate import PchipInterpolator

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

from raman.data.io import load_arc_intensity


# ===== 归一化 =====

EPS = 1e-8
SUPPORTED_NORM_METHODS = ("snv", "minmax", "l2")


def _normalize_method(method):
    method = str(method).lower()
    if method not in SUPPORTED_NORM_METHODS:
        supported = ", ".join(SUPPORTED_NORM_METHODS)
        raise ValueError(f"未知的 norm_method: {method}，可选值: {supported}")
    return method


def _standardize_numpy(values, eps):
    mean = np.mean(values, axis=-1, keepdims=True)
    std = np.std(values, axis=-1, keepdims=True)
    return (values - mean) / (std + eps)


def _scale_minmax_numpy(values, eps):
    min_value = np.min(values, axis=-1, keepdims=True)
    max_value = np.max(values, axis=-1, keepdims=True)
    denom = np.maximum(max_value - min_value, eps)
    return (values - min_value) / denom


def _scale_l2_numpy(values, eps):
    norm = np.sqrt(np.sum(values * values, axis=-1, keepdims=True))
    norm = np.maximum(norm, eps)
    return values / norm


def _normalize_numpy(values, method, eps):
    if method == "snv":
        return _standardize_numpy(values, eps)
    if method == "minmax":
        return _scale_minmax_numpy(values, eps)
    if method == "l2":
        return _scale_l2_numpy(values, eps)
    raise AssertionError(method)


def _normalize_numpy_preserve_nan(values, method, eps):
    output = np.full(values.shape, np.nan, dtype=np.float32)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_output = output.reshape(-1, output.shape[-1])
    for idx, row in enumerate(flat_values):
        finite = np.isfinite(row)
        if finite.any():
            normalized = _normalize_numpy(row[finite].astype(np.float32, copy=False), method, eps)
            flat_output[idx, finite] = normalized.astype(np.float32, copy=False)
    return output


def _standardize_tensor(values, eps):
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, unbiased=False, keepdim=True)
    return (values - mean) / (std + eps)


def _scale_minmax_tensor(values, eps):
    min_value = values.amin(dim=-1, keepdim=True)
    max_value = values.amax(dim=-1, keepdim=True)
    denom = (max_value - min_value).clamp_min(eps)
    return (values - min_value) / denom


def _scale_l2_tensor(values, eps):
    if torch is None:
        raise RuntimeError("torch is required for tensor normalization")
    norm = torch.sqrt(torch.sum(values * values, dim=-1, keepdim=True)).clamp_min(eps)
    return values / norm


def _normalize_tensor(values, method, eps):
    if method == "snv":
        return _standardize_tensor(values, eps)
    if method == "minmax":
        return _scale_minmax_tensor(values, eps)
    if method == "l2":
        return _scale_l2_tensor(values, eps)
    raise AssertionError(method)


def _normalize_tensor_preserve_nan(values, method, eps):
    if torch is None:
        raise RuntimeError("torch is required for tensor normalization")
    output = torch.full_like(values, torch.nan)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_output = output.reshape(-1, output.shape[-1])
    for idx, row in enumerate(flat_values):
        finite = torch.isfinite(row)
        if bool(finite.any()):
            flat_output[idx, finite] = _normalize_tensor(row[finite], method, eps)
    return output


def normalize_spectrum(data, method, eps=EPS, preserve_nan=False):
    """按配置方法归一化光谱，1D 处理单条谱，2D 按最后一维逐条处理"""
    method = _normalize_method(method)

    if torch is not None and isinstance(data, torch.Tensor):
        values = data.to(dtype=torch.float32)
        if values.ndim == 0:
            raise ValueError("normalize_spectrum 需要至少一维输入")
        if preserve_nan:
            return _normalize_tensor_preserve_nan(values, method, eps)
        return _normalize_tensor(values, method, eps)

    values = np.asarray(data, dtype=np.float32)
    if values.ndim == 0:
        raise ValueError("normalize_spectrum 需要至少一维输入")
    if preserve_nan:
        return _normalize_numpy_preserve_nan(values, method, eps)
    return _normalize_numpy(values, method, eps).astype(np.float32, copy=False)


def _validate_sg_params(window_length, polyorder, deriv):
    """检查 SG 窗口、阶数和导数阶数是否合法"""
    if window_length is None:
        raise ValueError("window_length 不能为空")
    window_length = int(window_length)
    polyorder = int(polyorder)
    deriv = int(deriv)

    if window_length <= 0:
        raise ValueError(f"window_length 必须大于 0，当前为 {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length 必须是奇数，当前为 {window_length}")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length 必须大于 polyorder，当前 window_length={window_length}, polyorder={polyorder}"
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(
            f"deriv 必须落在 [0, polyorder]，当前 deriv={deriv}, polyorder={polyorder}"
        )

    return window_length, polyorder, deriv


def sg_coeff(window_length, polyorder, deriv):
    """生成 Savitzky-Golay 卷积核系数"""
    window_length, polyorder, deriv = _validate_sg_params(window_length, polyorder, deriv)

    half = (window_length - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    design = np.vander(x, N=polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(design)
    coeff = pinv[deriv] * math.factorial(deriv)
    coeff = coeff.astype(np.float32)

    if not np.isfinite(coeff).all():
        raise ValueError(
            f"sg_coeff 生成了非有限值：window_length={window_length}, polyorder={polyorder}, deriv={deriv}"
        )

    return coeff


def sg_kernel(window_length, polyorder, deriv, device):
    """把 SG 系数包装成 torch 一维卷积核"""
    coeff = sg_coeff(window_length, polyorder, deriv)
    kernel = torch.tensor(coeff, dtype=torch.float32, device=device)
    return kernel.view(1, 1, -1)


def build_sg_kernels(config, device):
    """按当前配置构造 smooth 和 d1 两个 SG 卷积核"""
    return (
        sg_kernel(config.win_smooth, 3, 0, device),
        sg_kernel(config.win1, 3, 1, device),
    )


# ===== 在线训练增强 =====
PIECEWISE_GAIN_STD = 0.12
NOISE_BASE_REL_MIN = 0.005
NOISE_BASE_REL_MAX = 0.02
NOISE_SLOPE_REL_MIN = 0.0
NOISE_SLOPE_REL_MAX = 0.015
AXIS_WARP_ALPHA = 0.002
AXIS_WARP_BETA = 1.0
BASELINE_LIN_MIN = 0.0
BASELINE_LIN_MAX = 0.02
BASELINE_SIN_MIN = 0.0
BASELINE_SIN_MAX = 0.01
BASELINE_FREQ_MIN = 0.5
BASELINE_FREQ_MAX = 2.0
BASELINE_STRONG_AMP_MIN = 0.05
BASELINE_STRONG_AMP_MAX = 0.15
BASELINE_STRONG_KNOTS_MIN = 5
BASELINE_STRONG_KNOTS_MAX = 9
SHIFT_MAX = 3
BROAD_SIGMA_MIN = 0.6
BROAD_SIGMA_MAX = 1.2
BROAD_TRUNCATE = 3.0
MASK_WIDTH_MIN = 20
MASK_WIDTH_MAX = 50
MASK_ATTEN_MIN = 0.5
MASK_ATTEN_MAX = 0.8


def _is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _robust_amp(x):
    """用 1% 和 99% 分位数估计稳定振幅"""
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    amp = float(p99 - p1)
    return max(amp, 1e-6)


def _random_piecewise_segments(n, min_len=50, max_len=200):
    """按随机长度切分为若干不重叠片段"""
    segments = []
    i = 0
    while i < n:
        seg_len = np.random.randint(min_len, max_len + 1)
        left = i
        right = min(n, i + seg_len)
        segments.append((left, right))
        i = right
    return segments


def aug_piecewise_gain(x, segments, gain_std=0.15):
    """按分段随机缩放，模拟相对峰高比例变化"""
    x = np.asarray(x, dtype=np.float32)
    out = x.copy()
    for left, right in segments:
        gain = np.random.normal(1.0, gain_std)
        out[left:right] *= gain
    return out


def aug_noise_gaussian(
    x,
    base_rel_min=0.005,
    base_rel_max=0.02,
    slope_rel_min=0.0,
    slope_rel_max=0.015,
):
    """加入强度相关高斯噪声"""
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    base = np.random.uniform(base_rel_min, base_rel_max) * amp
    slope = np.random.uniform(slope_rel_min, slope_rel_max)
    sigma = base + slope * np.abs(x)
    noise = np.random.randn(*x.shape).astype(np.float32) * sigma.astype(np.float32)
    return x + noise


def aug_weak_baseline(
    x,
    lin_min=0.0,
    lin_max=0.02,
    sin_min=0.0,
    sin_max=0.01,
    freq_min=0.5,
    freq_max=2.0,
):
    """加入弱基线扰动，模拟残余低频背景"""
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    length = len(x)
    t = np.linspace(-0.5, 0.5, length, dtype=np.float32)

    a = np.random.uniform(lin_min, lin_max) * amp
    b = np.random.uniform(sin_min, sin_max) * amp
    freq = np.random.uniform(freq_min, freq_max)
    phase = np.random.uniform(0, 2 * np.pi)

    baseline = a * t + b * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    return x + baseline


def aug_strong_baseline(
    x,
    amp_min=0.05,
    amp_max=0.15,
    n_knots_min=BASELINE_STRONG_KNOTS_MIN,
    n_knots_max=BASELINE_STRONG_KNOTS_MAX,
):
    """加入更明显的平滑低频背景扰动，模拟批次或仪器背景差异"""
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    length = len(x)
    if length < 2:
        return x.copy()

    n_knots_min = max(2, min(int(n_knots_min), length))
    n_knots_max = max(n_knots_min, min(int(n_knots_max), length))
    n_knots = np.random.randint(n_knots_min, n_knots_max + 1)
    xs = np.linspace(0, length - 1, n_knots, dtype=np.float32)
    ys = np.random.uniform(-1.0, 1.0, n_knots).astype(np.float32)
    ys *= np.random.uniform(amp_min, amp_max) * amp

    grid = np.arange(length, dtype=np.float32)
    baseline = PchipInterpolator(xs, ys)(grid).astype(np.float32)
    return x + baseline


def aug_axis_warp(x):
    """轻微扭曲采样轴，模拟标定偏差"""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    idx = np.arange(n, dtype=np.float32)
    center = (n - 1) / 2.0
    idx_norm = idx - center
    alpha = np.random.uniform(-AXIS_WARP_ALPHA, AXIS_WARP_ALPHA)
    warp_linear = alpha * idx_norm
    beta = np.random.uniform(-AXIS_WARP_BETA, AXIS_WARP_BETA)
    phase = np.random.uniform(0, 2 * np.pi)
    warp_nonlinear = beta * np.sin(2 * np.pi * idx / n + phase)

    idx_warped = idx + warp_linear + warp_nonlinear
    idx_warped = np.clip(idx_warped, 0, n - 1)
    y = np.interp(idx, idx_warped, x)
    return y.astype(np.float32)


def aug_shift(x, max_shift=3):
    """随机平移峰位，模拟轻微坐标偏移"""
    x = np.asarray(x, dtype=np.float32)
    if max_shift <= 0:
        return x

    shift = int(np.random.randint(-max_shift, max_shift + 1))
    if shift == 0:
        return x

    out = np.empty_like(x)
    if shift > 0:
        out[:shift] = x[0]
        out[shift:] = x[:-shift]
    else:
        shift = -shift
        out[-shift:] = x[-1]
        out[:-shift] = x[shift:]
    return out


def _gauss_kernel1d(sigma, truncate=3.0):
    """构造一维高斯卷积核"""
    radius = int(truncate * sigma + 0.5)
    radius = max(radius, 1)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum() + EPS
    return kernel


def aug_broadening(x, sigma_min=0.4, sigma_max=1.0, truncate=3.0):
    """用小高斯核模拟峰展宽"""
    x = np.asarray(x, dtype=np.float32)
    sigma = float(np.random.uniform(sigma_min, sigma_max))
    kernel = _gauss_kernel1d(sigma, truncate=truncate)
    pad = len(kernel) // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(x_pad, kernel, mode="valid").astype(np.float32)


def aug_mask_attenuate(
    x,
    width_min=10,
    width_max=40,
    atten_min=0.0,
    atten_max=0.3,
):
    """局部衰减遮挡，模拟局部污染、失真或检测异常"""
    x = np.asarray(x, dtype=np.float32)
    length = len(x)
    if width_min <= 0 or width_min >= length:
        return x

    width = int(np.random.randint(width_min, min(width_max, length - 1) + 1))
    start = int(np.random.randint(0, length - width))
    atten = float(np.random.uniform(atten_min, atten_max))

    out = x.copy()
    window = np.ones(width, dtype=np.float32)
    edge = int(0.2 * width)
    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]

    factor = atten * window + (1.0 - window)
    out[start : start + width] *= factor
    return out


def augment_raw_spectrum(x, config):
    """在归一化前对原始强度谱做随机增强"""
    is_tensor = _is_tensor(x)
    if is_tensor:
        device = x.device
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)

    pre_ops = []

    if np.random.rand() < config.p_piecewise_gain:
        segments = _random_piecewise_segments(len(x), min_len=60, max_len=180)
        pre_ops.append(
            lambda z: aug_piecewise_gain(
                z,
                segments=segments,
                gain_std=PIECEWISE_GAIN_STD,
            )
        )

    if np.random.rand() < config.p_noise:
        pre_ops.append(
            lambda z: aug_noise_gaussian(
                z,
                base_rel_min=NOISE_BASE_REL_MIN,
                base_rel_max=NOISE_BASE_REL_MAX,
                slope_rel_min=NOISE_SLOPE_REL_MIN,
                slope_rel_max=NOISE_SLOPE_REL_MAX,
            )
        )

    if np.random.rand() < config.p_axis:
        pre_ops.append(lambda z: aug_axis_warp(z))

    r = np.random.rand()
    if r < config.p_baseline_weak:
        pre_ops.append(
            lambda z: aug_weak_baseline(
                z,
                lin_min=BASELINE_LIN_MIN,
                lin_max=BASELINE_LIN_MAX,
                sin_min=BASELINE_SIN_MIN,
                sin_max=BASELINE_SIN_MAX,
                freq_min=BASELINE_FREQ_MIN,
                freq_max=BASELINE_FREQ_MAX,
            )
        )
    elif r < config.p_baseline_weak + config.p_baseline_strong:
        pre_ops.append(
            lambda z: aug_strong_baseline(
                z,
                amp_min=BASELINE_STRONG_AMP_MIN,
                amp_max=BASELINE_STRONG_AMP_MAX,
                n_knots_min=BASELINE_STRONG_KNOTS_MIN,
                n_knots_max=BASELINE_STRONG_KNOTS_MAX,
            )
        )

    np.random.shuffle(pre_ops)
    for fn in pre_ops[: config.max_pre_augs]:
        x = fn(x)

    if is_tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.astype(np.float32)


def augment_norm_spectrum(x, config):
    """在归一化后做弱形状扰动"""
    is_tensor = _is_tensor(x)
    if is_tensor:
        device = x.device
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)

    post_ops = []
    if np.random.rand() < config.p_shift:
        post_ops.append(lambda z: aug_shift(z, SHIFT_MAX))

    if np.random.rand() < config.p_broadening:
        post_ops.append(
            lambda z: aug_broadening(
                z,
                BROAD_SIGMA_MIN,
                BROAD_SIGMA_MAX,
                BROAD_TRUNCATE,
            )
        )

    if np.random.rand() < config.p_cut:
        post_ops.append(
            lambda z: aug_mask_attenuate(
                z,
                width_min=MASK_WIDTH_MIN,
                width_max=MASK_WIDTH_MAX,
                atten_min=MASK_ATTEN_MIN,
                atten_max=MASK_ATTEN_MAX,
            )
        )

    np.random.shuffle(post_ops)
    for fn in post_ops[: config.max_post_augs]:
        x = fn(x)

    if is_tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.astype(np.float32)


# ===== 模型输入构造 =====
def build_pre_smooth_source(signal, config, sg_smooth):
    """为 smooth 通道构造先平滑、后归一化的输入源"""
    smooth = F.conv1d(signal, sg_smooth, padding=config.win_smooth // 2)
    return smooth[0, 0]


def build_pre_d1_source(signal, config, sg_smooth, sg_d1):
    """为 d1 通道构造先平滑、再求导、后归一化的输入源"""
    smooth = F.conv1d(signal, sg_smooth, padding=config.win_smooth // 2)
    d1 = F.conv1d(smooth, sg_d1, padding=config.win1 // 2)[0, 0]
    return d1 / config.delta


def build_input_channels(signal, config, smooth_signal=None, d1_signal=None):
    """将各输入分支堆叠成最终 [C, L] 输入"""
    channels = [signal[0, 0]]

    if config.smooth_use:
        if smooth_signal is None:
            raise ValueError("smooth_use=True 时必须显式传入 smooth_signal")
        channels.append(smooth_signal[0, 0])

    if config.d1_use:
        if d1_signal is None:
            raise ValueError("d1_use=True 时必须显式传入 d1_signal")
        channels.append(d1_signal[0, 0])

    if len(channels) != config.in_channels:
        raise ValueError(
            f"通道数不匹配：实际构造了 {len(channels)} 个通道，但 config.in_channels={config.in_channels}"
        )

    return torch.stack(channels, dim=0)


def build_model_input(raw_intensity, config, sg_smooth, sg_d1, device, augment=False):
    """把单条强度谱转换成模型输入，返回单个 [C, L] 张量"""
    mother_raw = np.asarray(raw_intensity, dtype=np.float32)
    if augment:
        mother_raw = augment_raw_spectrum(mother_raw, config)
    else:
        mother_raw = mother_raw.copy()

    mother_tensor = torch.as_tensor(mother_raw, dtype=torch.float32, device=device).view(1, 1, -1)

    base_x = normalize_spectrum(mother_raw, config.norm_method)
    if augment:
        base_x = augment_norm_spectrum(base_x, config)

    smooth_signal = None
    if config.smooth_use:
        smooth_x = build_pre_smooth_source(mother_tensor, config, sg_smooth)
        smooth_x = normalize_spectrum(smooth_x, config.norm_method)
        smooth_signal = smooth_x.view(1, 1, -1)

    d1_signal = None
    if config.d1_use:
        d1_x = build_pre_d1_source(mother_tensor, config, sg_smooth, sg_d1)
        d1_x = normalize_spectrum(d1_x, config.norm_method)
        d1_signal = d1_x.view(1, 1, -1)

    signal = torch.as_tensor(base_x, dtype=torch.float32, device=device).view(1, 1, -1)

    return build_input_channels(
        signal,
        config,
        smooth_signal=smooth_signal,
        d1_signal=d1_signal,
    )


class InputPreprocessor:
    """统一 train / eval / predict 的输入构造逻辑"""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.sg_smooth, self.sg_d1 = build_sg_kernels(config, device)

    def __call__(self, path):
        return self.preprocess_arc(path)

    def preprocess_arc(self, path):
        """对单个 .arc_data 文件构造推理侧输入"""
        raw_intensity = load_arc_intensity(path)
        x = build_model_input(
            raw_intensity,
            config=self.config,
            sg_smooth=self.sg_smooth,
            sg_d1=self.sg_d1,
            device=self.device,
            augment=False,
        )
        return x.unsqueeze(0)

