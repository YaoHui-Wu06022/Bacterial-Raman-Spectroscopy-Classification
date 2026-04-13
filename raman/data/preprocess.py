import math

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def SNV(x):
    """
    作用：
    - 消除整体强度尺度差异
    - 保留相对谱形信息
    - 对整条输入光谱做标准化
    """
    if isinstance(x, torch.Tensor):
        mean = x.mean()
        std = x.std(unbiased=False)
        return (x - mean) / (std + EPS)

    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + EPS)


def L2Normalize(x, eps=1e-8):
    """对单条光谱做 L2 标准化。"""
    if isinstance(x, torch.Tensor):
        norm = torch.sqrt(torch.sum(x**2)).clamp_min(eps)
        return x / norm

    x = np.asarray(x, dtype=np.float32)
    norm = max(float(np.sqrt(np.sum(x**2))), eps)
    return x / norm


def MinMaxNormalize(x, eps=1e-8):
    """对单条光谱做 Min-Max 归一化。"""
    if isinstance(x, torch.Tensor):
        min_v = torch.min(x)
        max_v = torch.max(x)
        denom = (max_v - min_v).clamp_min(eps)
        return (x - min_v) / denom

    x = np.asarray(x, dtype=np.float32)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    denom = max(max_v - min_v, eps)
    return (x - min_v) / denom


def load_arc_intensity(path):
    """读取单个 .arc_data 文件的强度列。"""
    data = np.loadtxt(path, dtype=np.float32)
    data = np.atleast_2d(data)
    return data[:, 1].astype(np.float32, copy=False)


def normalize_spectrum(x, norm_method):
    """按配置指定的方法对单条光谱做标准化。"""
    if norm_method == "snv":
        return SNV(x)
    if norm_method == "l2":
        return L2Normalize(x)
    if norm_method == "minmax":
        return MinMaxNormalize(x)
    raise ValueError(f"未知的 norm_method: {norm_method}")


def _robust_amp(x):
    """用 1% 和 99% 分位数估计稳健振幅。"""
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    amp = float(p99 - p1)
    return max(amp, 1e-6)


def _random_piecewise_segments(n, min_len=50, max_len=200):
    """按谱长自动切分若干不重叠区段。"""
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
    """按分段随机缩放，模拟相对峰高比例变化。"""
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
    """
    强度相关高斯噪声。

    噪声标准差形式为：
        sigma = a + b * |x|
    """
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
    """弱 baseline 扰动，模拟残余低频背景。"""
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


def aug_strong_baseline(x, amp_min=0.05, amp_max=0.15, n_knots_min=3, n_knots_max=6):
    """强 baseline 扰动，模拟更明显的批次或仪器背景差异。"""
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    length = len(x)

    n_knots = np.random.randint(n_knots_min, n_knots_max + 1)
    xs = np.linspace(0, length - 1, n_knots, dtype=np.float32)
    ys = np.random.uniform(-1.0, 1.0, n_knots).astype(np.float32)
    ys *= np.random.uniform(amp_min, amp_max) * amp

    baseline = np.interp(np.arange(length, dtype=np.float32), xs, ys).astype(np.float32)
    return x + baseline


def aug_axis_warp(x, config):
    """非刚性波数轴扰动，模拟轻微标定偏差。"""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    idx = np.arange(n, dtype=np.float32)
    center = (n - 1) / 2.0
    idx_norm = idx - center

    alpha = np.random.uniform(-config.axis_warp_alpha, config.axis_warp_alpha)
    warp_linear = alpha * idx_norm

    beta = np.random.uniform(-config.axis_warp_beta, config.axis_warp_beta)
    phase = np.random.uniform(0, 2 * np.pi)
    warp_nonlinear = beta * np.sin(2 * np.pi * idx / n + phase)

    idx_warped = idx + warp_linear + warp_nonlinear
    idx_warped = np.clip(idx_warped, 0, n - 1)
    y = np.interp(idx, idx_warped, x)
    return y.astype(np.float32)


def aug_shift(x, max_shift=3):
    """随机平移峰位，模拟轻微坐标偏移。"""
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
    """构造一维高斯卷积核。"""
    radius = int(truncate * sigma + 0.5)
    radius = max(radius, 1)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum() + EPS
    return kernel


def aug_broadening(x, sigma_min=0.4, sigma_max=1.0, truncate=3.0):
    """用小核高斯卷积模拟峰展宽。"""
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
    """局部衰减遮挡，模拟局部污染、局部失真或检测异常。"""
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


# RAW 数据增强入口
def augment_raw_spectrum(x, config):
    """
    - training=True : 做增强
    - 模型输入用 RAW
    """
    is_tensor = isinstance(x, torch.Tensor)
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
                gain_std=config.piecewise_gain_std,
            )
        )

    if np.random.rand() < config.p_noise:
        pre_ops.append(
            lambda z: aug_noise_gaussian(
                z,
                base_rel_min=config.noise_base_rel_min,
                base_rel_max=config.noise_base_rel_max,
                slope_rel_min=config.noise_slope_rel_min,
                slope_rel_max=config.noise_slope_rel_max,
            )
        )

    if np.random.rand() < config.p_axis:
        pre_ops.append(lambda z: aug_axis_warp(z, config))

    r = np.random.rand()
    if r < config.p_baseline_weak:
        pre_ops.append(
            lambda z: aug_weak_baseline(
                z,
                lin_min=config.baseline_lin_min,
                lin_max=config.baseline_lin_max,
                sin_min=config.baseline_sin_min,
                sin_max=config.baseline_sin_max,
                freq_min=config.baseline_freq_min,
                freq_max=config.baseline_freq_max,
            )
        )
    elif r < config.p_baseline_weak + config.p_baseline_strong:
        pre_ops.append(
            lambda z: aug_strong_baseline(
                z,
                amp_min=config.baseline_strong_amp_min,
                amp_max=config.baseline_strong_amp_max,
            )
        )

    np.random.shuffle(pre_ops)
    for fn in pre_ops[: config.max_pre_augs]:
        x = fn(x)

    if is_tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.astype(np.float32)


def augment_norm_spectrum(x, config):
    """在标准化后做弱形状扰动。"""
    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        device = x.device
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)

    post_ops = []
    if np.random.rand() < config.p_shift:
        post_ops.append(lambda z: aug_shift(z, config.shift_max))

    if np.random.rand() < config.p_broadening:
        post_ops.append(
            lambda z: aug_broadening(
                z,
                config.broad_sigma_min,
                config.broad_sigma_max,
                config.broad_truncate,
            )
        )

    if np.random.rand() < config.p_cut:
        post_ops.append(
            lambda z: aug_mask_attenuate(
                z,
                width_min=config.mask_width_min,
                width_max=config.mask_width_max,
                atten_min=config.mask_atten_min,
                atten_max=config.mask_atten_max,
            )
        )

    np.random.shuffle(post_ops)
    for fn in post_ops[: config.max_post_augs]:
        x = fn(x)

    if is_tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.astype(np.float32)


def _validate_sg_params(window_length, polyorder, deriv):
    """对 SG 参数做基本合法性检查。"""
    if window_length is None:
        raise ValueError("window_length 不能为空。")

    window_length = int(window_length)
    polyorder = int(polyorder)
    deriv = int(deriv)

    if window_length <= 0:
        raise ValueError(f"window_length 必须大于 0，当前为 {window_length}。")
    if window_length % 2 == 0:
        raise ValueError(f"window_length 必须是奇数，当前为 {window_length}。")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length 必须大于 polyorder，当前 window_length={window_length}, polyorder={polyorder}。"
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(
            f"deriv 必须落在 [0, polyorder]，当前 deriv={deriv}, polyorder={polyorder}。"
        )

    return window_length, polyorder, deriv


def sg_coeff(window_length, polyorder, deriv):
    """生成 Savitzky-Golay 卷积核系数。"""
    window_length, polyorder, deriv = _validate_sg_params(window_length, polyorder, deriv)

    half = (window_length - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    design = np.vander(x, N=polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(design)
    coeff = pinv[deriv] * math.factorial(deriv)
    coeff = coeff.astype(np.float32)

    if not np.isfinite(coeff).all():
        raise ValueError(
            f"sg_coeff 生成了非有限值：window_length={window_length}, polyorder={polyorder}, deriv={deriv}。"
        )

    return coeff


def sg_kernel(window_length, polyorder, deriv, device):
    """将 SG 系数包装成 torch 卷积核。"""
    coeff = sg_coeff(window_length, polyorder, deriv)
    kernel = torch.tensor(coeff, dtype=torch.float32, device=device)
    return kernel.view(1, 1, -1)


def build_sg_kernels(config, device):
    """按当前配置一次性构造 smooth 和 d1 的 SG 卷积核。"""
    return (
        sg_kernel(config.win_smooth, 3, 0, device),
        sg_kernel(config.win1, 3, 1, device),
    )


def build_pre_smooth_source(signal, config, sg_smooth):
    """为 smooth 通道构造“先平滑、后标准化”的输入源。"""
    smooth = F.conv1d(signal, sg_smooth, padding=config.win_smooth // 2)
    return smooth[0, 0]


def build_pre_d1_source(signal, config, sg_smooth, sg_d1):
    """为 d1 通道构造“先平滑、再求导、后标准化”的输入源。"""
    smooth = F.conv1d(signal, sg_smooth, padding=config.win_smooth // 2)
    d1 = F.conv1d(smooth, sg_d1, padding=config.win1 // 2)[0, 0]
    return d1 / config.delta


def build_input_channels(signal, config, smooth_signal=None, raw_signal=None, d1_signal=None):
    """将已经准备好的各支路信号堆叠成最终的 [C, L] 输入。"""
    channels = [signal[0, 0]]

    if config.smooth_use:
        if smooth_signal is None:
            raise ValueError("smooth_use=True 时必须显式传入 smooth_signal。")
        channels.append(smooth_signal[0, 0])

    if getattr(config, "raw_use", False):
        if raw_signal is None:
            raise ValueError("raw_use=True 时必须显式传入 raw_signal。")
        channels.append(raw_signal[0, 0])

    if config.d1_use:
        if d1_signal is None:
            raise ValueError("d1_use=True 时必须显式传入 d1_signal。")
        channels.append(d1_signal[0, 0])

    if len(channels) != config.in_channels:
        raise ValueError(
            f"通道数不匹配：实际构造了 {len(channels)} 个通道，但 config.in_channels={config.in_channels}。"
        )

    return torch.stack(channels, dim=0)


def build_model_input(raw_intensity, config, sg_smooth, sg_d1, device, augment=False):
    """
    将一条原始强度光谱转换成模型输入。

    当前始终返回单个 [C, L]，不再在这里生成双视图。
    """
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

    raw_signal = mother_tensor if getattr(config, "raw_use", False) else None
    signal = torch.as_tensor(base_x, dtype=torch.float32, device=device).view(1, 1, -1)

    return build_input_channels(
        signal,
        config,
        smooth_signal=smooth_signal,
        raw_signal=raw_signal,
        d1_signal=d1_signal,
    )


class InputPreprocessor:
    """统一 train / eval / predict 的输入构造逻辑。"""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.sg_smooth, self.sg_d1 = build_sg_kernels(config, device)

    def __call__(self, path):
        return self.preprocess_arc(path)

    def preprocess_arc(self, path):
        """对单个 .arc_data 文件做推理侧输入构造。"""
        raw_intensity = load_arc_intensity(path)
        X = build_model_input(
            raw_intensity,
            config=self.config,
            sg_smooth=self.sg_smooth,
            sg_d1=self.sg_d1,
            device=self.device,
            augment=False,
        )
        return X.unsqueeze(0)
