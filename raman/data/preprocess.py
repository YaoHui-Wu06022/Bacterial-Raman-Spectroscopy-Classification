def _PREPROCESS_PIPELINE_DOC():
    #   RAW intensity
    #     ├─ 分段强度比例扰动（成分比例差异）
    #     ├─ 噪声（高斯 / 强度相关，二选一）
    #     ├─ baseline 扰动（弱 / 强，二选一）
    #     ├─ 波数轴非刚性扰动
    #     ↓
    #   SNV / L2 normalize
    #     ↓
    #   形状空间增强
    #     ├─ 峰位平移
    #     ├─ 峰展宽
    #     ├─ 局部衰减遮挡
    # - pre-augment（RAW 域）主要模拟“跨批次 / 跨仪器差异”
    # - post-augment（SNV 后）主要模拟“重复测量不稳定性”
    pass

import math
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8  # 数值稳定项，防止除零

# SNV（逐样本）
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
    else:
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x)
        std  = np.std(x)
        return (x - mean) / (std + EPS)

def L2Normalize(x, eps=1e-8):
    """
    L2 标准化（逐样本）
    - 消除整体强度尺度
    - 保留谱形方向
    - 不强制零均值
    """
    if isinstance(x, torch.Tensor):
        norm = torch.sqrt(torch.sum(x ** 2)).clamp_min(eps)
        return x / norm
    else:
        x = np.asarray(x, dtype=np.float32)
        norm = np.sqrt(np.sum(x ** 2))
        norm = max(norm, eps)
        return x / norm

def MinMaxNormalize(x, eps=1e-8):
    """
    Min-Max 归一化（逐样本 / per-spectrum）
    作用：
    - 将单条光谱缩放到 [0, 1]
    - 不引入跨样本统计
    """
    if isinstance(x, torch.Tensor):
        min_v = torch.min(x)
        max_v = torch.max(x)
        denom = (max_v - min_v).clamp_min(eps)
        return (x - min_v) / denom
    else:
        x = np.asarray(x, dtype=np.float32)
        min_v = np.min(x)
        max_v = np.max(x)
        denom = max(max_v - min_v, eps)
        return (x - min_v) / denom


def load_arc_intensity(path):
    """
    读取 .arc_data 文件的强度列，统一返回 float32 一维数组。
    """
    data = np.loadtxt(path, dtype=np.float32)
    data = np.atleast_2d(data)
    return data[:, 1].astype(np.float32, copy=False)


def normalize_spectrum(x, norm_method):
    """
    按配置指定的方式对单条光谱做标准化。
    """
    if norm_method == "snv":
        return SNV(x)
    if norm_method == "l2":
        return L2Normalize(x)
    if norm_method == "minmax":
        return MinMaxNormalize(x)
    raise ValueError(f"Unknown norm_method: {norm_method}")

def _robust_amp(x):
    """
    估计光谱的稳健振幅范围（robust amplitude）
    - 使用 1% 和 99% 分位数抑制极端值影响
    """
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    amp = float(p99 - p1)
    return max(amp, 1e-6)

def _random_piecewise_segments(n, min_len=50, max_len=200):
    """
    根据谱长自动生成若干非重叠区段
    物理意义：
    - 模拟不同 Raman band 相对比例变化
    - 不依赖固定波数位置（更泛化）
    """
    segments = []
    i = 0
    while i < n:
        seg_len = np.random.randint(min_len, max_len + 1)
        l = i
        r = min(n, i + seg_len)
        segments.append((l, r))
        i = r
    return segments

# RAW 域增强
# - 发生在标准化之前
# - 主要模拟仪器和采集条件变化
# - 对原始强度输入影响最直接
def aug_piecewise_gain(x, segments, gain_std=0.15):
    """
    分段幅度扰动
    模拟：
    - 不同培养条件
    - 生化组分比例变化
    - 峰位不变，但相对高度系统性改变
    """
    x = np.asarray(x, dtype=np.float32)
    out = x.copy()
    for (l, r) in segments:
        g = np.random.normal(1.0, gain_std)
        out[l:r] *= g
    return out

def aug_noise_gaussian(x, rel_min=0.005, rel_max=0.02):
    """
    高斯噪声增强
    噪声幅度与光谱振幅相关，
    避免强谱 / 弱谱噪声比例失衡
    峰形：轻微抖动
    基线：随机抖动
    """
    # 相对噪声：sigma = rel * robust_amp
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    rel = np.random.uniform(rel_min, rel_max)
    sigma = rel * amp
    noise = sigma * np.random.randn(*x.shape).astype(np.float32)
    return x + noise

def aug_noise_poisson(x, strength_min=0.0, strength_max=0.015):
    """
    强度相关噪声
    噪声幅度与 |x| 相关
    """
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    k = np.random.uniform(strength_min, strength_max) * amp
    max_abs = np.max(np.abs(x)) + EPS
    # sqrt(|x|) 型噪声（光子统计的味道）
    sigma = k * np.sqrt(np.abs(x) / max_abs)
    noise = sigma * np.random.randn(*x.shape).astype(np.float32)
    return x + noise

# 模拟基线去除不干净
def aug_weak_baseline(x,
                      lin_min=0.0, lin_max=0.02,
                      sin_min=0.0, sin_max=0.01,
                      freq_min=0.5, freq_max=2.0):
    """
    弱 baseline 扰动
    模拟：baseline 去不干净的残余低频背景
    适用于：同一实验条件下的小鲁棒性
    峰形：整体被“托起 / 压低”
    背景出现缓慢起伏
    """
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    L = len(x)
    t = np.linspace(-0.5, 0.5, L, dtype=np.float32)

    a = np.random.uniform(lin_min, lin_max) * amp
    b = np.random.uniform(sin_min, sin_max) * amp
    f = np.random.uniform(freq_min, freq_max)
    phi = np.random.uniform(0, 2*np.pi)

    baseline = a * t + b * np.sin(2*np.pi*f*t + phi).astype(np.float32)
    return x + baseline

def aug_strong_baseline(x,
                        amp_min=0.05, amp_max=0.15,
                        n_knots_min=3, n_knots_max=6):
    """
    强 baseline 扰动（跨域级）
    模拟：
    - 不同实验批次 / 不同仪器的荧光背景
    - 探测器整体响应曲线差异
    峰形：叠加在一个明显的“弯曲底座”上
    局部峰对比度被压缩或放大
    """
    x = np.asarray(x, dtype=np.float32)
    amp = _robust_amp(x)
    L = len(x)

    # ---- baseline 控制点（极低频）----
    n_knots = np.random.randint(n_knots_min, n_knots_max + 1)
    xs = np.linspace(0, L - 1, n_knots, dtype=np.float32)
    ys = np.random.uniform(-1.0, 1.0, n_knots).astype(np.float32)

    # 振幅控制
    scale = np.random.uniform(amp_min, amp_max) * amp
    ys *= scale

    # 插值生成平滑 baseline
    baseline = np.interp(
        np.arange(L, dtype=np.float32),
        xs,
        ys
    ).astype(np.float32)
    return x + baseline

def aug_axis_warp(x, config):
    """
    非刚性波数轴扰动（axis warp）
    物理意义：
    - 模拟仪器标定误差 / 波数轴非线性
    - 只允许低频、小幅、单调扰动
    峰位：整体微小偏移（非刚性）
    峰间距：略有拉伸/压缩
    峰形：基本保持
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    idx = np.arange(n, dtype=np.float32)
    center = (n - 1) / 2.0
    idx_norm = idx - center

    # -------- 线性漂移（标定斜率误差）--------
    alpha = np.random.uniform(
        -config.axis_warp_alpha,
        config.axis_warp_alpha
    )
    warp_linear = alpha * idx_norm

    # -------- 低频非线性扰动 --------
    beta = np.random.uniform(
        -config.axis_warp_beta,
        config.axis_warp_beta
    )
    phase = np.random.uniform(0, 2 * np.pi)
    warp_nonlinear = beta * np.sin(
        2 * np.pi * idx / n + phase
    )

    idx_warped = idx + warp_linear + warp_nonlinear
    idx_warped = np.clip(idx_warped, 0, n - 1)

    # 插值回原采样网格
    y = np.interp(idx, idx_warped, x)

    return y.astype(np.float32)

# 标准化后形状增强
# - 发生在标准化之后
# - 主要影响峰位置与峰形
# - 不改变整体强度统计
def aug_shift(x, max_shift=3):
    """
    随机峰位平移
    模拟：
    - 光谱轴轻微校准误差
    - 仪器重复测量的系统漂移
    """
    x = np.asarray(x, dtype=np.float32)
    if max_shift <= 0:
        return x
    s = np.random.randint(-max_shift, max_shift + 1)
    if s == 0:
        return x

    x2 = np.empty_like(x)
    if s > 0:
        x2[:s] = x[0]
        x2[s:] = x[:-s]
    else:
        k = -s
        x2[-k:] = x[-1]
        x2[:-k] = x[k:]
    return x2

def _gauss_kernel1d(sigma, truncate=3.0):
    radius = int(truncate * sigma + 0.5)
    radius = max(radius, 1)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k /= (k.sum() + EPS)
    return k

def aug_broadening(x, sigma_min=0.4, sigma_max=1.0, truncate=3.0):
    """
    小核高斯展宽
    设计约束：
    - 仅局部展宽峰形
    - 避免全谱模糊导致物理语义丢失
    """
    x = np.asarray(x, dtype=np.float32)
    sigma = float(np.random.uniform(sigma_min, sigma_max))
    k = _gauss_kernel1d(sigma, truncate=truncate)
    pad = len(k) // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(x_pad, k, mode="valid").astype(np.float32)

def aug_mask_attenuate(x, width_min=10, width_max=40,
                       atten_min=0.0, atten_max=0.3):
    """
    局部衰减遮挡（带平滑边缘）
    模拟：
    - 局部污染
    - 探测器异常
    - 坏点或饱和区域
    """
    x = np.asarray(x, dtype=np.float32)
    L = len(x)
    if width_min <= 0 or width_min >= L:
        return x

    w = int(np.random.randint(width_min, min(width_max, L - 1) + 1))
    start = int(np.random.randint(0, L - w))
    atten = float(np.random.uniform(atten_min, atten_max))

    x2 = x.copy()

    # -----------------------------
    # 平滑遮挡窗（cosine taper）
    # -----------------------------
    window = np.ones(w, dtype=np.float32)
    edge = int(0.2 * w)  # 20% 作为过渡区

    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]

    # atten 只作用在中心，边缘自然过渡
    factor = atten * window + (1.0 - window)

    x2[start:start + w] *= factor
    return x2

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

    # 1. 跨批次风格增强
    # --- 分段峰比例扰动（核心） ---
    if np.random.rand() < config.p_piecewise_gain:
        segments = _random_piecewise_segments(
            len(x),
            min_len = 60,
            max_len = 180
        )
        pre_ops.append(lambda z: aug_piecewise_gain(
            z,
            segments = segments,
            gain_std = config.piecewise_gain_std
        ))


    # 2. 噪声增强，最多启用 1 项
    r = np.random.rand()
    if r < config.p_noise:
        pre_ops.append(lambda z: aug_noise_gaussian(
            z, config.noise_rel_min, config.noise_rel_max))
    elif r < config.p_noise + config.p_poisson:
        pre_ops.append(lambda z: aug_noise_poisson(
            z, config.poisson_strength_min, config.poisson_strength_max))

    if np.random.rand() < config.p_axis:
        pre_ops.append(lambda z: aug_axis_warp(z, config))

    # 3. 基线相关增强，最多启用 1 项
    r = np.random.rand()
    if r < config.p_baseline_weak:
        pre_ops.append(lambda z: aug_weak_baseline(
            z,
            lin_min=config.baseline_lin_min,
            lin_max=config.baseline_lin_max,
            sin_min=config.baseline_sin_min,
            sin_max=config.baseline_sin_max,
            freq_min=config.baseline_freq_min,
            freq_max=config.baseline_freq_max))
    elif r < config.p_baseline_weak + config.p_baseline_strong:
        pre_ops.append(lambda z: aug_strong_baseline(
            z,
            amp_min=config.baseline_strong_amp_min,
            amp_max=config.baseline_strong_amp_max))

    np.random.shuffle(pre_ops)
    for fn in pre_ops[:config.max_pre_augs]:
        x = fn(x)

    if is_tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)
    else:
        return x.astype(np.float32)

def augment_norm_spectrum(x, config):
    """
    - training=True : 做增强
    - 模型输入用 标准化后数据
    """
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
            lambda z: aug_broadening(z, config.broad_sigma_min, config.broad_sigma_max, config.broad_truncate))

    if np.random.rand() < config.p_cut:
        post_ops.append(lambda z: aug_mask_attenuate(
            z,
            width_min=config.mask_width_min,
            width_max=config.mask_width_max,
            atten_min=config.mask_atten_min,
            atten_max=config.mask_atten_max,
        ))

    np.random.shuffle(post_ops)
    for fn in post_ops[:config.max_post_augs]:
        x = fn(x)

    if is_tensor:
        return torch.tensor(x, dtype=torch.float32, device=device)
    else:
        return x.astype(np.float32)

def _validate_sg_params(window_length, polyorder, deriv):
    """
    SG 参数硬校验，防止 GPU device-side assert
    """
    if window_length is None:
        raise ValueError("window_length is None")

    window_length = int(window_length)
    polyorder = int(polyorder)
    deriv = int(deriv)

    if window_length <= 0:
        raise ValueError(f"window_length must be > 0, got {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length must be > polyorder, "
            f"got window_length={window_length}, polyorder={polyorder}"
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(
            f"deriv must be in [0, polyorder], got deriv={deriv}, polyorder={polyorder}"
        )

    return window_length, polyorder, deriv

def sg_coeff(window_length, polyorder, deriv):
    """
    生成 Savitzky-Golay 核系数(numpy, CPU-only), 用于平滑与求导
    """
    window_length, polyorder, deriv = _validate_sg_params(
        window_length, polyorder, deriv
    )

    half = (window_length - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float32)

    A = np.vander(x, N=polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)

    coeff = pinv[deriv] * math.factorial(deriv)
    coeff = coeff.astype(np.float32)

    if not np.isfinite(coeff).all():
        raise ValueError(
            f"sg_coeff produced non-finite values: "
            f"window_length={window_length}, polyorder={polyorder}, deriv={deriv}"
        )

    return coeff

def sg_kernel(window_length, polyorder, deriv, device):
    """
    torch 封装版本（用于 GPU / 模型侧）
    """
    coeff = sg_coeff(window_length, polyorder, deriv)
    kernel = torch.tensor(coeff, dtype=torch.float32).to(device)
    return kernel.view(1, 1, -1)


def build_sg_kernels(config, device):
    """
    按当前配置一次性构造 smooth 和 d1 的 SG 卷积核。
    """
    return (
        sg_kernel(config.win_smooth, 3, 0, device),
        sg_kernel(config.win1, 3, 1, device),
    )


def build_pre_smooth_source(signal, config, sg_smooth):
    """为 smooth 通道构造“先平滑、后标准化”的输入源。"""
    smooth = F.conv1d(
        signal,
        sg_smooth,
        padding=config.win_smooth // 2,
    )
    return smooth[0, 0]


def build_pre_d1_source(signal, config, sg_smooth, sg_d1):
    """为 d1 通道构造“先平滑、后求导、再标准化”的输入源。"""
    smooth = F.conv1d(
        signal,
        sg_smooth,
        padding=config.win_smooth // 2,
    )
    d1 = F.conv1d(
        smooth,
        sg_d1,
        padding=config.win1 // 2,
    )[0, 0]
    return d1 / config.delta


def build_input_channels(
    signal,
    config,
    smooth_signal=None,
    raw_signal=None,
    d1_signal=None,
):
    """
    将已经准备好的各支路信号按配置堆叠成模型最终输入通道。
    """
    base = signal[0, 0]
    channels = [base]

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
            f"Channel mismatch: built {len(channels)} channels, "
            f"but config.in_channels={config.in_channels}."
        )

    return torch.stack(channels, dim=0)

def build_model_input(
    raw_intensity,
    config,
    sg_smooth,
    sg_d1,
    device,
    augment=False,
):
    """
    把单条原始强度光谱统一转换为模型输入张量 [C, L]。
    """
    x = np.asarray(raw_intensity, dtype=np.float32)

    if augment:
        x = augment_raw_spectrum(x, config)
    else:
        x = x.copy()

    raw_signal = None
    if getattr(config, "raw_use", False):
        raw_signal = torch.as_tensor(x, dtype=torch.float32, device=device).view(1, 1, -1)

    smooth_x = None
    if config.smooth_use:
        smooth_source = (
            raw_signal
            if raw_signal is not None
            else torch.as_tensor(x, dtype=torch.float32, device=device).view(1, 1, -1)
        )
        smooth_x = build_pre_smooth_source(smooth_source, config, sg_smooth)

    d1_x = None
    if config.d1_use:
        d1_source = (
            raw_signal
            if raw_signal is not None
            else torch.as_tensor(x, dtype=torch.float32, device=device).view(1, 1, -1)
        )
        d1_x = build_pre_d1_source(d1_source, config, sg_smooth, sg_d1)

    x = normalize_spectrum(x, config.norm_method)
    if smooth_x is not None:
        smooth_x = normalize_spectrum(smooth_x, config.norm_method)
    if d1_x is not None:
        d1_x = normalize_spectrum(d1_x, config.norm_method)

    if augment:
        x = augment_norm_spectrum(x, config)

    signal = torch.as_tensor(x, dtype=torch.float32, device=device).view(1, 1, -1)
    smooth_signal = None
    if smooth_x is not None:
        smooth_signal = smooth_x.view(1, 1, -1)
    d1_signal = None
    if d1_x is not None:
        d1_signal = d1_x.view(1, 1, -1)
    return build_input_channels(
        signal,
        config,
        smooth_signal=smooth_signal,
        raw_signal=raw_signal,
        d1_signal=d1_signal,
    )

class InputPreprocessor:
    """
    模型输入前的光谱预处理
    - 严格对齐某一次实验(config.yaml)
    - train / eval / predict 完全一致
    - 避免因手工预处理导致分布漂移
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # ------------------------------
        # SG kernels（一次性构造）
        # ------------------------------
        self.sg_smooth, self.sg_d1 = build_sg_kernels(config, device)

    def __call__(self, path):
        """
        对单个 .arc_data 文件进行预处理
        返回 shape: [1, C, L]
        """
        return self.preprocess_arc(path)

    def preprocess_arc(self, path):
        """
        对单个 .arc_data 文件进行预处理
        返回 shape: [1, C, L]
        """
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
