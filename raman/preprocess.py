def _PREPROCESS_PIPELINE_DOC():
    # 数据增强参数（增强概率）
    # 设计总原则：
    # 1）增强严格区分 RAW 强度空间 与 SNV 后形状空间
    # 2）所有增强均具有明确的仪器 / 物理来源，不引入非物理畸变
    # 3）增强只改变“观测条件”，不改变“物质类别语义”
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
    - 仅对有效波段做统计
    - NaN 位置保持 NaN（代表不可用物理区域）
    """
    if isinstance(x, torch.Tensor):
        mask = ~torch.isnan(x)
        mean = x[mask].mean()
        std = x[mask].std(unbiased=False)
        return (x - mean) / (std + EPS)
    else:
        x = np.asarray(x, dtype=np.float32)
        mean = np.nanmean(x)
        std  = np.nanstd(x)
        return (x - mean) / (std + EPS)

def L2Normalize(x, eps=1e-8):
    """
    L2 标准化（逐样本）
    - 消除整体强度尺度
    - 保留谱形方向
    - 不强制零均值
    - NaN-safe
    """
    if isinstance(x, torch.Tensor):
        mask = torch.isfinite(x)
        if not mask.any():
            return x

        norm = torch.sqrt(torch.sum(x[mask] ** 2)).clamp_min(eps)
        out = x.clone()
        out[mask] = out[mask] / norm
        return out
    else:
        x = np.asarray(x, dtype=np.float32)
        mask = np.isfinite(x)
        if not mask.any():
            return x

        norm = np.sqrt(np.sum(x[mask] ** 2))
        norm = max(norm, eps)

        out = x.copy()
        out[mask] = out[mask] / norm
        return out

def MinMaxNormalize(x, eps=1e-8):
    """
    Min-Max 归一化（逐样本 / per-spectrum）
    作用：
    - 将单条光谱缩放到 [0, 1]
    - 不引入跨样本统计
    - NaN-safe
    """
    if isinstance(x, torch.Tensor):
        mask = torch.isfinite(x)
        if not mask.any():
            return x

        min_v = torch.min(x[mask])
        max_v = torch.max(x[mask])
        denom = (max_v - min_v).clamp_min(eps)

        out = x.clone()
        out[mask] = (out[mask] - min_v) / denom
        return out
    else:
        x = np.asarray(x, dtype=np.float32)
        mask = np.isfinite(x)
        if not mask.any():
            return x

        min_v = np.min(x[mask])
        max_v = np.max(x[mask])
        denom = max(max_v - min_v, eps)

        out = x.copy()
        out[mask] = (out[mask] - min_v) / denom
        return out

def _robust_amp(x):
    """
    估计光谱的稳健振幅范围（robust amplitude）
    - 只使用 finite 区域
    - NaN 表示物理无效波段，不参与统计
    """
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    p1, p99 = np.percentile(x[valid], [1, 99])
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

# =========================
# Pre-NORM / RAW 空间增强（强度空间）
# - 该类增强发生在 NORM 之前（或 RAW 输入空间）
# - 主要模拟仪器与采集条件变化
# - 对 RAW 输入影响最直接，对 NORM 输入仍可改变相对形态
# =========================
def aug_piecewise_gain(x, segments, gain_std=0.15):
    """
    分段幅度扰动（NaN-safe）
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
    高斯噪声增强（NaN-safe）
    噪声幅度与光谱有效振幅相关，
    避免强谱 / 弱谱噪声比例失衡
    峰形：轻微抖动
    基线：随机抖动
    """
    # 相对噪声：sigma = rel * robust_amp
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)

    amp = _robust_amp(x)
    rel = np.random.uniform(rel_min, rel_max)
    sigma = rel * amp

    noise = np.zeros_like(x)
    noise[mask] = sigma * np.random.randn(mask.sum()).astype(np.float32)

    out = x.copy()
    out[mask] = out[mask] + noise[mask]
    return out

def aug_noise_poisson(x, strength_min=0.0, strength_max=0.015):
    """
    强度相关噪声（NaN-safe）
    噪声幅度与 |x| 相关
    """
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)

    amp = _robust_amp(x)
    k = np.random.uniform(strength_min, strength_max) * amp
    max_abs = np.max(np.abs(x[mask])) + EPS
    sigma = np.zeros_like(x)
    # sqrt(|x|) 型噪声（光子统计的味道）
    sigma[mask] = k * np.sqrt(np.abs(x[mask]) / max_abs)
    noise = np.zeros_like(x)
    noise[mask] = sigma[mask] * np.random.randn(mask.sum()).astype(np.float32)

    out = x.copy()
    out[mask] = out[mask] + noise[mask]
    return out

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
    mask = np.isfinite(x)
    amp = _robust_amp(x)
    L = len(x)
    t = np.linspace(-0.5, 0.5, L, dtype=np.float32)

    a = np.random.uniform(lin_min, lin_max) * amp
    b = np.random.uniform(sin_min, sin_max) * amp
    f = np.random.uniform(freq_min, freq_max)
    phi = np.random.uniform(0, 2*np.pi)

    baseline = a * t + b * np.sin(2*np.pi*f*t + phi).astype(np.float32)

    out = x.copy()
    out[mask] = out[mask] + baseline[mask]
    return out

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
    mask = np.isfinite(x)
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

    out = x.copy()
    out[mask] = out[mask] + baseline[mask]
    return out

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

# =========================
# Post-NORM / 形状空间增强（峰位置/形状）
# - 该类增强发生在 NORM 之后
# - 主要影响峰位置与峰形
# - 不改变整体强度统计
# =========================
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
    小核高斯展宽（NaN-safe）
    设计约束：
    - 仅局部展宽峰形
    - 避免全谱模糊导致物理语义丢失
    """
    x = np.asarray(x, dtype=np.float32)
    sigma = float(np.random.uniform(sigma_min, sigma_max))
    k = _gauss_kernel1d(sigma, truncate=truncate)

    mask = np.isfinite(x).astype(np.float32)
    x0 = np.nan_to_num(x, nan=0.0).astype(np.float32)
    pad = len(k) // 2
    x_pad = np.pad(x0, (pad, pad), mode="reflect")
    m_pad = np.pad(mask, (pad, pad), mode="reflect")

    num = np.convolve(x_pad, k, mode="valid").astype(np.float32)
    den = np.convolve(m_pad, k, mode="valid").astype(np.float32)

    y = num / (den + EPS)

    # 恢复坏段：原来是 NaN 的位置仍保持 NaN（不制造信息）
    y[mask == 0] = np.nan
    return y

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

# =========================
# RAW数据增强
# =========================
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

    # =====================================================
    # 1. Cross-domain augment
    # =====================================================
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


    # =====================================================
    # 2. Noise augment（最多 1 个）
    # =====================================================
    r = np.random.rand()
    if r < config.p_noise:
        pre_ops.append(lambda z: aug_noise_gaussian(
            z, config.noise_rel_min, config.noise_rel_max))
    elif r < config.p_noise + config.p_poisson:
        pre_ops.append(lambda z: aug_noise_poisson(
            z, config.poisson_strength_min, config.poisson_strength_max))

    if np.random.rand() < config.p_axis:
        pre_ops.append(lambda z: aug_axis_warp(z, config))

    # =====================================================
    # 3. Intra-domain augment（最多 1 个）
    # =====================================================
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

# =========================
# 标准化数据增强
# =========================
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
        self.sg_smooth = sg_kernel(config.win_smooth, 3, 0, device)
        self.sg_d1 = sg_kernel(config.win1, 3, 1, device)

    def __call__(self, path):
        """
        对单个 .arc_data 文件进行预处理
        返回 shape: [1, C, L]
        """
        return self.preprocess_arc(path)

    # ======================================================
    # Core API
    # ======================================================
    def preprocess_arc(self, path):
        """
        对单个 .arc_data 文件进行预处理
        返回 shape: [1, C, L]
        """

        data = np.loadtxt(path).astype(np.float32)
        raw_intensity = data[:, 1]

        x = raw_intensity.copy()

        # 如果没标准化过处理一下
        if not self.config.input_is_norm:
            if self.config.norm_method == "snv":
                x = SNV(x)
            elif self.config.norm_method == "l2":
                x = L2Normalize(x)
            elif self.config.norm_method == "minmax":
                x = MinMaxNormalize(x)
            else:
                raise ValueError(
                    f"Unknown norm_method: {self.config.norm_method}"
                )

        # 构造 Tensor
        signal = torch.tensor(
            x, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)

        # 通道构建
        base = signal[0, 0]
        if getattr(self.config, "snv_posneg_split", False):
            pos = torch.clamp(base, min=0.0)
            neg = torch.clamp(-base, min=0.0)
            channels = [pos, neg]
        else:
            channels = [base]

        # -----------------------------
        # smooth 通道
        # -----------------------------
        if self.config.smooth_use:
            smooth = F.conv1d(
                signal,
                self.sg_smooth,
                padding=self.config.win_smooth // 2
            )[0, 0]
            channels.append(smooth)


        # -----------------------------
        # 一阶导数
        # -----------------------------
        if self.config.d1_use:
            d1 = F.conv1d(
                signal,
                self.sg_d1,
                padding=self.config.win1 // 2
            )[0, 0]

            d1 = d1 / self.config.delta
            # ---- 弱 scale 对齐 ----
            scale = torch.max(torch.abs(d1)).clamp_min(1e-8)
            d1 = d1 / scale
            channels.append(d1)


        # -----------------------------
        # 最终检查 & stack
        # -----------------------------
        if len(channels) != self.config.in_channels:
            raise ValueError(
                f"Channel mismatch: built {len(channels)} channels, "
                f"but config.in_channels={self.config.in_channels}."
            )

        X = torch.stack(channels, dim=0)

        return X.unsqueeze(0)
