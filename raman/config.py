# WARNING:
# config.py is intended for TRAINING ONLY.
# Use config_io.load_experiment for eval / predict.

class Config:

    # ===================== 全局设置 =====================
    # 训练层级（评估/预测层级在 evalute/predict 脚本里单独设置）
    train_level = "level_1"
    # train_level = "level_2"
    # train_level = "level_3"
    train_per_parent = True  # 是否层级训练

    # 训练分割层级（默认按 leaf 分组，避免泄漏）
    split_level = "leaf"

    use_align_loss = True
    align_loss_weight = 0.01
    align_start = 20
    align_end = 50

    use_supcon_loss = True
    supcon_loss_weight = 0.03
    supcon_tau = 0.15
    supcon_start = 30
    supcon_end = 50
    supcon_level = "leaf"
    # 对齐/SupCon 后期衰减起点（占总 epoch 比例）
    # 直接手动设置具体值，例如 0.6 / 0.7
    decay_start_ratio = 0.7

    # ===================== 基础目录 =====================
    # 数据根目录
    # dataset_root = "dataset/厌氧菌"
    # dataset_root = "dataset/耐药菌"
    dataset_root = "dataset/细菌"
    cut_min = 600
    cut_max = 1800
    target_points = 896  # 插值参考点数
    delta = (cut_max - cut_min) / (target_points - 1)
    # BAD_BANDS = [(900, 940)] # 厌氧菌
    # BAD_BANDS = [(905, 940.0)] # 耐药菌
    BAD_BANDS = [(900, 950.0)] # 细菌
    bad_bands = BAD_BANDS

    # 输出目录（由 train 在运行期确定，绑定时间戳）
    timestamp = None
    output_dir = None
    seed = 88 # 分组随机种子
    deterministic = True  # 是否保持训练可复现（GPU 稳定性优先）
    early_stop_w_f1 = 0.6
    early_stop_w_acc = 0.4

    # ===================== 模型参数 =====================
    input_is_norm = False    # 输入数据是否已经标准化过
    norm_method = "snv"       # 选择标准化方式 snv/l2/minmax

    # 打开通道
    snv_posneg_split = True  # SNV pos/neg split
    smooth_use = True    # 是否使用smooth作为额外通道
    d1_use = False       # 是否使用一阶导作为额外通道

    @property
    def in_channels(self):
        n = 2 if self.snv_posneg_split else 1  # base channels
        if self.smooth_use:
            n += 1
        if self.d1_use:
            n += 1
        return n

    # SE 模块
    se_use = True
    reduction = 8
    # backbone_activation:
    # - "relu": 当前默认配置
    # - "silu": 更平滑，常用于小模型或希望保留弱负响应时
    backbone_activation = "relu"

    # Focal loss 强度
    gamma = 0.8 # 控制“压容易样本”的力度
    use_severity_weight = True  # 训练时是否启用严重程度感知重加权
    use_drw = True  # 是否启用动态类权重（EMA / DRW）
    label_smoothing = 0.0  # 分层掩码训练默认关闭 label smoothing，避免数值不稳定

    # SG 预处理窗口参数
    win_res = 15      # 残差窗口
    win_smooth = 15   # 平滑窗口
    win1 = 15   # 一阶导窗口

    # backbone_type:
    # - "cnn": ResNeXt1D 主干
    # - "identity": 跳过 CNN，只做平均下采样 + 1x1 通道投影
    backbone_type = "cnn"
    # identity 路径的时序下采样倍率；默认 16，用来和 CNN 主干的长度压缩量大致对齐
    # 1 表示不下采样
    identity_pool_kernel = 16

    # encoder_type: "transformer" | "lstm" | "none"
    encoder_type = "transformer"
    # 常用消融组合：
    # - 仅 CNN: backbone_type="cnn", encoder_type="none"
    # - 仅 LSTM: backbone_type="identity", encoder_type="lstm"
    # - 仅 Transformer: backbone_type="identity", encoder_type="transformer"
    # - CNN + LSTM: backbone_type="cnn", encoder_type="lstm"
    # - CNN + Transformer: backbone_type="cnn", encoder_type="transformer"
    # Transformer 配置
    transformer_nhead = 6  # 定死
    transformer_dim = 192  # 定死
    transformer_ffn_dim = 384  # 定死
    transformer_layers = 1 # 轻量
    transformer_dropout = 0.2  # 定死
    # LSTM 配置
    lstm_hidden = 192
    lstm_layers = 1
    lstm_dropout = 0.2
    lstm_bidirectional = False # 是否双向LSTM

    # Attention Pooling dropout（pooling_type="attn"时生效）
    att_pool_dropout = 0.2
    # pooling_type:
    # - "attn": 注意力池化（原版）
    # - "stat": 统计池化(mean+std)
    pooling_type = "stat"
    # cosine_head:
    # - True : 余弦分类头（特征/权重归一化，配合cosine_scale）
    # - False: 线性分类头
    cosine_head = True
    # 余弦分类头的缩放系数
    cosine_scale = 25

    # ===================== ResNeXt 参数 =====================
    # cardinality = 路数，分组卷积的并行分支数量
    cardinality = 4
    # base_width = 每条路径的宽度
    base_width = 4
    # Stem 卷积核大小（单尺度）
    stem_kernel_size = 15
    # stem_multiscale:
    # - True : 多尺度 stem（并联多卷积核后拼接）
    # - False: 单尺度 stem（使用 stem_kernel_size）
    stem_multiscale = True
    # 多尺度 stem 的卷积核列表
    stem_kernel_sizes = (3, 7, 15)

    # ===================== 训练相关 =====================
    # 训练轮数
    epochs = 80
    # 每 batch 的光谱数量
    batch_size = 64
    # Adam 学习率
    learning_rate = 4e-4
    # 训练集划分比例
    train_split = 0.8
    # EarlyStopping 容忍次数
    patience = 40
    # 是否使用 GPU（如果为 False 强制使用 CPU）
    use_gpu = True
    # CosineAnnealingLR 调度器
    scheduler_Tmax = int(epochs)
    scheduler_eta_min = 1e-5

    # ===================== 可视化/嵌入 =====================
    embedding_method = "tsne"  # "umap" | "tsne"
    umap_neighbors = 15
    umap_min_dist = 0.1
    tsne_perplexity = 30
    tsne_iter = 1000

    # ========= RAW 强度域（跨批次核心） =========
    # 噪声（两者互斥抽取）
    # p_noise：
    #   高斯加性噪声
    # p_poisson：
    #   强度相关噪声（近似泊松）
    #   避免同时叠加两类噪声导致非真实噪声分布
    p_noise = 0.4
    p_poisson = 0.2

    # baseline 扰动
    # p_baseline_weak：
    #   弱基线扰动
    # p_baseline_strong：
    #   强基线扰动（跨域级）
    p_baseline_weak = 0.5  # 同域残余，稳定训练
    p_baseline_strong = 0.3  # 跨域模拟，但低频

    # 频轴扰动（域级，但比 baseline 弱）
    p_axis = 0.2
    # 实际位移量 ≈ alpha * (x - mean_x)
    axis_warp_alpha = 0.002  # 建议 0.001 ~ 0.005
    # 非线性扰动幅度（单位：采样点）
    axis_warp_beta = 1.0  # 建议 0.5 ~ 2.0

    # 分段峰比例扰动（核心）
    p_piecewise_gain = 0.30
    piecewise_gain_std = 0.12

    # ========= SNV 后形状域 =========
    # ---------- 峰位平移 ----------
    p_shift = 0.3
    shift_max = 3

    # ---------- 峰展宽 ----------
    # broad_sigma_min/max：
    #   高斯核 sigma（单位：点）
    #   值越大，峰越宽
    # broad_truncate：
    #   高斯核截断范围（sigma 的倍数）
    #   控制卷积核长度，避免全谱模糊
    p_broadening = 0.35
    broad_sigma_min = 0.6
    broad_sigma_max = 1.2
    broad_truncate = 3.0

    # ---------- 局部衰减遮挡 ----------
    # mask_width_min/max：
    #   被衰减区域长度（单位：点）
    # mask_atten_min/max：
    #   衰减比例（0 表示完全抑制，0.3 表示保留 30%）
    p_cut = 0.3
    mask_width_min = 40
    mask_width_max = 100
    mask_atten_min = 0.1
    mask_atten_max = 0.3

    # 增强叠加数量控制
    max_pre_augs = 4
    max_post_augs = 2

    # ---------- 高斯噪声 ----------
    #   噪声相对幅度（相对于谱的有效振幅）
    #   避免强谱/弱谱噪声比例不一致
    noise_rel_min = 0.005
    noise_rel_max = 0.02

    # ---------- 强度相关噪声（泊松型） ----------
    #   控制噪声与信号幅度的耦合强度
    #   用于模拟光子统计噪声
    poisson_strength_min = 0.0
    poisson_strength_max = 0.015

    # ---------- 残余基线扰动 ----------
    # baseline_lin_min/max：
    #   线性基线扰动强度（相对于谱振幅）
    # baseline_sin_min/max：
    #   正弦基线扰动强度（相对于谱振幅）
    # baseline_freq_min/max：
    #   正弦基线频率（越小越“慢”，越像荧光背景）
    baseline_lin_min = 0.0
    baseline_lin_max = 0.02
    baseline_sin_min = 0.0
    baseline_sin_max = 0.01
    baseline_freq_min = 0.5
    baseline_freq_max = 2.0

    baseline_strong_amp_min = 0.05
    baseline_strong_amp_max = 0.15

config = Config()
