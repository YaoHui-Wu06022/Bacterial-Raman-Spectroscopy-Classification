from pathlib import Path

from dataset_process.pipeline import DEFAULT_PIPELINE_CONFIG
from dataset_process.profiles import COMMON_BAD_BANDS, get_profile


class Config:
    # 全局设置
    train_per_parent = True  # 是否按父类分别训练子模型

    # 训练切分层级（默认按 leaf 分组，避免泄漏；leaf 仅用于切分）
    split_level = "leaf"

    # 对齐损失
    use_align_loss = True
    align_loss_weight = 0.01
    align_start = 20
    align_end = 50

    # SupCon 损失
    use_supcon_loss = True
    supcon_loss_weight = 0.03
    supcon_tau = 0.15
    supcon_start = 30
    supcon_end = 50
    # 对齐/SupCon 后期衰减起点（占总 epoch 比例）
    decay_start_ratio = 0.7

    # 数据集设置
    # 只需要改这里的数据集名称，预处理设置来源于dataset_process
    dataset_name = "细菌"

    @property
    def dataset_root(self):
        profile = get_profile(self.dataset_name)
        return str(Path("dataset") / profile.dataset_name)

    @property
    def cut_min(self):
        return float(DEFAULT_PIPELINE_CONFIG.cut_min)

    @property
    def cut_max(self):
        return float(DEFAULT_PIPELINE_CONFIG.cut_max)

    @property
    def target_points(self):
        return int(DEFAULT_PIPELINE_CONFIG.target_points)

    @property
    def delta(self):
        return (self.cut_max - self.cut_min) / (self.target_points - 1)

    @property
    def bad_bands(self):
        return [tuple(band) for band in COMMON_BAD_BANDS]

    # 输出目录（由 train 在运行期确定，绑定时间戳）
    timestamp = None
    output_dir = None
    seed = 42  # 分组随机种子
    deterministic = True  # 是否保持训练可复现（GPU 稳定性优先）
    early_stop_w_f1 = 0.6
    early_stop_w_acc = 0.4

    # 模型参数
    norm_method = "snv"  # 选择标准化方式 snv/l2/minmax

    # 输入通道
    smooth_use = True  # 是否使用 smooth 作为额外通道
    raw_use = False  # 是否保留 raw 增强后、未标准化的输入通道
    d1_use = False  # 是否使用一阶导作为额外通道

    @property
    def in_channels(self):
        channels = 1
        if self.smooth_use:
            channels += 1
        if self.raw_use:
            channels += 1
        if self.d1_use:
            channels += 1
        return channels

    # SE 模块
    se_use = True
    reduction = 8
    # 激活函数设置
    backbone_activation_negative_slope = 0.05

    # Focal loss 强度
    gamma = 0.8  # 控制“压容易样本”的力度
    use_severity_weight = True  # 是否启用严重程度感知重加权
    use_drw = True  # 是否启用动态类权重
    # SG 预处理窗口参数
    win_smooth = 15
    win1 = 15

    # backbone_type:
    # - "cnn": 使用卷积主干
    # - "identity": 跳过 CNN，只做平均下采样 + 1x1 通道投影
    backbone_type = "cnn"
    # cnn_block_type:
    # - "resnext": 当前默认配置
    # - "resnet": 共用同一套 bottleneck 骨架，但中间卷积改成普通卷积
    cnn_block_type = "resnext"
    # ResNeXt 参数
    cardinality = 4
    base_width = 4
    # ResNet 模式下 bottleneck 中间通道缩放比例
    resnet_bottleneck_ratio = 4
    # identity 路径的时序下采样倍率；1 表示不下采样
    identity_pool_kernel = 8

    # encoder_type: "transformer" | "lstm" | "none"
    encoder_type = "transformer"
    # transformer
    transformer_nhead = 6
    transformer_dim = 192
    transformer_ffn_dim = 384
    transformer_layers = 1
    transformer_dropout = 0.2
    # lstm
    lstm_hidden = 192
    lstm_layers = 1
    lstm_dropout = 0.2
    lstm_bidirectional = False

    # 池化
    # - "attn": 注意力池化
    # - "stat": 统计池化(mean+std)
    pooling_type = "stat"
    # 分类头
    # - True : 余弦分类头
    # - False: 线性分类头
    cosine_head = True
    cosine_scale = 25

    # stem_kernel_sizes:
    # - (15,)      : 自动退化为单尺度 stem
    # - (3, 7, 15) : 多尺度 stem
    stem_kernel_sizes = (3, 7, 15)

    # 训练相关
    epochs = 80
    batch_size = 64
    train_loader_num_workers = 4
    eval_loader_num_workers = 4
    loader_pin_memory = True
    loader_persistent_workers = True
    loader_prefetch_factor = 2
    learning_rate = 4e-4
    train_split = 0.8
    patience = 40
    use_gpu = True
    scheduler_Tmax = int(epochs)
    scheduler_eta_min = 1e-5

    # 可视化与嵌入
    embedding_method = "tsne"  # "umap" | "tsne"
    umap_neighbors = 15
    umap_min_dist = 0.1
    tsne_perplexity = 30
    tsne_iter = 1000

    # RAW 域增强概率
    p_piecewise_gain = 0.30
    p_noise = 0.60
    p_axis = 0.20
    p_baseline_weak = 0.50
    p_baseline_strong = 0.30

    # 标准化后增强概率
    p_shift = 0.30
    p_broadening = 0.35
    p_cut = 0.30

    # 增强叠加数量上限
    max_pre_augs = 4
    max_post_augs = 2

    # 分段峰强比例扰动
    piecewise_gain_std = 0.12

    # 强度相关高斯噪声，sigma = a + b * |x|
    noise_base_rel_min = 0.005
    noise_base_rel_max = 0.02
    noise_slope_rel_min = 0.0
    noise_slope_rel_max = 0.015

    # 波数轴扰动
    axis_warp_alpha = 0.002
    axis_warp_beta = 1.0

    # 弱 baseline 扰动
    baseline_lin_min = 0.0
    baseline_lin_max = 0.02
    baseline_sin_min = 0.0
    baseline_sin_max = 0.01
    baseline_freq_min = 0.5
    baseline_freq_max = 2.0

    # 强 baseline 扰动
    baseline_strong_amp_min = 0.05
    baseline_strong_amp_max = 0.15

    # 标准化后弱形状扰动
    shift_max = 3
    broad_sigma_min = 0.6
    broad_sigma_max = 1.2
    broad_truncate = 3.0

    # 局部衰减遮挡
    mask_width_min = 40
    mask_width_max = 100
    mask_atten_min = 0.1
    mask_atten_max = 0.3


config = Config()
