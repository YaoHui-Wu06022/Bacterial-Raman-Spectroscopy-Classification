from dataclasses import dataclass
from pathlib import Path

from raman.data.build import COMMON_BAD_BANDS, DEFAULT_PIPELINE_CONFIG
from raman.data.profiles import get_profile


# ==================== 配置文件字段分组 ====================
# shared_config.yaml：同一实验根内必须一致，主要决定输入张量如何生成。
SHARED_CONFIG_FIELDS = (
    "dataset_name",
    "cut_min",
    "cut_max",
    "target_points",
    "bad_bands",
    "norm_method",
    "smooth_use",
    "d1_use",
    "win_smooth",
    "win1",
    "p_piecewise_gain",
    "p_noise",
    "p_axis",
    "p_baseline_weak",
    "p_baseline_strong",
    "p_shift",
    "p_broadening",
    "p_cut",
    "max_pre_augs",
    "max_post_augs",
)

# model_config.yaml：每个 level/parent 的单次训练 run 可以独立调整。
MODEL_CONFIG_FIELDS = (
    "current_train_level",
    "train_only_parent",
    "train_only_parent_name",
    "train_filter_level",
    "train_filter_value",
    "train_per_parent",
    "split_by_source_prefix",
    "train_split",
    "seed",
    "epochs",
    "patience",
    "batch_size",
    "train_loader_num_workers",
    "val_loader_num_workers",
    "loader_pin_memory",
    "loader_persistent_workers",
    "loader_prefetch_factor",
    "learning_rate",
    "grad_clip_norm",
    "scheduler_Tmax",
    "scheduler_eta_min",
    "use_align_loss",
    "align_loss_weight",
    "align_start",
    "align_end",
    "use_supcon_loss",
    "supcon_loss_weight",
    "supcon_tau",
    "supcon_start",
    "supcon_end",
    "decay_start_ratio",
    "early_stop_w_f1",
    "early_stop_w_acc",
    "se_use",
    "reduction",
    "backbone_activation_negative_slope",
    "gamma",
    "use_severity_weight",
    "use_ema",
    "backbone_type",
    "cnn_block_type",
    "cardinality",
    "base_width",
    "resnet_bottleneck_ratio",
    "identity_pool_kernel",
    "encoder_type",
    "transformer_nhead",
    "transformer_dim",
    "transformer_ffn_dim",
    "transformer_layers",
    "transformer_dropout",
    "lstm_hidden",
    "lstm_layers",
    "lstm_dropout",
    "lstm_bidirectional",
    "pooling_type",
    "cosine_head",
    "cosine_scale",
    "stem_kernel_sizes",
    "umap_neighbors",
    "umap_min_dist",
)

# 运行期字段：用于当前进程定位输出目录、续训和设备，不写入 model_config.yaml。
RUNTIME_CONFIG_FIELDS = (
    "timestamp",
    "output_dir",
    "experiment_dir",
    "run_dir",
    "resume_training",
    "checkpoint_interval",
    "use_gpu",
    "deterministic",
)

# 推理/评估加载模型时必须和实验根保持一致的输入字段。
INPUT_COMPAT_FIELDS = (
    "dataset_name",
    "cut_min",
    "cut_max",
    "target_points",
    "bad_bands",
    "norm_method",
    "smooth_use",
    "d1_use",
    "win_smooth",
    "win1",
)


@dataclass
class SharedInputConfig:
    """同一实验根内必须保持一致的输入、预处理和增强配置。"""

    # 数据集
    dataset_name: str = "GN"

    # 输入波数范围和重采样点数
    cut_min: float = float(DEFAULT_PIPELINE_CONFIG.cut_min)
    cut_max: float = float(DEFAULT_PIPELINE_CONFIG.cut_max)
    target_points: int = int(DEFAULT_PIPELINE_CONFIG.target_points)
    bad_bands: list = None

    # 标准化方式和输入通道
    norm_method: str = "snv"
    smooth_use: bool = True
    d1_use: bool = False

    # SG 平滑和一阶导窗口
    win_smooth: int = 15
    win1: int = 15

    # 标准化前增强概率
    p_piecewise_gain: float = 0.40
    p_noise: float = 0.70
    p_axis: float = 0.30
    p_baseline_weak: float = 0.55
    p_baseline_strong: float = 0.35

    # 标准化后增强概率
    p_shift: float = 0.40
    p_broadening: float = 0.45
    p_cut: float = 0.20

    # 单条光谱最多叠加的增强数量
    max_pre_augs: int = 4
    max_post_augs: int = 2

    def __post_init__(self):
        # dataclass 默认值不能直接放可变 list，这里运行期补默认坏段。
        if self.bad_bands is None:
            self.bad_bands = [tuple(band) for band in COMMON_BAD_BANDS]


@dataclass
class ModelRunConfig:
    """单个层级或 parent 子模型可独立调整的训练与模型配置。"""

    # 训练任务范围
    current_train_level: str | None = None
    train_only_parent: int | None = None
    train_only_parent_name: str | None = None
    train_filter_level: str | None = None
    train_filter_value: object | None = None
    train_per_parent: bool = True

    # 训练/验证划分
    split_by_source_prefix: bool = False
    train_split: float = 0.8
    seed: int = 42

    # 训练控制
    epochs: int = 80
    patience: int = 30
    batch_size: int = 64

    # DataLoader 参数
    train_loader_num_workers: int = 2
    val_loader_num_workers: int = 2
    loader_pin_memory: bool = True
    loader_persistent_workers: bool = True
    loader_prefetch_factor: int = 2

    # 优化器和调度器
    learning_rate: float = 4e-4
    grad_clip_norm: float = 5.0
    scheduler_Tmax: int | None = None
    scheduler_eta_min: float = 1e-5

    # Align Loss
    use_align_loss: bool = True
    align_loss_weight: float = 0.01
    align_start: int = 20
    align_end: int = 50

    # SupCon Loss
    use_supcon_loss: bool = True
    supcon_loss_weight: float = 0.03
    supcon_tau: float = 0.15
    supcon_start: int = 30
    supcon_end: int = 50

    # 对齐/SupCon 后期衰减起点，占总 epoch 的比例
    decay_start_ratio: float = 0.7

    # Early stopping 打分权重
    early_stop_w_f1: float = 0.6
    early_stop_w_acc: float = 0.4

    # SE 模块和损失重加权
    se_use: bool = True
    reduction: int = 8
    backbone_activation_negative_slope: float = 0.05
    gamma: float = 0.8
    use_severity_weight: bool = True
    use_ema: bool = True

    # CNN 主干结构
    backbone_type: str = "cnn"
    cnn_block_type: str = "resnext"
    cardinality: int = 4
    base_width: int = 4
    resnet_bottleneck_ratio: int = 4
    identity_pool_kernel: int = 8

    # 时序编码器
    encoder_type: str = "transformer"
    transformer_nhead: int = 8
    transformer_dim: int = 256
    transformer_ffn_dim: int = 512
    transformer_layers: int = 2
    transformer_dropout: float = 0.2
    lstm_hidden: int = 192
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = False

    # 池化和分类头
    pooling_type: str = "attn"
    cosine_head: bool = False
    cosine_scale: int = 25

    # 多尺度 stem
    stem_kernel_sizes: tuple = (3, 7, 15)

    # UMAP 可视化
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1

    def __post_init__(self):
        # 未显式指定时，scheduler_Tmax 跟随 epochs，方便 notebook 只改训练轮次。
        if self.scheduler_Tmax is None:
            self.scheduler_Tmax = int(self.epochs)


@dataclass
class RuntimeConfig:
    """运行期字段，不作为模型结构或共享输入的一部分。"""

    # 当前实验和 run 的定位信息
    timestamp: str | None = None
    dataset_root_override: str | None = None
    output_dir: str | None = None
    experiment_dir: str | None = None
    run_dir: str | None = None

    # 续训、保存和设备控制
    resume_training: bool = True
    checkpoint_interval: int = 20
    use_gpu: bool = True
    deterministic: bool = True


class Config:
    """组合配置对象，同时保持旧代码使用的扁平属性访问方式。"""

    _GROUPS = ("shared", "model", "runtime")

    def __init__(self, shared=None, model=None, runtime=None):
        # 三组配置内部保存；外部仍可像原有代码一样用 config.xxx 访问。
        object.__setattr__(self, "shared", shared or SharedInputConfig())
        object.__setattr__(self, "model", model or ModelRunConfig())
        object.__setattr__(self, "runtime", runtime or RuntimeConfig())

    def __getattr__(self, name):
        # 兼容旧代码：config.batch_size 会自动落到 model.batch_size。
        for group_name in self._GROUPS:
            group = object.__getattribute__(self, group_name)
            if hasattr(group, name):
                return getattr(group, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        # 兼容旧代码：给扁平属性赋值时写回对应配置组。
        for group_name in self._GROUPS:
            group = object.__getattribute__(self, group_name)
            if hasattr(group, name):
                setattr(group, name, value)
                if name == "epochs":
                    self._sync_scheduler_tmax()
                return
        object.__setattr__(self, name, value)

    @property
    def dataset_root(self):
        # eval 会把 dataset_root 对齐到具体阶段目录；平时按 dataset_name 自动推导。
        if self.runtime.dataset_root_override:
            return self.runtime.dataset_root_override
        profile = get_profile(self.dataset_name)
        return str(Path("dataset") / profile.dataset_name)

    @dataset_root.setter
    def dataset_root(self, value):
        self.runtime.dataset_root_override = str(value) if value is not None else None

    @property
    def delta(self):
        # 相邻两个重采样点对应的大致波数间隔。
        return (float(self.cut_max) - float(self.cut_min)) / (int(self.target_points) - 1)

    @property
    def in_channels(self):
        # 原始标准化谱固定 1 通道，smooth/d1 按开关额外叠加。
        channels = 1
        if self.smooth_use:
            channels += 1
        if self.d1_use:
            channels += 1
        return channels

    def _sync_scheduler_tmax(self):
        # notebook 或脚本改 epochs 时，同步默认调度周期。
        self.model.scheduler_Tmax = int(self.model.epochs)

    def to_shared_dict(self):
        """导出实验根 shared_config.yaml 所需字段。"""
        return _select_existing(self, SHARED_CONFIG_FIELDS)

    def to_model_dict(self):
        """导出单次 run 的 model_config.yaml 所需字段。"""
        return _select_existing(self, MODEL_CONFIG_FIELDS)

    def to_runtime_dict(self):
        """导出运行期字段，主要用于 resolved_config.yaml 快照。"""
        return _select_existing(self, RUNTIME_CONFIG_FIELDS)

    def to_dict(self):
        """导出完整扁平配置，供 resolved_config.yaml 复现和排查。"""
        data = {}
        data.update(self.to_shared_dict())
        data.update(self.to_model_dict())
        data.update(self.to_runtime_dict())
        data["dataset_root"] = self.dataset_root
        data["delta"] = self.delta
        data["in_channels"] = self.in_channels
        return data

    @classmethod
    def from_dict(cls, data):
        """从 yaml 字典恢复配置对象。"""
        cfg = cls()
        for key, value in (data or {}).items():
            if key in {"dataset_root", "delta", "in_channels"}:
                continue
            setattr(cfg, key, value)
        if cfg.scheduler_Tmax is None:
            cfg.scheduler_Tmax = int(cfg.epochs)
        return cfg


def _select_existing(config_obj, fields):
    """按字段列表从配置对象中取值，并把 tuple 转成 yaml 更友好的 list。"""
    data = {}
    for key in fields:
        try:
            value = getattr(config_obj, key)
        except AttributeError:
            continue
        if isinstance(value, tuple):
            value = list(value)
        data[key] = value
    return data


def make_config(shared=None, model=None, runtime=None):
    """按 shared -> model -> runtime 的顺序创建临时配置对象。"""
    cfg = Config()
    for payload in (shared or {}, model or {}, runtime or {}):
        for key, value in payload.items():
            setattr(cfg, key, value)
    if cfg.scheduler_Tmax is None:
        cfg.scheduler_Tmax = int(cfg.epochs)
    return cfg


config = Config()
