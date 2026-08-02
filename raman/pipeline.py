"""离线拉曼预处理的共享配置。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman.tool.spectrum import build_wn_ref


CUT_MIN = 600
CUT_MAX = 1800
COMMON_BAD_BANDS = ((890.0, 950.0),)
TARGET_POINTS = 896
INPUT_GRID_STANDARD = "standard"
INPUT_GRID_STANFORD_TRANSFER = "stanford_transfer"

BASELINE_METHOD = "airPLS"
BASELINE_LAM = 1e5
BASELINE_ASLS_P = 0.01
BASELINE_MAX_ITER = 15
BASELINE_FIT_MIN = 400
BASELINE_FIT_MAX = 2000

COSMIC_RAY_ENABLED_PROFILE_IDS = ("original",)
COSMIC_RAY_WINDOW_POINTS = 7
COSMIC_RAY_THRESHOLD = 7.0
COSMIC_RAY_MAX_ITER = 2

MIN_SAMPLES_PER_CLASS = 8

PCA_ENABLED = False
PCA_COMPONENTS = 0.95
PCA_CENTER = True
PCA_OUTLIER_RATIO = 0.01


@dataclass(frozen=True)
class PipelineConfig:
    """集中管理离线预处理阶段的固定参数。"""

    cut_min: float = CUT_MIN
    cut_max: float = CUT_MAX
    target_points: int = TARGET_POINTS
    input_grid_mode: str = INPUT_GRID_STANDARD
    bad_bands: tuple[tuple[float, float], ...] = COMMON_BAD_BANDS
    baseline_method: str = BASELINE_METHOD
    baseline_lam: float = BASELINE_LAM
    baseline_asls_p: float = BASELINE_ASLS_P
    baseline_max_iter: int = BASELINE_MAX_ITER
    baseline_fit_min: float = BASELINE_FIT_MIN
    baseline_fit_max: float = BASELINE_FIT_MAX
    cosmic_ray_enabled_profile_ids: tuple[str, ...] = COSMIC_RAY_ENABLED_PROFILE_IDS
    cosmic_ray_window_points: int = COSMIC_RAY_WINDOW_POINTS
    cosmic_ray_threshold: float = COSMIC_RAY_THRESHOLD
    cosmic_ray_max_iter: int = COSMIC_RAY_MAX_ITER
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
    pca_enabled: bool = PCA_ENABLED
    pca_components: float | int = PCA_COMPONENTS
    pca_center: bool = PCA_CENTER
    pca_outlier_ratio: float = PCA_OUTLIER_RATIO

    def __post_init__(self):
        """两个输入网格始终使用同一段固定坏区。"""
        normalized_bands = tuple(
            (float(start), float(end)) for start, end in self.bad_bands
        )
        if normalized_bands != COMMON_BAD_BANDS:
            raise ValueError(f"坏区固定为 {COMMON_BAD_BANDS}，不支持按网格模式分别设置")
        object.__setattr__(self, "bad_bands", COMMON_BAD_BANDS)

    def build_wn_ref(self):
        """按输入网格模式生成统一波数轴。"""
        if self.input_grid_mode == INPUT_GRID_STANFORD_TRANSFER:
            path = Path(__file__).resolve().parents[1] / "dataset" / "Stanforddataset" / "reference_wavenumbers.npy"
            axis = np.load(path)
            if axis.ndim != 1 or axis.size != self.target_points:
                raise ValueError(f"共享波数轴长度不匹配：{path}")
            if not (np.isfinite(axis).all() and np.all(np.diff(axis) > 0)):
                raise ValueError(f"共享波数轴必须为有限升序序列：{path}")
            return axis.astype(np.float64, copy=False)
        if self.input_grid_mode != INPUT_GRID_STANDARD:
            raise ValueError(f"未知输入网格模式：{self.input_grid_mode}")
        return build_wn_ref(self.cut_min, self.cut_max, self.target_points)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()


def resolve_pipeline_config(pipeline_config=None):
    """返回调用方给定的配置，缺失时使用默认配置。"""
    return pipeline_config or DEFAULT_PIPELINE_CONFIG
