"""离线拉曼预处理的共享配置。"""

from dataclasses import dataclass

from raman.tool.spectrum import build_wn_ref


CUT_MIN = 600
CUT_MAX = 1800
COMMON_BAD_BANDS = ((890.0, 950.0),)
TARGET_POINTS = 896

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

    def build_wn_ref(self):
        """根据裁剪范围和目标点数生成统一插值坐标。"""
        return build_wn_ref(self.cut_min, self.cut_max, self.target_points)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()


def resolve_pipeline_config(pipeline_config=None):
    """返回调用方给定的配置，缺失时使用默认配置。"""
    return pipeline_config or DEFAULT_PIPELINE_CONFIG
