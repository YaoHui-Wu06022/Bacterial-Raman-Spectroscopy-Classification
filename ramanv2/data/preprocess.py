"""离线数据清洗与绘图共用的预处理参数。"""

from __future__ import annotations

from ramanv2.core.config import InputConfig
from ramanv2.data.config import DataBuildConfig, build_cosmic_ray_options


def build_preprocess_kwargs(
    profile_id: str,
    input_config: InputConfig,
    build_config: DataBuildConfig,
    reference_axis,
) -> dict:
    """构造 preprocess_single_spectrum 的统一参数，避免各入口漏配字段。"""
    return {
        "cut_min": input_config.cut_min,
        "cut_max": input_config.cut_max,
        "reference_wavenumbers": reference_axis,
        "bad_bands": input_config.bad_bands,
        "baseline_lam": build_config.baseline_lam,
        "baseline_asls_p": build_config.baseline_asls_p,
        "baseline_max_iter": build_config.baseline_max_iter,
        "baseline_fit_min": build_config.baseline_fit_min,
        "baseline_fit_max": build_config.baseline_fit_max,
        "baseline_method": build_config.baseline_method,
        **build_cosmic_ray_options(profile_id, build_config),
    }
