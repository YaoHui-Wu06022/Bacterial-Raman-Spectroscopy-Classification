from dataclasses import replace

import numpy as np
import torch

from ramanv2.core.config import InputConfig
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.data.runtime.input import build_model_input, build_sg_kernels


def test_input_spec_splits_segments_at_bad_bands() -> None:
    spec = build_input_spec(InputConfig())
    assert spec.point_count == 851
    assert spec.segment_bounds == ((0, 217), (217, 851))

    whole = build_input_spec(InputConfig(bad_bands=()))
    assert whole.point_count == 896
    assert whole.segment_bounds == ((0, 896),)


def _segment_spec(d1_enable: bool) -> InputSpec:
    return InputSpec(
        in_channels=3 if d1_enable else 2,
        point_count=851,
        norm_method="none",
        smooth_enable=True,
        d1_enable=d1_enable,
        smooth_window=15,
        d1_window=15,
        step_cm=(1800.0 - 600.0) / 895,
        segment_bounds=((0, 217), (217, 851)),
    )


def test_smoothing_does_not_cross_bad_band_seam() -> None:
    spec = _segment_spec(d1_enable=False)
    smooth_kernel, d1_kernel = build_sg_kernels(spec, "cpu")
    base = np.linspace(0.0, 1.0, spec.point_count, dtype=np.float32)
    shifted = base.copy()
    shifted[217:] += 5.0

    before = build_model_input(base, spec, smooth_kernel, d1_kernel, "cpu")
    after = build_model_input(shifted, spec, smooth_kernel, d1_kernel, "cpu")
    assert before.shape == (2, 851)
    assert torch.equal(before[1, :217], after[1, :217])

    # 对照：不分段时接缝左侧会被右侧数据污染，证明断言能捕捉跨段混点。
    merged = build_model_input(
        shifted,
        replace(spec, segment_bounds=((0, 851),)),
        smooth_kernel,
        d1_kernel,
        "cpu",
    )
    assert not torch.allclose(merged[1, 210:217], before[1, 210:217])


def test_derivative_does_not_cross_bad_band_seam() -> None:
    spec = _segment_spec(d1_enable=True)
    smooth_kernel, d1_kernel = build_sg_kernels(spec, "cpu")
    base = np.linspace(0.0, 1.0, spec.point_count, dtype=np.float32)
    shifted = base.copy()
    shifted[217:] += 5.0

    before = build_model_input(base, spec, smooth_kernel, d1_kernel, "cpu")
    after = build_model_input(shifted, spec, smooth_kernel, d1_kernel, "cpu")
    assert before.shape == (3, 851)
    assert torch.equal(before[2, :217], after[2, :217])
