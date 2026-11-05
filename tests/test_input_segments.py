from dataclasses import replace

import numpy as np
import torch

from ramanv2.core.config import InputConfig
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.data.runtime.input import build_model_input, build_sg_kernels


def _segment_spec(d1_enable: bool) -> InputSpec:
    return InputSpec(
        in_channels=3 if d1_enable else 2,
        point_count=866,
        norm_method="none",
        smooth_enable=True,
        d1_enable=d1_enable,
        smooth_window=15,
        d1_window=15,
        step_cm=(1800.0 - 600.0) / 895,
        segment_bounds=((0, 224), (224, 866)),
    )


def test_input_spec_segments_isolate_bad_band_seam() -> None:
    spec = build_input_spec(InputConfig())
    assert spec.point_count == 866
    assert spec.segment_bounds == ((0, 224), (224, 866))

    whole = build_input_spec(InputConfig(bad_bands=()))
    assert whole.point_count == 896
    assert whole.segment_bounds == ((0, 896),)

    smooth_spec = _segment_spec(d1_enable=False)
    smooth_kernel, d1_kernel = build_sg_kernels(smooth_spec, "cpu")
    base = np.linspace(0.0, 1.0, smooth_spec.point_count, dtype=np.float32)
    shifted = base.copy()
    shifted[224:] += 5.0

    before = build_model_input(base, smooth_spec, smooth_kernel, d1_kernel, "cpu")
    after = build_model_input(shifted, smooth_spec, smooth_kernel, d1_kernel, "cpu")
    assert before.shape == (2, 866)
    assert torch.equal(before[1, :224], after[1, :224])

    # 对照：不分段时接缝左侧会被右侧数据污染，证明断言能捕捉跨段混点。
    merged = build_model_input(
        shifted,
        replace(smooth_spec, segment_bounds=((0, 866),)),
        smooth_kernel,
        d1_kernel,
        "cpu",
    )
    assert not torch.allclose(merged[1, 217:224], before[1, 217:224])

    d1_spec = _segment_spec(d1_enable=True)
    smooth_kernel, d1_kernel = build_sg_kernels(d1_spec, "cpu")
    before = build_model_input(base, d1_spec, smooth_kernel, d1_kernel, "cpu")
    after = build_model_input(shifted, d1_spec, smooth_kernel, d1_kernel, "cpu")
    assert before.shape == (3, 866)
    assert torch.equal(before[2, :224], after[2, :224])
