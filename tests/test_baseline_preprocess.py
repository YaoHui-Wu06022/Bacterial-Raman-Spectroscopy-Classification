"""验证默认 airPLS 配置、基线算法、坏段隔离及固定轴预处理。"""

import numpy as np

from ramanv2.spectra import preprocess
from ramanv2.spectra.bands import build_valid_mask




def test_airpls_excludes_masked_values_from_fitting_and_statistics():
    axis = np.linspace(400.0, 2000.0, 1144)
    good = build_valid_mask(axis, ((890.0, 950.0),))
    values = 1000.0 + 0.05 * axis + np.sin(axis)
    values += 25.0 * np.exp(-((axis - 1004.0) / 10.0) ** 2)
    expected = preprocess.airpls_baseline(values, lam=1e5, niter=15, valid_mask=good)
    for replacement in (-10000.0, 10000.0):
        changed = values.copy()
        changed[~good] = replacement
        actual = preprocess.airpls_baseline(changed, lam=1e5, niter=15, valid_mask=good)
        np.testing.assert_array_equal(actual, expected)






def test_interpolation_uses_own_segment_endpoint_at_bad_band_edges():
    source = np.array([880.0, 888.0, 890.0, 920.0, 950.0, 952.0, 960.0])
    values = np.array([10.0, 12.0, 9999.0, 9999.0, 9999.0, 100.0, 120.0])
    target = np.array([880.0, 889.0, 951.0, 958.0])
    result = preprocess.interpolate_valid_bands(source, values, target, ((890.0, 950.0),))
    assert result[1] == 12.0
    assert result[2] == 100.0
    changed = values.copy()
    changed[source >= 890.0] *= 100.0
    actual = preprocess.interpolate_valid_bands(source, changed, target, ((890.0, 950.0),))
    np.testing.assert_array_equal(actual[:2], result[:2])
    changed = values.copy()
    changed[source <= 950.0] *= 100.0
    actual = preprocess.interpolate_valid_bands(source, changed, target, ((890.0, 950.0),))
    np.testing.assert_array_equal(actual[2:], result[2:])


def test_preprocess_rejects_mismatched_axis_and_intensity_lengths():
    axis = np.linspace(600.0, 1800.0, 8)
    values = np.ones(7)
    with np.testing.assert_raises_regex(ValueError, "wavenumbers=8, spectrum=7"):
        preprocess.preprocess_single_spectrum(
            axis,
            values,
            cut_min=600.0,
            cut_max=1800.0,
            reference_wavenumbers=np.linspace(600.0, 1800.0, 8),
            bad_bands=(),
            baseline_lam=1e5,
            baseline_asls_p=0.01,
            baseline_max_iter=1,
        )




