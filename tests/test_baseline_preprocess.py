"""验证默认 airPLS 配置、基线算法、坏段隔离及固定轴预处理。"""

import numpy as np
import pytest

from ramanv2.data.config import DEFAULT_BUILD_CONFIG
from ramanv2.spectra import preprocess
from ramanv2.spectra.axis import build_wn_ref
from ramanv2.spectra.bands import build_valid_mask


def test_default_uses_airpls_parameters():
    assert DEFAULT_BUILD_CONFIG.baseline_method == "airPLS"
    assert DEFAULT_BUILD_CONFIG.baseline_lam == 1e5
    assert DEFAULT_BUILD_CONFIG.baseline_max_iter == 15
    assert DEFAULT_BUILD_CONFIG.baseline_fit_min == 400.0
    assert DEFAULT_BUILD_CONFIG.baseline_fit_max == 2000.0


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


def test_removed_baseline_method_is_not_available():
    assert not hasattr(preprocess, "arpls_baseline")
    with pytest.raises(ValueError, match="Unknown baseline method"):
        preprocess.estimate_baseline(np.ones(128), method="arPLS")


def test_airpls_fits_constant_background_with_or_without_mask():
    values = np.full(128, 1200.0)
    for mask in (None, np.arange(128) % 5 != 0):
        result = preprocess.airpls_baseline(values, lam=1e5, niter=15, valid_mask=mask)
        assert result.shape == values.shape
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, values, rtol=0.0, atol=1e-8)


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


def test_interpolation_without_bad_bands_matches_numpy():
    axis = np.linspace(400.0, 2000.0, 1144)
    values = np.sin(axis / 80.0)
    target = np.linspace(600.0, 1800.0, 896)
    expected = np.interp(target, axis, values)
    actual = preprocess.interpolate_valid_bands(axis, values, target, ())
    np.testing.assert_array_equal(actual, expected)


def test_preprocessing_preserves_axis_length_and_negative_values(monkeypatch):
    axis = np.r_[np.linspace(400.0, 880.0, 100), [888.0, 890.0, 920.0, 950.0, 952.0], np.linspace(960.0, 2000.0, 100)]
    values = np.where(axis < 890.0, -4.0, 7.0)
    values[(axis >= 890.0) & (axis <= 950.0)] = 9999.0
    monkeypatch.setattr(preprocess, "estimate_baseline", lambda spectrum, **kwargs: np.zeros(len(spectrum)))
    target = build_wn_ref(600.0, 1800.0, 896)
    actual_axis, corrected, _ = preprocess.preprocess_single_spectrum(
        axis, values, 600.0, 1800.0, target, ((890.0, 950.0),), 1e5, 0.01, 15,
        400.0, 2000.0, "airPLS",
    )
    expected_axis = target[build_valid_mask(target, ((890.0, 950.0),))]
    np.testing.assert_array_equal(actual_axis, expected_axis)
    assert len(corrected) == 851
    np.testing.assert_array_equal(corrected[actual_axis < 890.0], -4.0)
    np.testing.assert_array_equal(corrected[actual_axis > 950.0], 7.0)
