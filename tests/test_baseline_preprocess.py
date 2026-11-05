"""验证 airPLS 坏段隔离、基线输入平滑与坏段边界插值。"""

import numpy as np
from scipy.signal import savgol_filter

from ramanv2.spectra import preprocess
from ramanv2.spectra.bands import build_valid_mask


def test_bad_band_masks_isolate_airpls_and_interpolation():
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


def test_preprocess_smooths_baseline_input_only(monkeypatch):
    axis = np.arange(400.0, 2001.0)
    values = 1000.0 - 0.1 * (axis - 400.0) + 40.0 * np.exp(-((axis - 1004.0) / 8.0) ** 2)
    values[604] += 500.0
    captured = {}

    def fake_estimate(spectrum, **kwargs):
        captured["input"] = np.asarray(spectrum, dtype=np.float64).copy()
        captured["method"] = kwargs["method"]
        return np.zeros_like(np.asarray(spectrum, dtype=np.float64))

    monkeypatch.setattr(preprocess, "estimate_baseline", fake_estimate)
    reference = np.linspace(600.0, 1800.0, 101)
    _, corrected, _ = preprocess.preprocess_single_spectrum(
        axis,
        values,
        cut_min=600.0,
        cut_max=1800.0,
        reference_wavenumbers=reference,
        bad_bands=(),
        baseline_lam=1e5,
        baseline_asls_p=0.01,
        baseline_max_iter=15,
        baseline_fit_min=400.0,
        baseline_fit_max=2000.0,
        baseline_method="airpls",
    )

    np.testing.assert_allclose(
        captured["input"],
        preprocess.smooth_baseline_input(values),
        rtol=1e-5,
        atol=1e-3,
    )
    assert not np.allclose(captured["input"], values)
    assert captured["method"] == "airpls"
    crop = (axis >= 600.0) & (axis <= 1800.0)
    np.testing.assert_allclose(
        corrected,
        np.interp(reference, axis[crop], values[crop]),
        rtol=1e-5,
        atol=1e-3,
    )


def test_smooth_baseline_input_uses_five_point_savgol():
    values = np.linspace(0.0, 10.0, 50) ** 2
    np.testing.assert_allclose(
        preprocess.smooth_baseline_input(values),
        savgol_filter(
            values,
            preprocess.BASELINE_INPUT_SMOOTH_WINDOW,
            3,
            mode="interp",
        ),
    )
    short = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(preprocess.smooth_baseline_input(short), short)
