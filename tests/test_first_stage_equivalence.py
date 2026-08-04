from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from raman.data.build import build_train as reference_build_train
from raman.data.profiles import DatasetProfile as ReferenceDatasetProfile
from raman.pipeline import DEFAULT_PIPELINE_CONFIG
from raman.data.input import normalize_spectrum as reference_normalize
from raman.data.input import sg_coeff as reference_sg_coeff
from raman.data.preprocess import preprocess_single_spectrum as reference_preprocess
from raman.tool import spectrum as reference_spectrum
from raman.tool.array import median_filter_1d as reference_median_filter
from raman.tool.array import odd_window_points as reference_odd_window_points
from raman.tool.path import PROJECT_ROOT as reference_project_root
from raman.tool.spectrum import estimate_gap_indices as reference_gap_indices
from raman.tool.spectrum import median_step_cm as reference_median_step
from ramanv2.core.paths import PROJECT_ROOT, stanford_reference_wavenumbers_path
from ramanv2.data.build import build_train as target_build_train
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.io import pack_init, unpack_init
from ramanv2.data.profiles import DatasetProfile, get_profile
from ramanv2.spectra import axis, bands, filters, normalize, preprocess


def test_project_root_and_reference_location_match_contract():
    assert PROJECT_ROOT == reference_project_root
    assert stanford_reference_wavenumbers_path() == (
        PROJECT_ROOT / "dataset" / "Stanforddataset" / "reference_wavenumbers.npy"
    )
    assert stanford_reference_wavenumbers_path().is_file()


def test_axis_and_bad_band_functions_match_existing_behavior():
    config = SimpleNamespace(cut_min=600.0, cut_max=1800.0, target_points=13, bad_bands=[(900, 800)])
    values = np.array([600.0, 700.0, 800.0, 900.0, 1000.0, 1800.0])

    np.testing.assert_array_equal(axis.build_wn_ref(600, 1800, 13), reference_spectrum.build_wn_ref(600, 1800, 13))
    assert bands.normalize_bad_bands(config.bad_bands) == reference_spectrum.normalize_bad_bands(config.bad_bands)
    np.testing.assert_array_equal(bands.build_valid_mask(values, config.bad_bands), reference_spectrum.build_valid_mask(values, config.bad_bands))
    np.testing.assert_array_equal(axis.build_wavenumber_axis(11, config), reference_spectrum.build_wavenumber_axis(11, config))
    np.testing.assert_array_equal(axis.expected_wavenumbers(config), reference_spectrum.expected_wavenumbers(config))
    assert axis.median_step_cm(values) == reference_median_step(values)
    assert axis.estimate_gap_indices([1, 2, 3, 10, 11]) == reference_gap_indices([1, 2, 3, 10, 11])


@pytest.mark.parametrize("method", ["none", "snv", "minmax", "l2"])
def test_normalization_matches_existing_numpy_behavior(method):
    values = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(
        normalize.normalize_spectrum(values, method, preserve_nan_enable=True),
        reference_normalize(values, method, preserve_nan=True),
        equal_nan=True,
    )


def test_filtering_matches_existing_behavior():
    values = np.array([1.0, 9.0, 2.0, 8.0, 3.0], dtype=np.float32)
    assert filters.odd_window_points(4) == reference_odd_window_points(4)
    np.testing.assert_array_equal(filters.median_filter_1d(values, 3), reference_median_filter(values, 3))
    np.testing.assert_array_equal(filters.sg_coeff(9, 3, 1), reference_sg_coeff(9, 3, 1))


def test_single_spectrum_preprocess_matches_existing_behavior():
    wavenumbers = np.linspace(500.0, 1900.0, 141)
    spectrum = 0.004 * (wavenumbers - 1200.0) + np.exp(-((wavenumbers - 1000.0) / 40.0) ** 2)
    reference = np.linspace(600.0, 1800.0, 101)
    reference_options = {
        "cut_min": 600.0,
        "cut_max": 1800.0,
        "bad_bands": [(900.0, 920.0)],
        "baseline_lam": 1e4,
        "baseline_asls_p": 0.01,
        "baseline_max_iter": 5,
        "baseline_method": "asls",
        "cosmic_ray_remove": True,
    }
    target_options = {
        key: value
        for key, value in reference_options.items()
        if key != "cosmic_ray_remove"
    }
    target_options["cosmic_ray_enable"] = reference_options["cosmic_ray_remove"]
    reference_axis, reference_values, reference_stats = reference_preprocess(
        wavenumbers,
        spectrum,
        wn_ref=reference,
        **reference_options,
    )
    target_axis, target_values, target_stats = preprocess.preprocess_single_spectrum(
        wavenumbers,
        spectrum,
        reference_wavenumbers=reference,
        **target_options,
    )
    np.testing.assert_allclose(target_axis, reference_axis)
    np.testing.assert_allclose(target_values, reference_values, rtol=1e-6, atol=1e-6)
    assert int(target_stats) == int(reference_stats)


def test_regular_profiles_exclude_stanford():
    assert get_profile("GN").dataset_name == "GN"
    with pytest.raises(KeyError):
        get_profile("Stanford")


def test_pack_and_unpack_round_trip(tmp_path):
    init_dir = tmp_path / "init" / "AA01"
    init_dir.mkdir(parents=True)
    source_axis = np.array([600.0, 601.0, 602.0])
    source_values = np.array([1.0, 2.0, 3.0])
    from ramanv2.data.io import read_arc_data, write_arc_data

    write_arc_data(init_dir / "sample.arc_data", source_axis, source_values)
    packed = tmp_path / "init.npz"
    restored = tmp_path / "restored"
    pack_init(tmp_path / "init", packed, is_verbose=False)
    unpack_init(packed, restored, is_verbose=False)
    restored_axis, restored_values = read_arc_data(restored / "AA01" / "sample.arc_data")
    np.testing.assert_allclose(restored_axis, source_axis)
    np.testing.assert_allclose(restored_values, source_values)


def test_train_build_matches_reference_and_skips_pca_log_when_disabled(tmp_path):
    axis_values = np.linspace(500.0, 1900.0, 141)
    source_values = np.sin(axis_values / 80.0) + 0.003 * axis_values
    for root in (tmp_path / "reference", tmp_path / "target"):
        source_dir = root / "init" / "AA01"
        source_dir.mkdir(parents=True)
        from ramanv2.data.io import write_arc_data

        write_arc_data(source_dir / "sample.arc_data", axis_values, source_values)

    reference_profile = ReferenceDatasetProfile(profile_id="temp", dataset_name="temp")
    target_profile = DatasetProfile(profile_id="temp", dataset_name="temp")
    reference_config = replace(DEFAULT_PIPELINE_CONFIG, min_samples_per_class=1, pca_enabled=False)
    target_config = DataBuildConfig(min_samples_per_class=1, pca_enable=False)
    reference_build_train(reference_profile, tmp_path / "reference", pipeline_config=reference_config)
    target_build_train(target_profile, tmp_path / "target", config=target_config)

    reference_file = next((tmp_path / "reference" / "train").rglob("*.arc_data"))
    target_file = next((tmp_path / "target" / "train").rglob("*.arc_data"))
    np.testing.assert_allclose(np.loadtxt(target_file), np.loadtxt(reference_file), rtol=1e-6, atol=1e-6)
    assert not (tmp_path / "target" / "pca_log.txt").exists()


def test_train_build_creates_pca_log_only_when_enabled(tmp_path):
    axis_values = np.linspace(500.0, 1900.0, 141)
    source_dir = tmp_path / "init" / "AA01"
    source_dir.mkdir(parents=True)
    from ramanv2.data.io import write_arc_data

    write_arc_data(source_dir / "first.arc_data", axis_values, np.sin(axis_values / 80.0))
    write_arc_data(source_dir / "second.arc_data", axis_values, np.cos(axis_values / 80.0))
    profile = DatasetProfile(profile_id="temp", dataset_name="temp")
    build_config = DataBuildConfig(min_samples_per_class=1, pca_enable=True)

    target_build_train(profile, tmp_path, config=build_config)

    assert (tmp_path / "pca_log.txt").is_file()
