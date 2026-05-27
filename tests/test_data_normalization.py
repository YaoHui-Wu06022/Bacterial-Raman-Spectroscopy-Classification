import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from raman.data.input import normalize_spectrum
from raman.tool.array import contiguous_regions, nonnegative_points, odd_window_points
from raman.tool.plotting import insert_nan_gaps


class NormalizeSpectrumTest(unittest.TestCase):
    def test_numpy_1d_methods(self):
        values = np.array([1.0, 2.0, 4.0], dtype=np.float32)

        expected_standardized = (values - values.mean()) / (values.std() + 1e-8)
        np.testing.assert_allclose(
            normalize_spectrum(values, "snv"),
            expected_standardized,
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            normalize_spectrum(values, "minmax"),
            np.array([0.0, 1.0 / 3.0, 1.0], dtype=np.float32),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            normalize_spectrum(values, "l2"),
            values / np.sqrt(np.sum(values * values)),
            rtol=1e-6,
        )

    def test_numpy_2d_normalizes_each_row(self):
        values = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 8.0],
            ],
            dtype=np.float32,
        )

        result = normalize_spectrum(values, "minmax")

        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [0.0, 0.5, 1.0],
                    [0.0, 1.0 / 3.0, 1.0],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
        )

    def test_torch_tensor_returns_tensor(self):
        if torch is None:
            self.skipTest("torch is not installed in this Python environment")

        values = torch.tensor([[1.0, 2.0, 3.0]])

        result = normalize_spectrum(values, "l2")

        self.assertIsInstance(result, torch.Tensor)
        torch.testing.assert_close(
            result,
            values / torch.sqrt(torch.sum(values * values, dim=-1, keepdim=True)),
        )

    def test_preserve_nan_only_normalizes_finite_values(self):
        values = np.array([1.0, np.nan, 3.0], dtype=np.float32)

        result = normalize_spectrum(values, "snv", preserve_nan=True)

        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[[0, 2]], np.array([-1.0, 1.0], dtype=np.float32), rtol=1e-6)

    def test_unknown_method_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "norm_method"):
            normalize_spectrum(np.array([1.0, 2.0], dtype=np.float32), "unknown")


class DataToolTest(unittest.TestCase):
    def test_window_and_point_normalizers(self):
        self.assertEqual(odd_window_points(4), 5)
        self.assertEqual(odd_window_points(1), 3)
        self.assertEqual(nonnegative_points(-2.2), 0)
        self.assertEqual(nonnegative_points(2.6), 3)

    def test_contiguous_regions(self):
        regions = contiguous_regions(np.array([False, True, True, False, True]))
        self.assertEqual(regions, [(1, 3), (4, 5)])

    def test_insert_nan_gaps(self):
        wn = np.array([1.0, 2.0, 3.0, 20.0], dtype=np.float32)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        wn_out, values_out = insert_nan_gaps(wn, values)

        self.assertEqual(wn_out.shape[0], 5)
        self.assertTrue(np.isnan(values_out[3]))


class ModelInputSmokeTest(unittest.TestCase):
    def test_build_model_input_all_norm_methods(self):
        if torch is None:
            self.skipTest("torch is not installed in this Python environment")

        from raman.data.input import build_model_input

        raw = np.linspace(1.0, 5.0, 16, dtype=np.float32)
        cfg = SimpleNamespace(
            norm_method="snv",
            smooth_use=False,
            d1_use=False,
            in_channels=1,
        )

        for method in ("snv", "minmax", "l2"):
            cfg.norm_method = method
            x = build_model_input(
                raw,
                config=cfg,
                sg_smooth=None,
                sg_d1=None,
                device="cpu",
                augment=False,
            )
            self.assertEqual(tuple(x.shape), (1, raw.shape[0]))
            self.assertTrue(bool(torch.isfinite(x).all()))


if __name__ == "__main__":
    unittest.main()
