from pathlib import Path

import numpy as np

from ramanv2.data import build
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.io import write_arc_data
from ramanv2.data.profiles import DatasetProfile
from ramanv2.core.config import InputConfig
from ramanv2.spectra.preprocess import airpls_baseline


def _collect_train_values(train_dir: Path) -> dict[str, np.ndarray]:
    """按相对路径读取构建结果，便于比较串行和并行输出。"""
    return {
        path.relative_to(train_dir).as_posix(): np.loadtxt(path)
        for path in sorted(train_dir.rglob("*.arc_data"))
    }


def test_airpls_penalty_cache_preserves_baseline_values():
    values = np.sin(np.linspace(0.0, 12.0, 128)) + np.linspace(0.0, 2.0, 128)
    first = airpls_baseline(values, lam=1e5, niter=10)
    second = airpls_baseline(values, lam=1e5, niter=10)
    np.testing.assert_allclose(second, first, rtol=0.0, atol=0.0)


def test_build_train_parallel_matches_serial(tmp_path, monkeypatch):
    init_dir = tmp_path / "init"
    axis = np.linspace(500.0, 1900.0, 141)
    for folder, phase in (("AA01", 0.0), ("BB01", 0.4)):
        for index in range(2):
            values = np.sin(axis / 70.0 + phase + index * 0.1) + axis * 0.002
            write_arc_data(init_dir / folder / f"sample_{index}.arc_data", axis, values)

    profile = DatasetProfile("temp", "temp")
    input_config = InputConfig(target_points=101)
    build_config = DataBuildConfig(min_samples_per_class=1, baseline_max_iter=5)

    monkeypatch.setattr(build, "_resolve_build_worker_count", lambda: 1)
    build.build_train(profile, tmp_path, build_config, input_config)
    serial = _collect_train_values(tmp_path / "train")

    monkeypatch.setattr(build, "_resolve_build_worker_count", lambda: 2)
    build.build_train(profile, tmp_path, build_config, input_config)
    parallel = _collect_train_values(tmp_path / "train")

    assert parallel.keys() == serial.keys()
    for relative_path in serial:
        np.testing.assert_allclose(parallel[relative_path], serial[relative_path], rtol=0.0, atol=0.0)
