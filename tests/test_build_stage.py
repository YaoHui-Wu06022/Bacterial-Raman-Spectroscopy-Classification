"""train 构建只替换 train，不触碰由 plot_train 维护的审图目录。"""

import numpy as np

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.builders import stage
from ramanv2.data.config import DataBuildConfig
from ramanv2.data.profiles import get_profile


def test_build_train_keeps_existing_figure_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "_resolve_build_worker_count", lambda: 1)
    profile = get_profile("GN")
    wavenumbers = np.linspace(600.0, 1800.0, 32)
    for index in range(2):
        write_arc_data(
            tmp_path / profile.root_init / "Escherichia" / "EC01" / f"{index}.arc_data",
            wavenumbers,
            np.linspace(1.0, 2.0, 32),
        )
    plot = tmp_path / profile.root_train_fig / "Escherichia" / "EC.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"old plot")

    stage.build_train(profile, tmp_path, DataBuildConfig(min_samples_per_class=1))

    assert plot.read_bytes() == b"old plot"
    assert any((tmp_path / profile.root_train_clean).rglob("*.arc_data"))
