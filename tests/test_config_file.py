from __future__ import annotations

from pathlib import Path

from ramanv2.core.config import build_config
from ramanv2.core.config_file import (
    assert_shared_compatible,
    read_yaml_dict,
    save_model_config,
    save_resolved_config,
    save_shared_config,
)


def test_config_snapshots_round_trip_shared_fields(tmp_path: Path) -> None:
    """共享、模型和完整快照均可写入并读取。"""
    config = build_config({"bad_bands": [(100.0, 110.0)]})
    shared_path = tmp_path / "shared_config.yaml"
    model_path = tmp_path / "model_config.yaml"
    resolved_path = tmp_path / "resolved_config.yaml"

    save_shared_config(shared_path, config)
    save_model_config(model_path, config)
    save_resolved_config(resolved_path, config)

    assert_shared_compatible(config, read_yaml_dict(shared_path))
    assert read_yaml_dict(model_path)
    assert read_yaml_dict(resolved_path)
