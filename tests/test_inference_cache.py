from pathlib import Path
from types import SimpleNamespace

import torch

from ramanv2.common.arc_data import write_arc_data
from ramanv2.core.config import InputConfig
from ramanv2.core.input_spec import build_input_spec
from ramanv2.inference.cache import build_test_cache, load_test_cache


def _predictor() -> SimpleNamespace:
    config = InputConfig(
        cut_min=100.0,
        cut_max=101.0,
        target_points=2,
        norm_method="none",
        smooth_use=False,
        d1_use=False,
        bad_bands=(),
    )
    return SimpleNamespace(
        profile_id="GN",
        input_config=config,
        input_spec=build_input_spec(config),
        device=torch.device("cpu"),
    )


def test_test_cache_reuses_unchanged_files_and_rebuilds_changed_file(tmp_path: Path) -> None:
    source = tmp_path / "CSdata" / "CS01GN"
    cache = tmp_path / "GN" / "test"
    source.mkdir(parents=True)
    file_path = source / "sample.arc_data"
    write_arc_data(file_path, [100.0, 101.0], [1.0, 2.0])
    predictor = _predictor()

    first = build_test_cache(source.parent, cache, ["CS01GN"], predictor)
    assert first["rebuilt_count"] == 1
    assert first["reused_count"] == 0
    assert (cache / "CS01GN" / "sample.pt").is_file()
    assert not list(cache.rglob("*.npz"))

    second = build_test_cache(source.parent, cache, ["CS01GN"], predictor)
    assert second["rebuilt_count"] == 0
    assert second["reused_count"] == 1

    write_arc_data(file_path, [100.0, 101.0], [3.0, 4.0])
    third = build_test_cache(source.parent, cache, ["CS01GN"], predictor)
    assert third["rebuilt_count"] == 1
    assert third["reused_count"] == 0
    loaded = load_test_cache(cache, ["CS01GN"], predictor)
    payload = torch.load(loaded[0].files[0].path, map_location="cpu")
    assert payload["input"].shape == (1, 1, 2)


def test_test_cache_rebuilds_when_input_config_changes(tmp_path: Path) -> None:
    source = tmp_path / "CSdata" / "CS01GN"
    cache = tmp_path / "GN" / "test"
    source.mkdir(parents=True)
    write_arc_data(source / "sample.arc_data", [100.0, 101.0], [1.0, 2.0])

    first = build_test_cache(source.parent, cache, ["CS01GN"], _predictor())
    assert first["rebuilt_count"] == 1

    changed_config = InputConfig(
        cut_min=100.0,
        cut_max=101.0,
        target_points=2,
        norm_method="none",
        smooth_use=True,
        d1_use=False,
        bad_bands=(),
    )
    changed_predictor = SimpleNamespace(
        profile_id="GN",
        input_config=changed_config,
        input_spec=build_input_spec(changed_config),
        device=torch.device("cpu"),
    )
    second = build_test_cache(source.parent, cache, ["CS01GN"], changed_predictor)
    assert second["rebuilt_count"] == 1
    assert second["reused_count"] == 0
    manifest = (cache / "cache_manifest.json").read_text(encoding="utf-8")
    assert '"smooth_use": true' in manifest
