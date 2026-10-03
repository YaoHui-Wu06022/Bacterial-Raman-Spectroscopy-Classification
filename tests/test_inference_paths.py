from pathlib import Path
from types import SimpleNamespace

from ramanv2.inference.runner import _resolve_test_dir


def test_regular_profile_defaults_to_csdata_and_explicit_input_wins(tmp_path: Path) -> None:
    predictor = SimpleNamespace(profile_id="GN")

    assert _resolve_test_dir(predictor, None).name == "CSdata"
    assert _resolve_test_dir(predictor, tmp_path).resolve() == tmp_path.resolve()

