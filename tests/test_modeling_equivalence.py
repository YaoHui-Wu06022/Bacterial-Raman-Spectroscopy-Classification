from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from raman.config import make_config
from raman.model import RamanClassifier1D as ReferenceRamanClassifier1D
from ramanv2.core.config import build_config
from ramanv2.core.experiment_reader import load_run_snapshot
from ramanv2.core.input_spec import InputSpec, build_input_spec
from ramanv2.modeling.factory import build_model
from ramanv2.modeling.factory import validate_model_input
from ramanv2.modeling.spec import build_model_spec


def test_classifier_state_keys_and_logits_match_reference():
    config = make_config()
    target_config = build_config(config.to_dict())
    torch.manual_seed(20260803)
    reference = ReferenceRamanClassifier1D(5, config).eval()
    torch.manual_seed(20260803)
    target = build_model(5, build_model_spec(target_config.model, build_input_spec(target_config.input))).eval()

    assert tuple(target.state_dict()) == tuple(reference.state_dict())
    target.load_state_dict(reference.state_dict(), strict=True)
    values = torch.randn(2, config.in_channels, 896)
    with torch.no_grad():
        reference_logits, reference_features = reference(values, return_feat=True)
        target_logits, target_features = target(values, return_embedding_enable=True)
    torch.testing.assert_close(target_logits, reference_logits)
    torch.testing.assert_close(target_features, reference_features)
    assert not hasattr(target, "_build_config")


def test_gn_checkpoint_loads_strictly_and_runs_cpu_forward():
    run_dir = Path("output/GN/20260802_231209_88%/level_1/run_20260802_231210")
    state = torch.load(run_dir / "level_1_model.pt", map_location="cpu")
    snapshot = load_run_snapshot(run_dir)
    config = snapshot.config
    input_spec = build_input_spec(config.input)
    model = build_model(state["head.weight"].shape[0], build_model_spec(config.model, input_spec)).eval()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        logits = model(torch.randn(1, input_spec.in_channels, 896))
    assert logits.shape == (1, state["head.weight"].shape[0])


def test_supported_model_combinations_run_forward():
    config = build_config(make_config().to_dict())
    base = build_model_spec(config.model, build_input_spec(config.input))
    combinations = (
        replace(base, backbone_type="cnn", encoder_type="none", pooling_type="stat"),
        replace(base, backbone_type="direct", encoder_type="lstm", pooling_type="attn", cosine_head_enable=True),
    )
    for model_spec in combinations:
        model = build_model(3, model_spec).eval()
        with torch.no_grad():
            logits = model(torch.randn(2, model_spec.in_channels, 128))
        assert logits.shape == (2, 3)


def test_model_input_validation_checks_channels_and_encoder_length():
    config = build_config(make_config().to_dict())
    model_spec = build_model_spec(config.model, build_input_spec(config.input))
    validate_model_input(model_spec, InputSpec(model_spec.in_channels, 896))
    with pytest.raises(ValueError, match="通道数"):
        validate_model_input(model_spec, InputSpec(model_spec.in_channels + 1, 896))
    direct_spec = replace(model_spec, backbone_type="direct")
    with pytest.raises(ValueError, match="位置编码"):
        validate_model_input(direct_spec, InputSpec(direct_spec.in_channels, 1001))
