from ramanv2.pipeline.cli import build_parser
from ramanv2.pipeline.state import PipelineState
import pytest


def test_pipeline_cli_registers_single_pipeline_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["clean-status"]).command == "clean-status"
    assert parser.parse_args(["affine", "--date", "20240101"]).command == "affine"
    with pytest.raises(SystemExit):
        parser.parse_args(["audit", "--date", "20240101"])
    assert parser.parse_args(["init"]).command == "init"
    assert parser.parse_args(["acquire"]).command == "acquire"
    assert parser.parse_args(["plot-prefix", "--input-root", "a", "--output-root", "b"]).command == "plot-prefix"


def test_old_audit_state_is_ignored_and_not_saved_again():
    payload = PipelineState(affine_completed_dates={"20240101"}).to_payload()
    payload["audit_completed_dates"] = []
    state = PipelineState.from_payload(payload)
    assert state.affine_completed_dates == {"20240101"}
    assert not hasattr(state, "audit_completed_dates")
    assert "audit_completed_dates" not in state.to_payload()
