from ramanv2.pipeline.cli import build_parser
from ramanv2.pipeline.cli import _pending_audit_dates
from ramanv2.pipeline.state import PipelineState


def test_pipeline_cli_registers_single_pipeline_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["clean-status"]).command == "clean-status"
    assert parser.parse_args(["affine", "--date", "20240101"]).command == "affine"
    assert parser.parse_args(["audit", "--date", "20240101"]).command == "audit"
    assert parser.parse_args(["init"]).command == "init"
    assert parser.parse_args(["acquire"]).command == "acquire"
    assert parser.parse_args(["plot-prefix", "--input-root", "a", "--output-root", "b"]).command == "plot-prefix"


def test_pipeline_audit_default_is_incremental() -> None:
    state = PipelineState(
        affine_completed_dates={"20240101", "20240102"},
        audit_completed_dates={"20240101"},
    )
    assert _pending_audit_dates(state) == ["20240102"]
