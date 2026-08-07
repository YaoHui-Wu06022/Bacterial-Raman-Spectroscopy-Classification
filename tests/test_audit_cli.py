from ramanv2.audit.cli import build_parser


def test_audit_cli_parses_clean_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["clean", "--dataset", "alldata"]).command == "clean"
    assert parser.parse_args(["clean-test", "--profile", "test"]).command == "clean-test"
