from ramanv2.training.cli import _resolve_parent, build_parser


def test_training_cli_parses_request_scope() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--profile",
            "GN",
            "--level",
            "level_2",
            "--parent",
            "3",
            "--experiment-dir",
            "output/example",
            "--run-name",
            "run_test",
            "--resume-run-dir",
            "output/example/level_2/parent_3/run_test",
        ]
    )

    assert args.parent == "3"
    assert args.global_enable is False
    assert _resolve_parent(args.parent) == (3, None)
    assert _resolve_parent("GenusA") == (None, "GenusA")


def test_training_cli_global_mode_is_explicit() -> None:
    args = build_parser().parse_args(
        ["--profile", "GN", "--level", "level_2", "--global"]
    )

    assert args.global_enable is True
