import subprocess
import sys

import pytest

from ramanv2.cli.main import build_parser


def test_root_cli_registers_migrated_commands() -> None:
    parser = build_parser()
    pipeline_args = parser.parse_args(["pipeline", "audit", "--date", "20240101"])
    data_args = parser.parse_args(["data", "build", "train", "--profile", "GN"])
    train_args = parser.parse_args(["train", "--profile", "GN", "--level", "level_1"])
    infer_args = parser.parse_args(
        ["infer", "test", "--source-dir", "output/example", "--level", "level_1"]
    )
    evaluation_args = parser.parse_args(
        [
            "evaluate",
            "baseline",
            "parent-routed",
            "--source-dir",
            "output/example",
            "--level",
            "level_1",
        ]
    )
    zip_args = parser.parse_args(["zip"])

    assert pipeline_args.domain == "pipeline"
    assert pipeline_args.command == "audit"
    assert callable(pipeline_args.run_command)
    assert data_args.domain == "data"
    assert data_args.command == "build"
    assert data_args.build_target == "train"
    assert callable(data_args.run_command)
    assert train_args.domain == "train"
    assert train_args.level == "level_1"
    assert train_args.global_enable is False
    assert callable(train_args.run_command)
    assert infer_args.domain == "infer"
    assert infer_args.command == "test"
    assert callable(infer_args.run_command)
    assert evaluation_args.target == "baseline"
    assert evaluation_args.mode == "parent-routed"
    assert callable(evaluation_args.run_command)
    assert zip_args.domain == "zip"
    assert callable(zip_args.run_command)
    with pytest.raises(SystemExit):
        parser.parse_args(["stanford", "transfer"])


def test_root_help_does_not_load_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ramanv2.cli.main import build_parser; build_parser(); print('torch' in sys.modules)",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "False"
