"""analysis 命令参数解析。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-dir",
        required=True,
        help="模型槽位目录；--level-summary 时为实验目录",
    )
    parser.add_argument("--level", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--level-summary",
        action="store_true",
        dest="level_summary_enable",
        help="聚合该层多个模型的归因",
    )
    parser.set_defaults(run_command=run_command)


def run_command(args: argparse.Namespace) -> int:
    from ramanv2.analysis.interpret import interpret_level, interpret_slot

    device = "cpu" if args.cpu else None
    if args.level_summary_enable:
        print(interpret_level(args.source_dir, args.level, device))
    else:
        print(interpret_slot(args.source_dir, args.level, device))
    return 0
