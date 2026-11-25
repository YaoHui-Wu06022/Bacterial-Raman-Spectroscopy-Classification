"""analysis 命令参数解析。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", required=True, help="模型槽位目录")
    parser.add_argument("--level", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(run_command=run_command)


def run_command(args: argparse.Namespace) -> int:
    from ramanv2.analysis.interpret import interpret_slot

    device = "cpu" if args.cpu else None
    print(interpret_slot(args.source_dir, args.level, device))
    return 0
