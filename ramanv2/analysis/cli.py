"""analysis 命令参数解析。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="command", required=True)
    interpret = commands.add_parser("interpret", help="运行模型可解释性分析")
    modes = interpret.add_subparsers(dest="mode", required=True)
    for name in ("run", "parent-routed"):
        command = modes.add_parser(name)
        command.add_argument("--source-dir", required=True)
        command.add_argument("--level", required=True)
        command.add_argument("--cpu", action="store_true")
        if name == "parent-routed":
            command.add_argument("--parent", default=None)
        command.set_defaults(run_command=run_command)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raman 模型分析")
    configure_parser(parser)
    return parser


def run_command(args: argparse.Namespace) -> int:
    from ramanv2.analysis.runner import run_interpret_parent_routed, run_interpret_run

    device = "cpu" if args.cpu else None
    if args.mode == "run":
        print(run_interpret_run(args.source_dir, args.level, device))
    else:
        print(run_interpret_parent_routed(args.source_dir, args.level, args.parent, device))
    return 0
