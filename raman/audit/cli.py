"""审核工具统一命令入口"""

from __future__ import annotations

import argparse
import sys

from raman.audit import folder, full_scan, move, single


def build_parser():
    parser = argparse.ArgumentParser(description="Raman 数据审核与移除工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("single", help="组内单谱离群审核")
    subparsers.add_parser("folder", help="同属同前缀参考组审核")
    subparsers.add_parser("full", help="全库只读异常谱复查")
    subparsers.add_parser("move", help="按路径或清单移动到 delete")
    return parser


COMMANDS = {
    "single": single.main,
    "folder": folder.main,
    "full": full_scan.main,
    "move": move.main,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    command = argv[0]
    if command not in COMMANDS:
        build_parser().error(f"unknown command: {command}")
    return COMMANDS[command](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
