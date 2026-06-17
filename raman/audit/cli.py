"""audit 命令统一入口。"""

from __future__ import annotations

import argparse
import sys

from raman.audit import full_scan, move


def build_parser():
    """构建 audit 总命令解析器。"""
    parser = argparse.ArgumentParser(description="Raman 数据审核与移除工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("full", help="全库分阶段清洗")
    subparsers.add_parser("move", help="按路径或清单移动到 delete")
    return parser


COMMANDS = {
    "full": full_scan.main,
    "move": move.main,
}


def main(argv=None):
    """分发 audit 子命令。"""
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
