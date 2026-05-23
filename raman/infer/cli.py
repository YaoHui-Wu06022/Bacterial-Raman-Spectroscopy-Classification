import argparse
import sys


def build_parser():
    """构建 infer 总入口参数解析器"""
    parser = argparse.ArgumentParser(description="Raman inference tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("test", help="Run independent test inference")
    return parser


def main(argv=None):
    """分发 infer 子命令"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    if argv[0] == "test":
        from raman.infer import test as test_infer

        return test_infer.main(argv[1:])
    build_parser().error(f"unknown command: {argv[0]}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
