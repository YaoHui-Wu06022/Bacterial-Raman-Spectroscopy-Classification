"""audit 命令参数解析与工作流调用。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册层级数据集审核命令。"""
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("clean", help="在运行池执行完整三阶段审核")
    clean.add_argument("--dataset", default="alldata")
    clean_test = commands.add_parser("clean-test", help="清洗未知标签 CS 测试集")
    clean_test.add_argument("--profile", default="test", help="固定为 test 的 CS profile 标识")
    prefix = commands.add_parser("plot-prefix", help="绘制同前缀审核总览图")
    prefix.add_argument("--dataset", default="alldata")
    prefix.add_argument("--force", action="store_true", dest="force_enable")
    for command_parser in (clean, clean_test, prefix):
        command_parser.set_defaults(run_command=run_command)


def build_parser() -> argparse.ArgumentParser:
    """构建独立调用时使用的 audit 参数解析器。"""
    parser = argparse.ArgumentParser(description="Raman 数据审核")
    configure_parser(parser)
    return parser


def run_command(args: argparse.Namespace) -> int:
    """执行一个已经解析完成的 audit 子命令。"""
    if args.command == "clean":
        from ramanv2.audit.workflow import run_clean_dir

        print(run_clean_dir(args.dataset))
        return 0
    if args.command == "clean-test":
        from ramanv2.audit.workflow import run_cs_clean_dir
        from ramanv2.core.config import build_config
        from ramanv2.data.profiles import get_dataset_dir, get_profile

        profile = get_profile(args.profile)
        if profile.profile_id != "test":
            raise ValueError("clean-test 只处理 test profile 的 CS 文件夹")
        dataset_dir = get_dataset_dir(profile)
        config = build_config({"profile_id": profile.profile_id})
        print(run_cs_clean_dir(profile, dataset_dir, config.input))
        return 0
    if args.command == "plot-prefix":
        from ramanv2.audit.plots import plot_prefix_dataset

        outputs = plot_prefix_dataset(args.dataset, force_enable=args.force_enable)
        print(f"figures={len(outputs)}")
        return 0
    raise RuntimeError(f"未知 audit 命令：{args.command}")


def main(argv: list[str] | None = None) -> int:
    """解析并执行一个 audit 子命令。"""
    args = build_parser().parse_args(argv)
    return args.run_command(args)
