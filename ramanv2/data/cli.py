"""数据集离线构建与辅助产物的命令行适配。"""

from __future__ import annotations

import argparse


DATA_STAGES = ("init", "init_test", "train", "test")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册 data 子命令及其稳定参数。"""
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="从原始 init 构建离线数据集")
    build_commands = build.add_subparsers(dest="build_target", required=True)
    for target, description in (("train", "构建训练集"), ("test", "构建独立测试集")):
        target_parser = build_commands.add_parser(target, help=description)
        target_parser.add_argument("--profile", required=True, help="常规数据集 profile 标识")
        target_parser.set_defaults(run_command=run_command)

    for name, description in (("pack", "将 init 打包为 init.npz"), ("unpack", "将 init.npz 恢复为 init"), ("plot", "从 train 生成训练集均值图")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--profile", required=True, help="常规数据集 profile 标识")
        command.set_defaults(run_command=run_command)

    count = commands.add_parser("count", help="统计一个数据阶段的光谱文件数")
    count.add_argument("--profile", required=True, help="常规数据集 profile 标识")
    count.add_argument("--stage", choices=DATA_STAGES, default="train", help="数据阶段，默认 train")
    count.set_defaults(run_command=run_command)


def build_parser() -> argparse.ArgumentParser:
    """构建单独执行 data 包时使用的参数解析器。"""
    parser = argparse.ArgumentParser(description="Raman 数据集构建工具")
    configure_parser(parser)
    return parser


def _resolve_profile_dir(profile_key: str):
    """解析常规 profile 及其固定数据集目录。"""
    from ramanv2.data.profiles import get_dataset_dir, get_profile

    profile = get_profile(profile_key)
    return profile, get_dataset_dir(profile)


def run_command(args: argparse.Namespace) -> int:
    """执行一个已解析的 data 子命令。"""
    profile, dataset_dir = _resolve_profile_dir(args.profile)
    if args.command == "build":
        from ramanv2.core.config import InputConfig
        from ramanv2.data.build import build_test, build_train

        if args.build_target == "train":
            build_train(profile, dataset_dir, input_config=InputConfig())
        else:
            build_test(profile, dataset_dir, input_config=InputConfig())
        return 0
    if args.command == "pack":
        from ramanv2.data.io import pack_init

        pack_init(dataset_dir / profile.root_init, dataset_dir / profile.root_init_pack)
        return 0
    if args.command == "unpack":
        from ramanv2.data.io import unpack_init

        unpack_init(dataset_dir / profile.root_init_pack, dataset_dir / profile.root_init)
        return 0
    if args.command == "count":
        from ramanv2.data.count import count_dataset, print_count_results

        stage_path = dataset_dir / getattr(profile, f"root_{args.stage}")
        tree, total_files = count_dataset(stage_path)
        print_count_results(tree, total_files)
        return 0
    if args.command == "plot":
        from ramanv2.core.config import InputConfig
        from ramanv2.data.plot import plot_train

        plot_train(profile, dataset_dir, InputConfig())
        return 0
    raise RuntimeError(f"未知 data 命令：{args.command}")


def main(argv: list[str] | None = None) -> int:
    """解析并执行一个 data 子命令。"""
    args = build_parser().parse_args(argv)
    return args.run_command(args)
