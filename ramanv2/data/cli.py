"""数据集离线构建与辅助产物的命令行适配。"""

from __future__ import annotations

import argparse


DATA_STAGES = ("init", "init_test", "train", "test")
STAGE_ROOT_FIELDS = {
    "init": "root_init",
    "init_test": "root_init_test",
    "train": "root_train_clean",
    "test": "root_test",
}


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册 data 子命令及其稳定参数。"""
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="从原始 init 构建离线数据集")
    build_commands = build.add_subparsers(dest="build_target", required=True)
    for target, description in (("train", "构建训练集"), ("test", "构建独立测试集")):
        target_parser = build_commands.add_parser(target, help=description)
        target_parser.add_argument("--dataset", default=None, help="数据集标识；默认按 profile 归属解析")
        target_parser.add_argument("--profile", required=True, help="常规数据集 profile 标识")
        target_parser.set_defaults(run_command=run_command)

    for name, description in (("plot", "从 train 生成训练集均值图"),):
        command = commands.add_parser(name, help=description)
        command.add_argument("--dataset", default=None, help="数据集标识；默认按 profile 归属解析")
        command.add_argument("--profile", required=True, help="常规数据集 profile 标识")
        command.set_defaults(run_command=run_command)

    count = commands.add_parser("count", help="统计一个数据阶段的光谱文件数")
    count.add_argument("--dataset", default=None, help="数据集标识；默认按 profile 归属解析")
    count.add_argument("--profile", required=True, help="常规数据集 profile 标识")
    count.add_argument("--stage", choices=DATA_STAGES, default="train", help="数据阶段，默认 train")
    count.set_defaults(run_command=run_command)


def run_command(args: argparse.Namespace) -> int:
    """执行一个已解析的 data 子命令。"""
    from ramanv2.core.config import InputConfig
    from ramanv2.data.profiles import resolve_selection

    selection = resolve_selection(args.dataset, args.profile)
    profile = selection.profile
    dataset_dir = selection.dataset_dir
    spec = selection.spec
    if args.command == "build":
        from ramanv2.data.builders.train_test import build_test, build_train

        if args.build_target == "train":
            build_train(
                profile,
                dataset_dir,
                input_config=InputConfig(bad_bands=spec.bad_bands),
                hierarchy=spec.hierarchy,
            )
        else:
            build_test(
                profile,
                dataset_dir,
                input_config=InputConfig(bad_bands=spec.bad_bands),
            )
        return 0
    if args.command == "count":
        from ramanv2.data.count import count_dataset, print_count_results

        stage_path = dataset_dir / getattr(profile, STAGE_ROOT_FIELDS[args.stage])
        tree, total_files = count_dataset(stage_path)
        print_count_results(tree, total_files)
        return 0
    if args.command == "plot":
        from ramanv2.core.config import InputConfig
        from ramanv2.data.plots.train import plot_train

        plot_train(profile, dataset_dir, InputConfig(bad_bands=spec.bad_bands))
        return 0
    raise RuntimeError(f"未知 data 命令：{args.command}")
