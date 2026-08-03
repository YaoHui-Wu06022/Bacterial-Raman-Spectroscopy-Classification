"""evaluation 命令参数解析与延迟分发。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册模型评估和 PCA-SVM baseline 的子命令。"""
    targets = parser.add_subparsers(dest="target", required=True)
    model = targets.add_parser("model", help="评估深度模型")
    baseline = targets.add_parser("baseline", help="运行 PCA-SVM baseline")
    _configure_model_parser(model)
    _configure_baseline_parser(baseline)


def build_parser() -> argparse.ArgumentParser:
    """构建独立调用时使用的 evaluation 参数解析器。"""
    parser = argparse.ArgumentParser(description="Raman 验证集评估")
    configure_parser(parser)
    return parser


def run_command(args: argparse.Namespace) -> int:
    """执行已经解析完成的模型评估或 PCA-SVM baseline。"""
    if args.target == "model":
        from raman_temp.evaluation.model_eval import (
            evaluate_model_cascade,
            evaluate_model_parent_routed,
            evaluate_model_run,
        )

        device = "cpu" if args.cpu else None
        if args.mode == "run":
            result_dir = evaluate_model_run(args.source_dir, args.level, device)
        elif args.mode == "parent-routed":
            result_dir = evaluate_model_parent_routed(args.source_dir, args.level, device)
        else:
            result_dir = evaluate_model_cascade(args.source_dir, args.level, device)
    else:
        from raman_temp.evaluation.baseline import (
            BaselineSpec,
            evaluate_baseline_parent_routed,
            evaluate_baseline_run,
        )

        spec = BaselineSpec(
            all_channels_enable=args.all_channels_enable,
            pca_components=args.pca_components,
            svm_c=args.svm_c,
            svm_kernel=args.svm_kernel,
            svm_gamma=args.svm_gamma,
            random_state=args.random_state,
        )
        if args.mode == "run":
            result_dir = evaluate_baseline_run(args.source_dir, args.level, spec)
        else:
            result_dir = evaluate_baseline_parent_routed(args.source_dir, args.level, spec)
    print(result_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析并执行 evaluation 命令。"""
    args = build_parser().parse_args(argv)
    return run_command(args)


def _configure_model_parser(parser: argparse.ArgumentParser) -> None:
    """注册深度模型评估的三个已确认模式。"""
    commands = parser.add_subparsers(dest="mode", required=True)
    for name, help_text in (
        ("run", "评估一个明确 run"),
        ("parent-routed", "按真实父类选择目标层子模型"),
        ("cascade", "执行端到端层级级联评估"),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_common_options(command)
        command.add_argument("--cpu", action="store_true", help="强制使用 CPU")
        command.set_defaults(run_command=run_command)


def _configure_baseline_parser(parser: argparse.ArgumentParser) -> None:
    """注册 PCA-SVM 的两个已确认模式和可复现实验参数。"""
    commands = parser.add_subparsers(dest="mode", required=True)
    for name, help_text in (
        ("run", "在一个明确 run 的类别空间拟合 PCA-SVM"),
        ("parent-routed", "按真实父类分别拟合 PCA-SVM"),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_common_options(command)
        command.add_argument("--all-channels", action="store_true", dest="all_channels_enable")
        command.add_argument("--pca-components", type=float, default=0.95)
        command.add_argument("--svm-c", type=float, default=1.0)
        command.add_argument("--svm-kernel", default="rbf")
        command.add_argument("--svm-gamma", default="scale")
        command.add_argument("--random-state", type=int, default=42)
        command.set_defaults(run_command=run_command)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """注册评估目标共有的实验目录和层级参数。"""
    parser.add_argument("--source-dir", required=True, help="实验目录或单个 run 目录")
    parser.add_argument("--level", required=True, help="目标层级，例如 level_1")
