"""evaluation 命令参数解析与延迟分发。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册模型评估和 PCA-SVM baseline 两个子命令。"""
    targets = parser.add_subparsers(dest="target", required=True)
    model = targets.add_parser("model", help="评估深度模型槽位")
    baseline = targets.add_parser("baseline", help="运行 PCA-SVM baseline")
    summary = targets.add_parser("summary", help="拼接某层级多个模型的验证/测试结果")
    _configure_model_parser(model)
    _configure_baseline_parser(baseline)
    _configure_summary_parser(summary)


def run_command(args: argparse.Namespace) -> int:
    """执行已经解析完成的模型评估、PCA-SVM baseline 或层级汇总。"""
    if args.target == "summary":
        from ramanv2.evaluation.level_summary import summarize_level_results

        report = summarize_level_results(
            args.source_dir,
            args.level,
            "cpu" if args.cpu else None,
        )
        print(report["level_summary_dir"])
        return 0
    if args.target == "model":
        from ramanv2.evaluation.model_eval import evaluate_model_run

        result_dir = evaluate_model_run(
            args.source_dir,
            args.level,
            "cpu" if args.cpu else None,
        )
    else:
        from ramanv2.evaluation.pca_svm import BaselineSpec, evaluate_baseline_run

        spec = BaselineSpec(
            all_channels_enable=args.all_channels_enable,
            pca_components=args.pca_components,
            svm_c=args.svm_c,
            svm_kernel=args.svm_kernel,
            svm_gamma=args.svm_gamma,
            random_state=args.random_state,
        )
        result_dir = evaluate_baseline_run(args.source_dir, args.level, spec)
    print(result_dir)
    return 0


def _configure_model_parser(parser: argparse.ArgumentParser) -> None:
    """注册深度模型槽位评估参数。"""
    _add_common_options(parser)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.set_defaults(run_command=run_command)


def _configure_baseline_parser(parser: argparse.ArgumentParser) -> None:
    """注册 PCA-SVM baseline 参数。"""
    _add_common_options(parser)
    parser.add_argument("--all-channels", action="store_true", dest="all_channels_enable")
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--random-state", type=int, default=42)
    parser.set_defaults(run_command=run_command)


def _configure_summary_parser(parser: argparse.ArgumentParser) -> None:
    """注册层级汇总参数。"""
    parser.add_argument("--source-dir", required=True, help="实验目录")
    parser.add_argument("--level", required=True, help="目标层级，例如 level_2")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.set_defaults(run_command=run_command)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """注册评估目标共有的模型槽位和层级参数。"""
    parser.add_argument("--source-dir", required=True, help="模型槽位目录")
    parser.add_argument("--level", required=True, help="目标层级，例如 level_1")
