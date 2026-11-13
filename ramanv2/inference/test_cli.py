"""test 文件夹独立推理命令参数解析。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册独立推理命令参数。"""
    parser.add_argument("--source-dir", required=True, help="实验目录")
    parser.add_argument("--level", required=True, help="目标层级，例如 level_1")
    parser.add_argument("--input-dir", default=None, help="覆盖 profile 默认测试目录")
    parser.add_argument("--one-dir", default=None, help="只预测输入根下的一个文件夹")
    parser.add_argument("--top-k", type=int, default=3, help="逐谱输出的 top-k 数量")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--no-evaluate", action="store_true", help="不按测试文件夹前缀评估")
    parser.add_argument("--plot-train-mean", action="store_true", help="绘图时叠加训练类别均值")


def run_command(args: argparse.Namespace) -> int:
    """执行已经解析完成的独立推理请求。"""
    from ramanv2.inference.test_inference import run_independent_inference

    run_independent_inference(
        args.source_dir,
        args.level,
        input_dir=args.input_dir,
        one_dir=args.one_dir,
        top_k=args.top_k,
        device="cpu" if args.cpu else None,
        evaluate_enable=not args.no_evaluate,
        plot_train_mean_enable=args.plot_train_mean,
    )
    return 0
