"""独立推理命令参数解析。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册独立推理命令参数。"""
    parser.add_argument("--source-dir", required=True, help="实验目录或单个 run 目录")
    parser.add_argument("--level", required=True, help="目标层级，例如 level_1")
    parser.add_argument("--input-dir", default=None, help="覆盖 profile 默认测试目录")
    parser.add_argument("--one-dir", default=None, help="只预测输入根下的一个文件夹")
    parser.add_argument("--top-k", type=int, default=3, help="逐谱输出的 top-k 数量")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--no-evaluate", action="store_true", help="不按测试文件夹前缀评估")
    parser.add_argument("--plot-train-mean", action="store_true", help="绘图时叠加训练类别均值")
    parser.add_argument("--skip-transferred", action="store_true", help="跳过迁移到训练集的测试谱")
    parser.add_argument("--transfer-manifest", default=None, help="迁移样本 CSV 清单")


def build_parser() -> argparse.ArgumentParser:
    """构建独立调用时使用的参数解析器。"""
    parser = argparse.ArgumentParser(description="Run independent Raman inference")
    configure_parser(parser)
    return parser


def run_command(args: argparse.Namespace) -> int:
    """执行已经解析完成的独立推理请求。"""
    from raman_temp.inference.runner import run_independent_inference

    run_independent_inference(
        args.source_dir,
        args.level,
        input_dir=args.input_dir,
        one_dir=args.one_dir,
        top_k=args.top_k,
        device="cpu" if args.cpu else None,
        evaluate_enable=not args.no_evaluate,
        plot_train_mean_enable=args.plot_train_mean,
        skip_transferred_enable=args.skip_transferred,
        transfer_manifest_path=args.transfer_manifest,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析并执行独立推理命令。"""
    args = build_parser().parse_args(argv)
    return run_command(args)
