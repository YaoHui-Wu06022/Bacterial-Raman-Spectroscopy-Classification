"""根级命令分发。"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """构建已迁移领域的根级命令解析器。"""
    parser = argparse.ArgumentParser(prog="python -m raman_temp", description="Raman 光谱分类工具")
    domains = parser.add_subparsers(dest="domain", required=True)

    import raman_temp.audit.cli
    import raman_temp.analysis.cli
    import raman_temp.data.cli
    import raman_temp.evaluation.cli
    import raman_temp.inference.cli
    import raman_temp.training.cli
    import raman_temp.extensions.stanford_finetune.cli

    audit_parser = domains.add_parser("audit", help="执行数据审核")
    raman_temp.audit.cli.configure_parser(audit_parser)

    data_parser = domains.add_parser("data", help="构建、打包和检查常规数据集")
    raman_temp.data.cli.configure_parser(data_parser)

    train_parser = domains.add_parser("train", help="训练层级分类模型")
    raman_temp.training.cli.configure_parser(train_parser)

    stanford_parser = domains.add_parser("stanford", help="Stanford 预训练与迁移扩展")
    raman_temp.extensions.stanford_finetune.cli.configure_parser(stanford_parser)

    infer_parser = domains.add_parser("infer", help="执行独立推理")
    infer_commands = infer_parser.add_subparsers(dest="command", required=True)
    test_parser = infer_commands.add_parser("test", help="推理测试文件夹")
    raman_temp.inference.cli.configure_parser(test_parser)
    test_parser.set_defaults(run_command=raman_temp.inference.cli.run_command)

    evaluation_parser = domains.add_parser("evaluate", help="评估模型或 PCA-SVM baseline")
    raman_temp.evaluation.cli.configure_parser(evaluation_parser)

    analysis_parser = domains.add_parser("analyze", help="运行模型可解释性分析")
    raman_temp.analysis.cli.configure_parser(analysis_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析根级命令并调用领域命令。"""
    args = build_parser().parse_args(argv)
    return args.run_command(args)
