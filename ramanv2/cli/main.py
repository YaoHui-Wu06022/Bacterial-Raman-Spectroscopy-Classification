"""根级命令分发。"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """构建已迁移领域的根级命令解析器。"""
    parser = argparse.ArgumentParser(prog="python -m ramanv2", description="Raman 光谱分类工具")
    domains = parser.add_subparsers(dest="domain", required=True)

    import ramanv2.audit.cli
    import ramanv2.analysis.cli
    import ramanv2.data.cli
    import ramanv2.evaluation.cli
    import ramanv2.inference.cli
    import ramanv2.package_archive
    import ramanv2.training.cli
    import ramanv2.extensions.stanford_finetune.cli

    audit_parser = domains.add_parser("audit", help="执行数据审核")
    ramanv2.audit.cli.configure_parser(audit_parser)

    data_parser = domains.add_parser("data", help="构建、打包和检查常规数据集")
    ramanv2.data.cli.configure_parser(data_parser)

    train_parser = domains.add_parser("train", help="训练层级分类模型")
    ramanv2.training.cli.configure_parser(train_parser)

    stanford_parser = domains.add_parser("stanford", help="Stanford 预训练与迁移扩展")
    ramanv2.extensions.stanford_finetune.cli.configure_parser(stanford_parser)

    infer_parser = domains.add_parser("infer", help="执行独立推理")
    infer_commands = infer_parser.add_subparsers(dest="command", required=True)
    test_parser = infer_commands.add_parser("test", help="推理测试文件夹")
    ramanv2.inference.cli.configure_parser(test_parser)
    test_parser.set_defaults(run_command=ramanv2.inference.cli.run_command)

    evaluation_parser = domains.add_parser("evaluate", help="评估模型或 PCA-SVM baseline")
    ramanv2.evaluation.cli.configure_parser(evaluation_parser)

    analysis_parser = domains.add_parser("analyze", help="运行模型可解释性分析")
    ramanv2.analysis.cli.configure_parser(analysis_parser)

    zip_parser = domains.add_parser("zip", help="仅打包 ramanv2，不包含 Stanford 扩展和数据集")
    zip_parser.set_defaults(run_command=ramanv2.package_archive.run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析根级命令并调用领域命令。"""
    args = build_parser().parse_args(argv)
    return args.run_command(args)
