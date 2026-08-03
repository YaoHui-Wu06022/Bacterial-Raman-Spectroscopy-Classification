"""训练 workflow 的命令行适配。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册训练请求的范围与输出位置参数。"""
    parser.add_argument("--profile", required=True, help="常规数据集 profile 标识")
    parser.add_argument("--level", required=True, help="训练层级，如 level_1")
    parser.add_argument("--parent", default=None, help="仅训练指定父类的子模型，可为名称或索引")
    parser.add_argument("--global", action="store_true", dest="global_enable", help="训练该层唯一的全局模型")
    parser.add_argument("--experiment-dir", default=None, help="固定实验根目录")
    parser.add_argument("--run-name", default=None, help="本次训练的 run 目录名")
    parser.add_argument("--resume-run-dir", default=None, help="从指定 run 目录恢复训练")
    parser.set_defaults(run_command=run_command)


def build_parser() -> argparse.ArgumentParser:
    """构建独立执行 training 包时使用的参数解析器。"""
    parser = argparse.ArgumentParser(description="Raman 层级模型训练")
    configure_parser(parser)
    return parser


def _resolve_parent(value: str | None) -> tuple[int | None, str | None]:
    """将父类命令参数区分为稳定索引或类别名称。"""
    if value is None:
        return None, None
    return (int(value), None) if value.isdigit() else (None, value)


def run_command(args: argparse.Namespace) -> int:
    """构建训练请求并调用 workflow。"""
    from raman_temp.core.config import build_config
    from raman_temp.training.workflow import TrainRequest, run_training

    only_parent, only_parent_name = _resolve_parent(args.parent)
    config = build_config({"profile_id": args.profile})
    request = TrainRequest(
        config=config,
        level_name=args.level,
        only_parent=only_parent,
        only_parent_name=only_parent_name,
        train_per_parent_enable=not args.global_enable,
        experiment_dir=args.experiment_dir,
        run_name=args.run_name,
        resume_run_dir=args.resume_run_dir,
    )
    run_training(request)
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析并执行一次训练请求。"""
    args = build_parser().parse_args(argv)
    return args.run_command(args)
