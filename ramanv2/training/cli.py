"""训练 workflow 的命令行适配。"""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册训练请求的范围与输出位置参数。"""
    parser.add_argument("--dataset", default=None, help="数据集标识；默认按 profile 归属解析")
    parser.add_argument("--profile", required=True, help="常规数据集 profile 标识")
    parser.add_argument("--level", required=True, help="训练层级，如 level_1")
    parser.add_argument(
        "--parent",
        default=None,
        help="目标父类名称或索引；除首层外必填，--global-head 时留空",
    )
    parser.add_argument(
        "--global-head",
        action="store_true",
        help="训练该层全部类别的全局模型；不能与 --parent 同时使用",
    )
    parser.add_argument("--experiment-dir", default=None, help="固定实验根目录")
    parser.add_argument("--resume-run-dir", default=None, help="从指定模型槽位目录恢复训练")
    parser.set_defaults(run_command=run_command)


def _resolve_parent(value: str | None) -> tuple[int | None, str | None]:
    """将父类命令参数区分为稳定索引或类别名称。"""
    if value is None:
        return None, None
    return (int(value), None) if value.isdigit() else (None, value)


def run_command(args: argparse.Namespace) -> int:
    """构建训练请求并调用 workflow。"""
    from ramanv2.core.config import build_config
    from ramanv2.data.profiles import resolve_selection
    from ramanv2.training.workflow import TrainRequest, run_training

    parent_id, parent_name = _resolve_parent(args.parent)
    selection = resolve_selection(args.dataset, args.profile)
    spec = selection.spec
    config = build_config(
        {
            "dataset_id": spec.dataset_id,
            "profile_id": selection.profile.profile_id,
            "label_schema": selection.hierarchy.label_schema,
            "bad_bands": spec.bad_bands,
        }
    )
    request = TrainRequest(
        config=config,
        level_name=args.level,
        parent_id=parent_id,
        parent_name=parent_name,
        global_head_enable=bool(args.global_head),
        experiment_dir=args.experiment_dir,
        resume_run_dir=args.resume_run_dir,
    )
    run_training(request)
    return 0
