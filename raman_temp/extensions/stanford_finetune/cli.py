"""Stanford 扩展命令参数与延迟分发。"""

from __future__ import annotations

import argparse
from dataclasses import replace


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册 Stanford 数据准备、预训练和迁移子命令。"""
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare", "准备 Stanford init 与预训练 train"),
        ("pretrain", "训练 Stanford 30 类预训练模型"),
        ("transfer", "迁移 Stanford 预训练模型到常规 profile"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(run_command=run_command)
    commands.choices["transfer"].add_argument("--target-profile", default=None)


def run_command(args: argparse.Namespace) -> int:
    """执行一个已解析 Stanford 子命令。"""
    from raman_temp.extensions.stanford_finetune.config import (
        PREPARE_CONFIG,
        PRETRAIN_CONFIG,
        TRANSFER_CONFIG,
    )

    if args.command == "prepare":
        from raman_temp.core.config import Config
        from raman_temp.extensions.stanford_finetune.dataset import prepare_stanford_dataset
        from raman_temp.extensions.stanford_finetune.pretrain import build_pretrain_config

        config = build_pretrain_config(Config())
        print(prepare_stanford_dataset(PREPARE_CONFIG, config.input))
        return 0
    if args.command == "pretrain":
        from raman_temp.extensions.stanford_finetune.pretrain import run_pretrain

        print(run_pretrain(PRETRAIN_CONFIG))
        return 0
    if args.command == "transfer":
        from raman_temp.extensions.stanford_finetune.transfer import run_transfer

        config = TRANSFER_CONFIG
        if args.target_profile is not None:
            config = replace(config, target_profile=args.target_profile)
        print(run_transfer(config))
        return 0
    raise RuntimeError(f"未知 Stanford 命令：{args.command}")
