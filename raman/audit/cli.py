"""新版 audit 命令入口。"""

from __future__ import annotations

import argparse

from raman.audit.plots import plot_prefix_dataset, plot_shift_folder
from raman.audit.workflow import (
    rollback_data_driven_shift,
    run_cleaning_pipeline,
)
from raman.audit.test_pool import rebuild_training_test_copies, sync_back, transfer_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="连续 Raman 数据清洗")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("clean", "全量测试菌参与的连续清洗与 *t 重建"),
        ("rollback-shift", "撤销当前数据集根目录 delta.txt 对应的平移"),
        ("preview", "增量绘制包含 t 文件夹的 fig_init 审核图"),
        ("plot-shift", "绘制一个自动平移文件夹的前后对比图"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        if command != "clean":
            child.add_argument("--dataset", default="cos", help="主数据集 profile id")
        if command == "plot-shift":
            child.add_argument("--folder", required=True, help="属/小文件夹，或唯一小文件夹名")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rollback-shift":
        print(f"rolled_back_folders={rollback_data_driven_shift(args.dataset)}")
        return 0
    if args.command == "preview":
        outputs = plot_prefix_dataset(args.dataset)
        print(f"figures={len(outputs)}")
        return 0
    if args.command == "plot-shift":
        print(plot_shift_folder(args.dataset, args.folder))
        return 0
    if args.command == "clean":
        copied, skipped = transfer_all()
        try:
            out_dir = run_cleaning_pipeline()
        finally:
            synced, missing = sync_back()
        rebuild_training_test_copies()
        print(f"test_pool_copied={copied}, skipped={skipped}")
        print(f"test_pool_synced={synced}, missing={missing}")
        print(out_dir)
        return 0
    raise RuntimeError(f"未处理的 audit 命令：{args.command}")
