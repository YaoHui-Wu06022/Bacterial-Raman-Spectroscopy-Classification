"""新版 audit 命令入口。"""

from __future__ import annotations

import argparse
import json

from raman.audit.manual_shift import apply_manual_shift
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
        ("manual-shift", "对一个最终文件夹追加人工平移"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--dataset", default="alldata", help="主数据集 profile id")
        if command == "clean":
            child.add_argument("--test-dataset", default="test", help="测试菌 profile id")
        if command == "plot-shift":
            child.add_argument("--folder", required=True, help="属/小文件夹，或唯一小文件夹名")
        if command == "manual-shift":
            child.add_argument("--folder", required=True, help="属/文件夹，例如 Serratia/SLI01")
            child.add_argument("--delta", required=True, type=float, help="追加平移量，单位 cm^-1")
            child.add_argument("--test-dataset", default="test", help="测试菌 profile id")
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
    if args.command == "manual-shift":
        result = apply_manual_shift(
            folder=args.folder,
            delta=args.delta,
            dataset_key=args.dataset,
            test_key=args.test_dataset,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "clean":
        copied, skipped = transfer_all(args.dataset, args.test_dataset)
        try:
            out_dir = run_cleaning_pipeline(args.dataset, args.test_dataset)
        finally:
            synced, missing = sync_back(args.dataset, args.test_dataset)
        rebuild_training_test_copies(args.dataset)
        outputs = plot_prefix_dataset(args.dataset, force=True)
        print(f"test_pool_copied={copied}, skipped={skipped}")
        print(f"test_pool_synced={synced}, missing={missing}")
        print(f"figures={len(outputs)}")
        print(out_dir)
        return 0
    raise RuntimeError(f"未处理的 audit 命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
