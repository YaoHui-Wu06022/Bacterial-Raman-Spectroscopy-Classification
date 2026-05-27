from __future__ import annotations

import argparse

from raman.shift.core import apply_shift, plot_prefix_dataset, plot_shift_folder, resolve_dataset


def run_apply(args) -> None:
    """执行单个小文件夹平移"""
    row, changed = apply_shift(args.dataset, args.folder, args.delta)
    paths = resolve_dataset(args.dataset)
    print(f"Dataset: {paths.dataset_dir}")
    print(f"Folder: {row['genus']}/{row['folder']}")
    print(f"Files changed: {changed}")
    print(f"delta: {row['delta']}")
    print(f"Delta file: {paths.delta_path}")


def run_preview(args) -> None:
    """输出同前缀 raw 和标准化后中位谱总览图"""
    outputs = plot_prefix_dataset(args.dataset)
    paths = resolve_dataset(args.dataset)
    print(f"Dataset: {paths.dataset_dir}")
    print(f"Output root: {paths.output_dir}")
    print(f"Figures: {len(outputs)}")


def run_plot_shift(args) -> None:
    """输出单文件夹平移前后 raw 对比图"""
    out_path = plot_shift_folder(args.dataset, args.folder)
    paths = resolve_dataset(args.dataset)
    print(f"Dataset: {paths.dataset_dir}")
    print(f"Figure: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    """构建 raman.shift 命令行入口"""
    parser = argparse.ArgumentParser(description="手动波数平移和 init 中位谱预览工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 举例: python -m raman.shift apply cos --folder Raoultella/RAL02 --delta -3
    apply = subparsers.add_parser("apply", help="按指定增量平移一个 init 小文件夹")
    apply.add_argument("dataset", help="数据集 profile id、名称或 dataset 下的文件夹名")
    apply.add_argument("--folder", required=True, help="属/小文件夹，或唯一小文件夹名，例如 Burkholderia/BCC01")
    apply.add_argument("--delta", required=True, type=float, help="平移增量，单位 cm-1，右移为正，左移为负")
    apply.set_defaults(func=run_apply)

    preview = subparsers.add_parser("preview", help="绘制每个属内同前缀小文件夹的 raw 和标准化后中位谱总览图")
    preview.add_argument("dataset", help="数据集 profile id、名称或 dataset 下的文件夹名")
    preview.set_defaults(func=run_preview)

    plot_shift = subparsers.add_parser("plot-shift", help="绘制单个已平移文件夹的平移前后 raw 对比图")
    plot_shift.add_argument("dataset", help="数据集 profile id、名称或 dataset 下的文件夹名")
    plot_shift.add_argument("--folder", required=True, help="属/小文件夹，或唯一小文件夹名，例如 Burkholderia/BCC01")
    plot_shift.set_defaults(func=run_plot_shift)

    return parser


def main(argv=None) -> None:
    """运行 raman.shift CLI"""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
