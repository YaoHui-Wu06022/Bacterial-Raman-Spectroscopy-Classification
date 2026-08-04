"""audit 命令参数解析与工作流调用。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """注册 audit 命令及各子命令参数。"""
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("clean", help="在运行池执行完整三阶段审核")
    clean.add_argument("--dataset", default="alldata")
    clean.add_argument("--test-dataset", default="test")
    prefix = commands.add_parser("plot-prefix", help="绘制同前缀审核总览图")
    prefix.add_argument("--dataset", default="alldata")
    prefix.add_argument("--force", action="store_true", dest="force_enable")
    shift = commands.add_parser("plot-shift", help="绘制一个文件夹的平移前后对照图")
    shift.add_argument("--dataset", default="alldata")
    shift.add_argument("--folder", required=True)
    manual = commands.add_parser("manual-shift", help="对 init 文件夹追加人工平移")
    manual.add_argument("--dataset", default="alldata")
    manual.add_argument("--test-dataset", default="test")
    manual.add_argument("--folder", required=True)
    manual.add_argument("--delta", required=True, type=float)
    for command_parser in (clean, prefix, shift, manual):
        command_parser.set_defaults(run_command=run_command)


def build_parser() -> argparse.ArgumentParser:
    """构建独立调用时使用的 audit 参数解析器。"""
    parser = argparse.ArgumentParser(description="Raman 数据审核")
    configure_parser(parser)
    return parser


def read_transfer_source(manifest_path: Path, genus: str, folder: str) -> str:
    """从派生副本清单定位一个 `*t` 文件夹对应的测试菌来源。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少测试菌派生副本清单：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        sources = {
            row["source_folder"]
            for row in csv.DictReader(file)
            if row.get("target_genus") == genus and row.get("target_folder") == folder
        }
    if len(sources) != 1:
        raise ValueError(f"无法唯一确定测试菌来源：{genus}/{folder}")
    return sources.pop()


def read_transfer_target(manifest_path: Path, source_folder: str) -> tuple[str, str]:
    """从派生副本清单定位一个测试菌来源对应的 `*t` 文件夹。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少测试菌派生副本清单：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        targets = {
            (row["target_genus"], row["target_folder"])
            for row in csv.DictReader(file)
            if row.get("source_folder") == source_folder
        }
    if len(targets) != 1:
        raise ValueError(f"无法唯一确定训练副本：{source_folder}")
    return targets.pop()


def resolve_manual_targets(
    dataset_key: str,
    test_key: str,
    folder_value: str,
) -> tuple[ManualShiftTarget, ManualShiftTarget | None]:
    """解析人工平移目标及测试菌与 `*t` 副本之间的对应目标。"""
    from ramanv2.audit.plots import resolve_folder_dir
    from ramanv2.audit.shift import ManualShiftTarget
    from ramanv2.common.naming import parse_folder_prefix
    from ramanv2.data.profiles import get_dataset_dir, get_profile

    profile = get_profile(dataset_key)
    dataset_dir = get_dataset_dir(profile)
    test_profile = get_profile(test_key)
    test_dir = get_dataset_dir(test_profile)
    if profile.profile_id == "test":
        folder_dir = resolve_folder_dir(dataset_dir / profile.root_init, folder_value)
        source_folder = folder_dir.name
        genus, training_folder = read_transfer_target(dataset_dir / "test_transfer_manifest.csv", source_folder)
        training_dir = get_dataset_dir(get_profile("alldata"))
        counterpart_dir = training_dir / "init" / genus / training_folder
        target = ManualShiftTarget(folder_dir, dataset_dir / "delta.txt", ".", source_folder, parse_folder_prefix(source_folder))
        counterpart = ManualShiftTarget(counterpart_dir, training_dir / "delta.txt", genus, training_folder, parse_folder_prefix(training_folder))
        return target, counterpart
    folder_dir = resolve_folder_dir(dataset_dir / profile.root_init, folder_value)
    genus = folder_dir.parent.name
    target = ManualShiftTarget(folder_dir, dataset_dir / "delta.txt", genus, folder_dir.name, parse_folder_prefix(folder_dir.name))
    if not folder_dir.name.lower().endswith("t"):
        return target, None
    source_folder = read_transfer_source(test_dir / "test_transfer_manifest.csv", genus, folder_dir.name)
    counterpart_dir = test_dir / test_profile.root_init / source_folder
    counterpart = ManualShiftTarget(counterpart_dir, test_dir / "delta.txt", ".", source_folder, parse_folder_prefix(source_folder))
    return target, counterpart


def run_command(args: argparse.Namespace) -> int:
    """执行一个已经解析完成的 audit 子命令。"""
    if args.command == "clean":
        from ramanv2.audit.workflow import run_clean

        print(run_clean(args.dataset, args.test_dataset))
        return 0
    if args.command == "plot-prefix":
        from ramanv2.audit.plots import plot_prefix_dataset

        outputs = plot_prefix_dataset(args.dataset, force_enable=args.force_enable)
        print(f"figures={len(outputs)}")
        return 0
    if args.command == "plot-shift":
        from ramanv2.audit.plots import plot_shift_folder

        print(plot_shift_folder(args.dataset, args.folder))
        return 0
    if args.command == "manual-shift":
        from ramanv2.audit.shift import apply_manual_shift

        target, counterpart = resolve_manual_targets(args.dataset, args.test_dataset, args.folder)
        print(json.dumps(apply_manual_shift(target, args.delta, counterpart), ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"未知 audit 命令：{args.command}")


def main(argv: list[str] | None = None) -> int:
    """解析并执行一个 audit 子命令。"""
    args = build_parser().parse_args(argv)
    return args.run_command(args)
