import argparse

from raman.data.build import build_test, build_train
from raman.data.io import pack_init, unpack_init
from raman.data.count import count_dataset, print_results
from raman.tool.dataset import resolve_dataset


def run_pack(args):
    """执行 init 目录打包"""
    profile, dataset_dir = resolve_dataset(args.dataset, create=True)
    pack_init(
        dataset_dir / profile.root_init,
        dataset_dir / profile.root_init_pack,
        verbose=not args.quiet,
    )


def run_unpack(args):
    """执行 init.npz 解包"""
    profile, dataset_dir = resolve_dataset(args.dataset, create=True)
    unpack_init(
        dataset_dir / profile.root_init_pack,
        dataset_dir / profile.root_init,
        verbose=not args.quiet,
    )


def run_train(args):
    """从 init 直接构建最终训练集"""
    profile, dataset_dir = resolve_dataset(args.dataset, create=True)
    build_train(profile, dataset_dir)


def run_test(args):
    """从 init_test 构建已预处理的独立测试集"""
    profile, dataset_dir = resolve_dataset(args.dataset, create=True)
    build_test(profile, dataset_dir)


def run_count(args):
    """统计某个数据阶段下的光谱文件数量"""
    _, dataset_dir = resolve_dataset(args.dataset, create=True)
    target_dir = dataset_dir / (args.subdir or "train")
    tree, total_files = count_dataset(target_dir)
    print_results(tree, total_files)


def build_parser():
    """构造 raman.data 命令行参数解析器"""
    parser = argparse.ArgumentParser(description="数据集预处理和统计工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, handler, help_text in (
        ("pack", run_pack, "把 init 打包成 init.npz"),
        ("unpack", run_unpack, "把 init.npz 解包回 init"),
        ("train", run_train, "从 init 直接构建 train"),
        ("test", run_test, "从 init_test 构建 test"),
        ("count", run_count, "统计指定数据阶段的 arc_data 数量"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("dataset", help="数据集 profile id、名称或 dataset 下的文件夹名")
        if command in {"pack", "unpack"}:
            sub.add_argument("--quiet", action="store_true", help="减少打包/解包过程输出")
        if command == "count":
            sub.add_argument("--subdir", default=None, help="指定要统计的子目录，默认 train")
        sub.set_defaults(func=handler)

    return parser


def main(argv=None):
    """运行数据处理 CLI"""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
