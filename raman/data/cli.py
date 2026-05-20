import argparse
from pathlib import Path

from raman.data.build import build_test, build_train, preview
from raman.data.archive import pack_init, unpack_init
from raman.data.count import count_dataset, print_results
from raman.data.profiles import get_dataset_dir, get_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def resolve_dataset_dir(profile):
    """只解析数据集根目录，不提前创建所有阶段目录"""
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def run_pack(args):
    """执行 init 目录打包"""
    profile = get_profile(args.dataset)
    dataset_dir = resolve_dataset_dir(profile)
    pack_init(
        dataset_dir / profile.root_init,
        dataset_dir / profile.root_init_pack,
        verbose=not args.quiet,
    )


def run_unpack(args):
    """执行 init.npz 解包"""
    profile = get_profile(args.dataset)
    dataset_dir = resolve_dataset_dir(profile)
    unpack_init(
        dataset_dir / profile.root_init_pack,
        dataset_dir / profile.root_init,
        verbose=not args.quiet,
    )


def run_train(args):
    """复用或生成 train_raw，再构建最终训练集"""
    profile = get_profile(args.dataset)
    dataset_dir = resolve_dataset_dir(profile)
    build_train(profile, dataset_dir)


def run_preview(args):
    """从 init 生成预览图"""
    profile = get_profile(args.dataset)
    dataset_dir = resolve_dataset_dir(profile)
    preview(profile, dataset_dir)


def run_test(args):
    """构建测试集预处理结果"""
    profile = get_profile(args.dataset)
    dataset_dir = resolve_dataset_dir(profile)
    build_test(profile, dataset_dir)


def run_count(args):
    """统计某个数据阶段下的光谱文件数量"""
    profile = get_profile(args.dataset)
    dataset_dir = resolve_dataset_dir(profile)
    target_dir = dataset_dir / (args.subdir or "train")
    tree, total_files = count_dataset(target_dir)
    print_results(tree, total_files)


def build_parser():
    """构造 raman.data 命令行参数解析器"""
    parser = argparse.ArgumentParser(description="Unified data prep entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, handler, help_text in (
        ("pack", run_pack, "Pack init into init.npz"),
        ("unpack", run_unpack, "Unpack init.npz into init"),
        ("preview", run_preview, "Generate per-folder preview figures from init"),
        ("train", run_train, "Build reusable train_raw, then build train"),
        ("test", run_test, "Build test from test_raw"),
        ("count", run_count, "Count arc_data files in a dataset subdir"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("dataset", help="Dataset name, such as 细菌 / 耐药菌 / 厌氧菌")
        if command in {"pack", "unpack"}:
            sub.add_argument("--quiet", action="store_true")
        if command == "count":
            sub.add_argument("--subdir", default=None, help="Override counted subdir")
        sub.set_defaults(func=handler)

    return parser


def main(argv=None):
    """运行数据处理 CLI"""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
