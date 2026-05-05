import argparse
from pathlib import Path

from raman.data.build import (
    build_test,
    build_train,
    classify,
    count_dataset,
    pack_init,
    preview,
    print_results,
    unpack_init,
)
from raman.data.profiles import get_dataset_dir, get_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_SUBDIRS = (
    "init",
    "train_raw",
    "train",
    "test",
    "fig_train",
    "fig_test",
    "test_raw",
    "fig_init",
)


def ensure_dataset_layout(profile):
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_DATASET_SUBDIRS:
        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
    return dataset_dir


def run_pack(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    pack_init(
        dataset_dir / profile.root_init,
        dataset_dir / profile.root_init_pack,
        verbose=not args.quiet,
    )


def run_classify(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    classify(profile, dataset_dir)


def run_unpack(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    unpack_init(
        dataset_dir / profile.root_init_pack,
        dataset_dir / profile.root_init,
        verbose=not args.quiet,
    )


def run_train(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    build_train(profile, dataset_dir)


def run_preview(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    preview(profile, dataset_dir)


def run_test(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    build_test(profile, dataset_dir)


def run_count(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    target_dir = dataset_dir / (args.subdir or profile.count_root)
    tree, total_files = count_dataset(target_dir)
    print_results(tree, total_files)


def build_parser():
    parser = argparse.ArgumentParser(description="Unified data prep entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, handler, help_text in (
        ("pack", run_pack, "Pack init into init.npz"),
        ("unpack", run_unpack, "Unpack init.npz into init"),
        ("classify", run_classify, "Classify init into train_raw"),
        ("preview", run_preview, "Generate per-folder preview figures from init"),
        ("train", run_train, "Build train from train_raw"),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
