import argparse
from pathlib import Path

from dataset_process.pipeline import (
    classify_dataset,
    count_dataset,
    pack_dataset_init,
    preview_init_dataset,
    preprocess_test_dataset,
    preprocess_train_dataset,
    print_results,
    unpack_dataset_init,
)
from dataset_process.profiles import get_dataset_dir, get_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_SUBDIRS = (
    "dataset_init",
    "dataset_raw",
    "dataset_train",
    "dataset_test",
    "dataset_train_fig",
    "dataset_test_fig",
    "测试菌",
)


def ensure_dataset_layout(profile):
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_DATASET_SUBDIRS:
        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
    return dataset_dir


def run_pack_init(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    pack_dataset_init(
        dataset_dir / profile.root_init,
        dataset_dir / profile.root_init_pack,
        verbose=not args.quiet,
    )


def run_classify(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    classify_dataset(profile, dataset_dir)


def run_unpack_init(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    unpack_dataset_init(
        dataset_dir / profile.root_init_pack,
        dataset_dir / profile.root_init,
        verbose=not args.quiet,
    )


def run_preprocess_train(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    preprocess_train_dataset(profile, dataset_dir)


def run_preview_init(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    preview_init_dataset(profile, dataset_dir)


def run_preprocess_test(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    preprocess_test_dataset(profile, dataset_dir)


def run_count(args):
    profile = get_profile(args.dataset)
    dataset_dir = ensure_dataset_layout(profile)
    target_dir = dataset_dir / (args.subdir or profile.count_root)
    tree, total_files = count_dataset(target_dir)
    print_results(tree, total_files)


def build_parser():
    parser = argparse.ArgumentParser(description="Unified dataset processing entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, handler, help_text in (
        ("pack-init", run_pack_init, "Pack dataset_init into dataset_init.npz"),
        ("unpack-init", run_unpack_init, "Unpack dataset_init.npz into dataset_init"),
        ("classify", run_classify, "Classify dataset_init into dataset_raw"),
        (
            "preview-init",
            run_preview_init,
            "Generate per-folder preview figures from dataset_init",
        ),
        ("preprocess-train", run_preprocess_train, "Build dataset_train from dataset_raw"),
        ("preprocess-test", run_preprocess_test, "Build dataset_test from 测试菌"),
        ("count", run_count, "Count arc_data files in a dataset subdir"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("dataset", help="Dataset name, such as 细菌 / 耐药菌 / 厌氧菌")
        if command in {"pack-init", "unpack-init"}:
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
