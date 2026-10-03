from pathlib import Path

from ramanv2.data.cli import build_parser
from ramanv2.data.count import count_dataset
from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.plots.train import plot_train
from ramanv2.data.profiles import DatasetProfile


def test_data_cli_parses_all_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["build", "train", "--profile", "GN"]).build_target == "train"
    assert parser.parse_args(["build", "test", "--profile", "GN"]).build_target == "test"
    for command in ("pack", "unpack"):
        try:
            parser.parse_args([command, "--profile", "GN"])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"旧命令仍被注册：{command}")
    assert parser.parse_args(["count", "--profile", "GN", "--stage", "test"]).stage == "test"
    assert parser.parse_args(["plot", "--profile", "GN"]).command == "plot"


def test_directory_init_and_count_preserve_relative_samples(tmp_path: Path) -> None:
    init_dir = tmp_path / "init"
    write_arc_data(init_dir / "genus" / "sample" / "a.arc_data", [600, 601], [1, 2])
    write_arc_data(init_dir / "genus" / "sample" / "b.arc_data", [600, 601], [3, 4])
    tree, total_files = count_dataset(init_dir)
    assert total_files == 2
    assert tree["genus"]["sample"]["__count__"] == 2


def test_plot_train_writes_leaf_and_hierarchy_figures(tmp_path: Path) -> None:
    train_dir = tmp_path / "train" / "genus" / "species"
    wavenumbers = list(range(600, 620))
    for index in range(3):
        write_arc_data(
            train_dir / f"sample_{index}.arc_data",
            wavenumbers,
            [value + index for value in range(20)],
        )

    profile = DatasetProfile("temp", "temp")
    figure_dir = plot_train(profile, tmp_path)

    assert (figure_dir / "genus" / "species.png").is_file()
    assert (figure_dir / "_hierarchy_mean" / "level_1" / "genus.png").is_file()
    assert (figure_dir / "_hierarchy_mean" / "summary" / "level_2.png").is_file()
