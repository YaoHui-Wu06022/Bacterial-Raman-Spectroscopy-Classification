from pathlib import Path

import pytest

from raman_temp.data.cli import build_parser
from raman_temp.data.count import count_dataset
from raman_temp.data.io import PackedArcDataset, pack_init, unpack_init, write_arc_data
from raman_temp.data.plot import plot_train
from raman_temp.data.profiles import DatasetProfile


def test_data_cli_parses_all_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["build", "train", "--profile", "GN"]).build_target == "train"
    assert parser.parse_args(["build", "test", "--profile", "GN"]).build_target == "test"
    assert parser.parse_args(["pack", "--profile", "GN"]).command == "pack"
    assert parser.parse_args(["unpack", "--profile", "GN"]).command == "unpack"
    assert parser.parse_args(["count", "--profile", "GN", "--stage", "test"]).stage == "test"
    assert parser.parse_args(["plot", "--profile", "GN"]).command == "plot"


def test_pack_unpack_and_count_preserve_relative_samples(tmp_path: Path) -> None:
    init_dir = tmp_path / "init"
    write_arc_data(init_dir / "genus" / "sample" / "a.arc_data", [600, 601], [1, 2])
    write_arc_data(init_dir / "genus" / "sample" / "b.arc_data", [600, 601], [3, 4])
    packed_path = tmp_path / "init.npz"

    pack_init(init_dir, packed_path, is_verbose=False)
    assert len(PackedArcDataset(packed_path)) == 2
    tree, total_files = count_dataset(init_dir)
    assert total_files == 2
    assert tree["genus"]["sample"]["__count__"] == 2

    restored_dir = tmp_path / "restored"
    unpack_init(packed_path, restored_dir, is_verbose=False)
    assert sorted(path.relative_to(restored_dir).as_posix() for path in restored_dir.rglob("*.arc_data")) == [
        "genus/sample/a.arc_data",
        "genus/sample/b.arc_data",
    ]
    with pytest.raises(FileExistsError):
        unpack_init(packed_path, restored_dir, is_verbose=False)


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
