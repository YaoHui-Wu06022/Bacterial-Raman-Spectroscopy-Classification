from pathlib import Path

from ramanv2.inference.folder_inference import (
    INFERENCE_SAMPLES_PER_FOLDER,
    _sample_folder_spectra,
)
from ramanv2.inference.test_evaluation import _scan_test_set


def make_items(file_names: list[str]) -> list[tuple[str, Path]]:
    return [(name, Path(name)) for name in file_names]


def test_sampling_limits_to_one_per_cell() -> None:
    names = [
        f"cell{cell}_Area01_{index:03d}.arc_data"
        for cell in range(1, 21)
        for index in range(2)
    ]

    sampled = _sample_folder_spectra(make_items(names))

    assert len(sampled) == INFERENCE_SAMPLES_PER_FOLDER
    cells = [name.split("_", 1)[0] for name, _ in sampled]
    assert len(set(cells)) == INFERENCE_SAMPLES_PER_FOLDER


    names = [f"cell{cell}_Area01_000.arc_data" for cell in range(1, 6)]
    names.append("cell1_Area01_001.arc_data")

    sampled = _sample_folder_spectra(make_items(names))

    assert len(sampled) == 5
    cells = [name.split("_", 1)[0] for name, _ in sampled]
    assert cells == ["cell1", "cell2", "cell3", "cell4", "cell5"]


    names = [f"sample{index}.arc_data" for index in range(3)]
    names.extend(f"cell{cell}_Area01_000.arc_data" for cell in (1, 2))

    sampled = _sample_folder_spectra(make_items(names))

    assert len(sampled) == 5
    assert len({name for name, _ in sampled}) == 5


def write_files(folder: Path, names: list[str]) -> None:
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_text("", encoding="utf-8")


def test_scan_test_set_samples_each_folder(tmp_path: Path) -> None:
    write_files(
        tmp_path / "B01",
        [
            f"cell{cell}_Area01_{index:03d}.arc_data"
            for cell in range(1, 16)
            for index in range(2)
        ],
    )
    write_files(
        tmp_path / "A01",
        [f"cell{cell}_Area01_000.arc_data" for cell in range(1, 4)],
    )

    folder_names, sample_paths, sample_folders = _scan_test_set(tmp_path)

    assert folder_names == ["A01", "B01"]
    assert len(sample_paths) == INFERENCE_SAMPLES_PER_FOLDER + 3
    assert sample_folders.count("A01") == 3
    assert sample_folders.count("B01") == INFERENCE_SAMPLES_PER_FOLDER
    sampled_cells = {
        folder: {path.name.split("_", 1)[0] for path, owner in zip(sample_paths, sample_folders) if owner == folder}
        for folder in folder_names
    }
    for folder, cells in sampled_cells.items():
        assert len(cells) == sample_folders.count(folder)
