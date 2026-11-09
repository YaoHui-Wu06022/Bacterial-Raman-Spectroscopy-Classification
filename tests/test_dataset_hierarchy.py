import json
from pathlib import Path

from ramanv2.data.datasets import load_dataset_registry, write_dataset_config
from ramanv2.data.hierarchy_config import (
    HierarchyLevel,
    HierarchySpec,
    write_hierarchy,
)


def make_dataset(root: Path, directory_name: str, dataset_id: str, *, default: bool = False) -> Path:
    directory = root / directory_name
    (directory / "cosmic_data" / "20240101" / "A01").mkdir(parents=True)
    (directory / "cosmic_data" / "20240101" / "A01" / "a.arc_data").write_text(
        "100 1\n101 2\n", encoding="utf-8"
    )
    write_dataset_config(
        directory,
        directory_name,
        (),
        dataset_id=dataset_id,
        aliases=("legacy-" + dataset_id.lower(),),
        is_default=default,
    )
    return directory


def test_dataset_discovery_uses_directory_and_ascii_id(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    pb = make_dataset(data_root, "病原菌", "PB", default=True)
    fbb = make_dataset(data_root, "食源菌", "FBB")
    write_hierarchy(
        pb / "hierarchy.json",
        HierarchySpec(
            levels=(
                HierarchyLevel("level_1", "属", directory=True),
                HierarchyLevel("level_2", "种", directory=False),
            ),
            mappings={"level_1": {"A": "GenusA"}},
            explicit=True,
        ),
    )

    registry = load_dataset_registry(data_root)

    assert registry.default_dataset_id == "PB"
    assert registry.resolve("PB").directory.name == "病原菌"
    assert registry.resolve("病原菌").dataset_id == "PB"
    assert registry.resolve("legacy-pb").dataset_id == "PB"
    assert registry.resolve("FBB").is_flat is True
    assert registry.resolve("PB").is_flat is False
    assert registry.resolve("PB").bad_bands == ()
    assert fbb.joinpath("dataset.json").is_file()


def test_flat_hierarchy_resolves_prefix_case() -> None:
    spec = HierarchySpec.flat()
    assert spec.resolve_values("Ecoli01")["level_1"] == "Ecoli"
    assert spec.init_relative_dir("Ecoli01").as_posix() == "Ecoli"
