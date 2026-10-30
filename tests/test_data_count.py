import numpy as np

from ramanv2.common.arc_data import write_arc_data
from ramanv2.data.count import summarize_dataset


def test_summarize_dataset_counts_unique_species_across_source_folders(tmp_path):
    axis = np.linspace(600.0, 1800.0, 8)
    values = np.ones(axis.shape)
    for folder in ("AB20240101_01", "AB20240102_01"):
        write_arc_data(
            tmp_path / "Acinetobacter" / folder / "spectrum.arc_data",
            axis,
            values,
        )

    summary = summarize_dataset(tmp_path)

    assert summary["genus_count"] == 1
    assert summary["species_count"] == 1
    assert summary["folder_count"] == 2
    assert summary["file_count"] == 2
