import numpy as np

from ramanv2.data.builders.init import _stage_profile_test_set
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.records.lineage import (
    clear_test_routes,
    load_folder_lineage,
    write_folder_lineage,
)


def _write_spectrum(path, value=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavenumbers = np.arange(10, dtype=float)
    intensities = np.linspace(value, value + 1.0, 10)
    np.savetxt(path, np.column_stack([wavenumbers, intensities]))


def test_profile_test_stage_filters_allowed_targets(tmp_path):
    master = tmp_path / "init_test"
    _write_spectrum(master / "ABC20240101_01" / "a.arc_data")
    _write_spectrum(master / "DEF20240101_01" / "b.arc_data")
    mapping = DatasetMapping.from_payload(
        {"genera": {"A": ["ABC"], "D": ["DEF"]}, "datasets": {"P": ["A", "D"]}}
    )
    temporary_roots = {}
    folder_count, file_count = _stage_profile_test_set(
        master,
        temporary_roots,
        tmp_path / "out" / "init_test",
        "P",
        mapping,
        allowed_test_targets={"ABC20240101_01"},
    )
    assert folder_count == 1
    assert file_count == 1
    assert (temporary_roots["P"] / "ABC20240101_01").is_dir()
    assert not (temporary_roots["P"] / "DEF20240101_01").exists()


def test_clear_test_routes_keeps_init_output(tmp_path):
    data_root = tmp_path / "cosmic_data"
    data_root.mkdir()
    init_path = str(tmp_path / "init" / "A" / "ABC20240101_01")
    payload = {
        "version": 1,
        "source_root": str(data_root),
        "folders": {
            "20240101/ABC01": {
                "source_id": "20240101/ABC01",
                "date": "20240101",
                "folder": "ABC01",
                "prefix": "ABC",
                "genus": "A",
                "files": {},
                "folder_status": "active",
                "outputs": {
                    "shift": None,
                    "init": init_path,
                    "profiles": {},
                    "test": str(tmp_path / "init_test" / "ABC20240101_01"),
                    "prefix_groups": [],
                },
                "history": [],
                "updated_at": None,
            }
        },
    }
    write_folder_lineage(data_root, payload)
    result = clear_test_routes(data_root, ["20240101/ABC01"])
    record = load_folder_lineage(data_root)["folders"]["20240101/ABC01"]
    assert result["count"] == 1
    assert record["outputs"]["test"] is None
    assert record["outputs"]["init"] == init_path
