"""属和种前缀按当前 init 汇总，同一种跨日期合并。"""

from ramanv2.data.count import collect_taxon_counts
from ramanv2.ui.cache import cached_dataset_taxon_counts


def _write_spectra(root, genus, folder, count):
    target = root / genus / folder
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (target / f"cell{index}.arc_data").write_text("600 1\n1000 2\n", encoding="utf-8")


def test_taxon_counts_merge_species_across_dates_and_preserve_genus_groups(tmp_path):
    _write_spectra(tmp_path, "Enterococcus", "EFA20231210_01", 2)
    _write_spectra(tmp_path, "Enterococcus", "efa20231211_02", 1)
    _write_spectra(tmp_path, "Enterococcus", "ESK20231211_01", 3)
    _write_spectra(tmp_path, "Escherichia", "EC20231211_01", 4)
    (tmp_path / "Empty/EC20231211_01").mkdir(parents=True)
    (tmp_path / "Enterococcus/EFA20231210_01/notes.txt").write_text("ignored", encoding="utf-8")
    result = collect_taxon_counts(tmp_path)
    assert result["genera"] == [
        {"genus": "Enterococcus", "species_count": 2, "folder_count": 3, "file_count": 6},
        {"genus": "Escherichia", "species_count": 1, "folder_count": 1, "file_count": 4},
    ]
    assert result["species"] == [
        {"genus": "Enterococcus", "prefix": "EFA", "folder_count": 2, "file_count": 3},
        {"genus": "Enterococcus", "prefix": "ESK", "folder_count": 1, "file_count": 3},
        {"genus": "Escherichia", "prefix": "EC", "folder_count": 1, "file_count": 4},
    ]


def test_taxon_counts_handle_missing_dataset_and_refresh_with_cache_version(tmp_path):
    root = tmp_path / "init"
    assert collect_taxon_counts(root) == {"genera": [], "species": []}
    _write_spectra(root, "Escherichia", "EC20231211_01", 1)
    assert cached_dataset_taxon_counts(str(root), 0)["species"][0]["file_count"] == 1
    _write_spectra(root, "Escherichia", "EC20231211_02", 2)
    assert cached_dataset_taxon_counts(str(root), 0)["species"][0]["file_count"] == 1
    counts = cached_dataset_taxon_counts(str(root), 1)
    assert counts["species"][0]["file_count"] == 3
    assert counts["species"][0]["folder_count"] == 2


def test_training_summary_counts_species_instead_of_source_folders(tmp_path):
    from streamlit.testing.v1 import AppTest
    from ramanv2.data.runtime.index import DatasetIndex

    _write_spectra(tmp_path, "Enterococcus", "EFA20231210_01", 2)
    _write_spectra(tmp_path, "Enterococcus", "efa20231211_02", 1)
    _write_spectra(tmp_path, "Enterococcus", "ESK20231211_01", 3)
    _write_spectra(tmp_path, "Escherichia", "EC20231211_01", 4)
    index = DatasetIndex(tmp_path, load_intensity_enable=False)
    assert index.num_classes_by_level["level_2"] == 3
    script = f'''
from pathlib import Path
from ramanv2.data.runtime.index import DatasetIndex
from ramanv2.ui.pages.hierarchical_training import _render_dataset_summary
root = Path({str(tmp_path)!r})
_render_dataset_summary(DatasetIndex(root, load_intensity_enable=False), root)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("数量", "10 条"),
        ("层数", "2 层"),
        ("第1层总类别数", "2 个"),
        ("第2层总类别数", "3 个"),
    ]
