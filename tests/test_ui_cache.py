from pathlib import Path

from ramanv2.ui.cache import cached_directory_names


def test_directory_cache_uses_version_for_invalidation(tmp_path: Path) -> None:
    first = tmp_path / "20240101"
    first.mkdir()

    assert cached_directory_names(str(tmp_path), 0) == ("20240101",)

    (tmp_path / "20240102").mkdir()
    assert cached_directory_names(str(tmp_path), 0) == ("20240101",)
    assert cached_directory_names(str(tmp_path), 1) == ("20240101", "20240102")
