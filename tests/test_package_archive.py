from pathlib import Path
from zipfile import ZipFile

from ramanv2.packaging.source_archive import build_package_archive


def test_package_archive_excludes_cache_but_keeps_regular_extensions(tmp_path: Path) -> None:
    package_dir = tmp_path / "ramanv2"
    (package_dir / "core").mkdir(parents=True)
    (package_dir / "extensions").mkdir(parents=True)
    (package_dir / "__pycache__").mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "core" / "config.py").write_text("CONFIG = {}\n", encoding="utf-8")
    (package_dir / "extensions" / "example.py").write_text(
        "", encoding="utf-8"
    )
    (package_dir / "__pycache__" / "config.pyc").write_bytes(b"cache")

    archive_path = build_package_archive(package_dir, tmp_path / "ramanv2.zip")

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert names == {
        "ramanv2/__init__.py",
        "ramanv2/core/config.py",
        "ramanv2/extensions/example.py",
    }
