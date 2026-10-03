from pathlib import Path
import json
from zipfile import ZipFile

import pytest

from ramanv2.packaging import data_archive as colab_export
from ramanv2.common.arc_data import write_arc_data


def _build_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    init_dir = project_root / "dataset" / "GN" / "init" / "Enterobacter" / "ECL20240101_01"
    write_arc_data(init_dir / "sample.arc_data", [100.0, 101.0], [1.0, 2.0])
    (project_root / "ramanv2").mkdir(parents=True)
    (project_root / "ramanv2" / "module.py").write_text("value = 1\n", encoding="utf-8")
    return project_root


def test_profile_data_archive_contains_raw_init_arc_data(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path)
    output = tmp_path / "data.zip"

    report = colab_export.build_profile_data_archive("GN", output, project_root)

    assert report["sample_count"] == 1
    with ZipFile(output) as archive:
        assert archive.namelist() == ["dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data"]
        archive.extractall(tmp_path / "unpacked")
    restored = tmp_path / "unpacked" / "dataset" / "GN" / "init" / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    assert restored.read_text(encoding="utf-8") == (project_root / "dataset" / "GN" / "init" / "Enterobacter" / "ECL20240101_01" / "sample.arc_data").read_text(encoding="utf-8")


def test_colab_bundle_contains_data_and_source_archives(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path)
    output_dir = tmp_path / "exports" / "GN"

    report = colab_export.build_colab_bundle("GN", output_dir, project_root)

    assert report["sample_count"] == 1
    with ZipFile(output_dir / "data.zip") as archive:
        assert "dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data" in archive.namelist()
        assert not any(name.endswith(".npz") for name in archive.namelist())
    with ZipFile(output_dir / "ramanv2.zip") as archive:
        assert "ramanv2/module.py" in archive.namelist()


def test_export_rejects_missing_or_empty_init(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "dataset" / "GN" / "init").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="没有 \.arc_data"):
        colab_export.build_profile_data_archive("GN", tmp_path / "data.zip", project_root)
    with pytest.raises(FileNotFoundError):
        colab_export.build_profile_data_archive("GN", tmp_path / "missing.zip", tmp_path / "other")


def test_bundle_failure_preserves_existing_archives(tmp_path: Path, monkeypatch) -> None:
    project_root = _build_project(tmp_path)
    output_dir = tmp_path / "exports" / "GN"
    output_dir.mkdir(parents=True)
    data_archive = output_dir / "data.zip"
    package_archive = output_dir / "ramanv2.zip"
    data_archive.write_bytes(b"old-data")
    package_archive.write_bytes(b"old-package")

    def fail_package(*args, **kwargs):
        raise RuntimeError("package failed")

    monkeypatch.setattr(colab_export, "build_package_archive", fail_package)
    with pytest.raises(RuntimeError, match="package failed"):
        colab_export.build_colab_bundle("GN", output_dir, project_root)
    assert data_archive.read_bytes() == b"old-data"
    assert package_archive.read_bytes() == b"old-package"


def test_colab_notebook_auto_restores_profile_data_archive() -> None:
    notebook = json.loads(Path("colab/colab_train_ramanv2.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "DATA_ARCHIVE = PROJECT_ROOT / \"data.zip\"" in source
    assert "prepare_data_archive" in source
    assert 'Path("dataset") / profile.dataset_name' in source
    assert "init.npz" not in source
    assert "unpack_init" not in source


def test_colab_notebook_is_single_stage_and_reports_train_snapshot() -> None:
    notebook = json.loads(Path("colab/colab_train_ramanv2.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "from ramanv2.data.builders.stage import build_train" in source
    assert "from ramanv2.data.build import build_train" not in source
    assert "train_data/manifest.json" in source
    assert "evaluate_model_run" in source
    assert "run_interpret_run" in source
    assert "evaluate_model_cascade" in source
    assert "run_interpret_parent_routed" in source
    for token in ("第二阶段微调", "SECOND_STAGE", "run_stage2"):
        assert token not in source
    assert 'EXPERIMENT_DIR = ""' not in source
    assert "shutil.make_archive" in source
    assert "files.download(str(result_archive))" in source
