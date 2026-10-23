from pathlib import Path
import ast
from io import BytesIO
import json
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
from uuid import uuid4
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
    output = tmp_path / "data.tar.xz"

    report = colab_export.build_profile_data_archive("GN", output, project_root)

    assert report["sample_count"] == 1
    with tarfile.open(output, "r:xz") as archive:
        assert archive.getnames() == ["dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data"]
        archive.extractall(tmp_path / "unpacked")
    restored = tmp_path / "unpacked" / "dataset" / "GN" / "init" / "Enterobacter" / "ECL20240101_01" / "sample.arc_data"
    assert restored.read_text(encoding="utf-8") == (project_root / "dataset" / "GN" / "init" / "Enterobacter" / "ECL20240101_01" / "sample.arc_data").read_text(encoding="utf-8")


def test_colab_bundle_contains_data_and_source_archives(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path)
    output_dir = tmp_path / "exports" / "GN"

    report = colab_export.build_colab_bundle("GN", output_dir, project_root)

    assert report["sample_count"] == 1
    with tarfile.open(output_dir / "data.tar.xz", "r:xz") as archive:
        assert "dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data" in archive.getnames()
        assert not any(name.endswith(".npz") for name in archive.getnames())
    assert not (output_dir / "data.zip").exists()
    with ZipFile(output_dir / "ramanv2.zip") as archive:
        assert "ramanv2/module.py" in archive.namelist()


def test_export_rejects_missing_or_empty_init(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "dataset" / "GN" / "init").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="没有 \.arc_data"):
        colab_export.build_profile_data_archive("GN", tmp_path / "data.tar.xz", project_root)
    with pytest.raises(FileNotFoundError):
        colab_export.build_profile_data_archive("GN", tmp_path / "missing.tar.xz", tmp_path / "other")


def test_bundle_failure_preserves_existing_archives(tmp_path: Path, monkeypatch) -> None:
    project_root = _build_project(tmp_path)
    output_dir = tmp_path / "exports" / "GN"
    output_dir.mkdir(parents=True)
    data_archive = output_dir / "data.tar.xz"
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
    assert "DATA_ARCHIVE = PROJECT_ROOT / \"data.tar.xz\"" in source
    assert 'REQUIRED_ARCHIVES = {"ramanv2.zip", "data.tar.xz"}' in source
    assert "data.zip" not in source
    assert "prepare_data_archive" in source
    assert 'Path("dataset") / profile.dataset_name' in source
    assert "init.npz" not in source
    assert "unpack_init" not in source


def _load_notebook_data_restore():
    notebook = json.loads(Path("colab/colab_train_ramanv2.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if "def prepare_data_archive(" in "".join(cell["source"]))
    tree = ast.parse("".join(cell["source"]))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = dict(Path=Path, tempfile=tempfile, shutil=shutil, tarfile=tarfile, uuid4=uuid4)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "colab_data_restore", "exec"), namespace)
    return namespace["prepare_data_archive"]


def test_colab_restores_exported_archive_without_changing_files(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path)
    archive_path = tmp_path / "data.tar.xz"
    colab_export.build_profile_data_archive("GN", archive_path, project_root)
    colab_root = tmp_path / "colab"
    init_dir = colab_root / "dataset/GN/init"
    init_dir.mkdir(parents=True)
    (init_dir / "old.arc_data").write_bytes(b"old")
    other = colab_root / "dataset/other.arc_data"
    other.write_bytes(b"unchanged")

    _load_notebook_data_restore()(archive_path, colab_root, SimpleNamespace(dataset_name="GN", root_init="init"))

    relative = Path("dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data")
    assert (colab_root / relative).read_bytes() == (project_root / relative).read_bytes()
    assert not (init_dir / "old.arc_data").exists()
    assert other.read_bytes() == b"unchanged"
    assert not list(colab_root.glob("data_unpack_*"))
    assert not list(init_dir.parent.glob(".init.previous_*"))


@pytest.mark.parametrize("name,kind", [
    ("../escaped.arc_data", tarfile.REGTYPE),
    ("/escaped.arc_data", tarfile.REGTYPE),
    ("dataset/GN/init/link", tarfile.SYMTYPE),
    ("dataset/GN/init/link", tarfile.LNKTYPE),
])
def test_colab_rejects_unsafe_archive_and_preserves_init(tmp_path: Path, name, kind) -> None:
    archive_path = tmp_path / "unsafe.tar.xz"
    with tarfile.open(archive_path, "w:xz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind == tarfile.REGTYPE:
            member.size = 1
        else:
            member.linkname = "../../escaped.arc_data"
        archive.addfile(member, BytesIO(b"x") if kind == tarfile.REGTYPE else None)
    init_dir = tmp_path / "dataset/GN/init"
    init_dir.mkdir(parents=True)
    previous = init_dir / "old.arc_data"
    previous.write_bytes(b"previous")

    with pytest.raises(ValueError, match="非法路径或文件类型"):
        _load_notebook_data_restore()(archive_path, tmp_path, SimpleNamespace(dataset_name="GN", root_init="init"))

    assert previous.read_bytes() == b"previous"
    assert not list(tmp_path.glob("data_unpack_*"))
    assert not (tmp_path / "escaped.arc_data").exists()


def test_colab_publish_failure_restores_previous_init(tmp_path: Path, monkeypatch) -> None:
    project_root = _build_project(tmp_path)
    archive_path = tmp_path / "data.tar.xz"
    colab_export.build_profile_data_archive("GN", archive_path, project_root)
    colab_root = tmp_path / "colab"
    init_dir = colab_root / "dataset/GN/init"
    init_dir.mkdir(parents=True)
    previous = init_dir / "old.arc_data"
    previous.write_bytes(b"previous")
    rename = Path.rename

    def fail_staged_publish(self, target):
        if any(part.startswith("data_unpack_") for part in self.parts):
            raise OSError("publish failed")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_staged_publish)
    with pytest.raises(OSError, match="publish failed"):
        _load_notebook_data_restore()(archive_path, colab_root, SimpleNamespace(dataset_name="GN", root_init="init"))

    assert previous.read_bytes() == b"previous"
    assert not list(colab_root.glob("data_unpack_*"))
    assert not list(init_dir.parent.glob(".init.previous_*"))


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
