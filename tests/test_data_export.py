from pathlib import Path
import ast
import json
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
from uuid import uuid4


from ramanv2.common.arc_data import write_arc_data
from ramanv2.packaging import data_archive


NOTEBOOK_PATH = Path("colab/colab_train_ramanv2.ipynb")


def _build_project(tmp_path: Path, with_test_enable: bool = False) -> Path:
    project_root = tmp_path / "project"
    init_dir = project_root / "dataset" / "GN" / "init" / "Enterobacter" / "ECL20240101_01"
    write_arc_data(init_dir / "sample.arc_data", [100.0, 101.0], [1.0, 2.0])
    if with_test_enable:
        # 只打包 profile 测试集；全局测试母本用于验证不会被写进数据包。
        test_init_dir = project_root / "dataset" / "init_test" / "ECL20240202_01"
        write_arc_data(test_init_dir / "sample.arc_data", [100.0, 101.0], [3.0, 4.0])
        profile_test_dir = project_root / "dataset" / "GN" / "init_test" / "ECL20240202_01"
        write_arc_data(profile_test_dir / "sample.arc_data", [100.0, 101.0], [3.0, 4.0])
    (project_root / "ramanv2").mkdir(parents=True)
    (project_root / "ramanv2" / "module.py").write_text("value = 1\n", encoding="utf-8")
    return project_root




def _load_notebook_data_restore():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if "def prepare_data_archive(" in "".join(cell["source"]))
    tree = ast.parse("".join(cell["source"]))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = dict(Path=Path, tempfile=tempfile, shutil=shutil, tarfile=tarfile, uuid4=uuid4)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "data_restore", "exec"), namespace)
    return namespace["prepare_data_archive"]


def _profile() -> SimpleNamespace:
    return SimpleNamespace(dataset_name="GN", root_init="init", root_init_test="init_test")




def test_profile_data_archive_includes_independent_test_init(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path, with_test_enable=True)
    output = tmp_path / "data.tar.xz"

    report = data_archive.build_profile_data_archive("GN", output, project_root)

    assert report["sample_counts"] == {
        "init": 1,
        "init_test": 1,
    }
    assert report["sample_count"] == 2
    with tarfile.open(output, "r:xz") as archive:
        assert archive.getnames() == [
            "dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data",
            "dataset/GN/init_test/ECL20240202_01/sample.arc_data",
        ]














def test_notebook_restores_exported_archive_without_changing_files(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path, with_test_enable=True)
    archive_path = tmp_path / "data.tar.xz"
    data_archive.build_profile_data_archive("GN", archive_path, project_root)
    target_root = tmp_path / "remote"
    init_dir = target_root / "dataset/GN/init"
    init_dir.mkdir(parents=True)
    (init_dir / "old.arc_data").write_bytes(b"old")
    other = target_root / "dataset/other.arc_data"
    other.write_bytes(b"unchanged")

    _load_notebook_data_restore()(archive_path, target_root, _profile())

    init_relative = Path("dataset/GN/init/Enterobacter/ECL20240101_01/sample.arc_data")
    profile_test_relative = Path("dataset/GN/init_test/ECL20240202_01/sample.arc_data")
    assert (target_root / init_relative).read_bytes() == (project_root / init_relative).read_bytes()
    assert (target_root / profile_test_relative).read_bytes() == (project_root / profile_test_relative).read_bytes()
    assert not (target_root / "dataset/init_test").exists()
    assert not (init_dir / "old.arc_data").exists()
    assert other.read_bytes() == b"unchanged"
    assert not list(target_root.glob("data_unpack_*"))
    assert not list(init_dir.parent.glob(".init.previous_*"))
    assert not list(target_root.rglob(".init_test.previous_*"))






