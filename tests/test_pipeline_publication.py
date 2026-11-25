import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from ramanv2.common import file_ops
from ramanv2.common.file_ops import write_json
from ramanv2.data.arc_data import write_arc_data
from ramanv2.data.dataset_mapping import DatasetMapping
from ramanv2.data.datasets import DatasetSpec
from ramanv2.data.hierarchy_spec import HierarchyLevel, HierarchySpec
from ramanv2.pipeline import db, init_builder, supplement
from ramanv2.pipeline.lineage import initialize_folder_lineage, load_folder_lineage
from ramanv2.pipeline.state import PipelineState, load_pipeline_state, pipeline_state_path_for_spec, write_pipeline_state


@pytest.fixture
def init_paths(tmp_path):
    date = "20240101"
    hierarchy = HierarchySpec(
        (HierarchyLevel("level_1", "group", True), HierarchyLevel("level_2", "prefix")),
        {"level_1": {"A": "G", "B": "G"}, "level_2": {"A": "A", "B": "B"}},
        explicit=True,
    )
    spec = DatasetSpec("SAMPLE", "Sample", tmp_path, tmp_path / "\u5149\u8c31", (), hierarchy)
    axis = np.linspace(400, 1800, 16)
    for root in (spec.cosmic_dir, spec.shift_dir):
        for group in ("A1", "B1"):
            write_arc_data(root / date / group / "cell1.arc_data", axis, axis)
    write_json(spec.shift_dir / date / "shift" / "affine.json", {"status": "manual_approved"})
    extra = tmp_path / "extra"
    write_arc_data(extra / "cell1.arc_data", axis, axis)
    plan = supplement.SupplementPlan(
        groups=(supplement.SupplementGroup("extra", "A", "G", extra, Path("G/A20230101_01"), 1),),
        errors=(),
        unbacked_init_folders=(),
    )
    for root in (spec.init_dir, spec.init_test_dir):
        root.mkdir()
        (root / "previous.txt").write_text(root.name, encoding="utf-8")
    state = PipelineState(affine_completed_dates={date}, dataset_builds={"full_init": {"previous": True}})
    state_path = pipeline_state_path_for_spec(spec)
    write_pipeline_state(state_path, state)
    initialize_folder_lineage(spec.cosmic_dir)
    return spec, state, state_path, plan, [f"{date}/B1"]


def run_full_init_fixture(args):
    spec, state, state_path, plan, selection = args
    return supplement.run_full_init(spec, state, state_path, None,
                                    selected_test_folders=selection, plan=plan)


def test_supplement_plan_stops_at_hierarchy_group_depth(tmp_path, monkeypatch):
    hierarchy = HierarchySpec(
        (HierarchyLevel("level_1", "group", True), HierarchyLevel("level_2", "species", True),
         HierarchyLevel("level_3", "prefix")),
        {"level_1": {"A": "G"}, "level_2": {"A": "S"}, "level_3": {"A": "A"}},
        explicit=True,
    )
    mapping = DatasetMapping(
        ("genus", "prefix"), "prefix", "genus", {"A": {"genus": "G", "prefix": "A"}},
    )
    extra = tmp_path / "extra"
    source = extra / "A_manual"
    source.mkdir(parents=True)
    (source / "cell1.arc_data").touch()
    init = tmp_path / "init"
    groups = init / "G" / "S"
    for name in ("A_manual", "A20240101_01", "manual10", "manual2"):
        (groups / name / "deeper").mkdir(parents=True)

    scandir = supplement.os.scandir
    visited = []

    def record_scan(path):
        visited.append(Path(path))
        return scandir(path)

    monkeypatch.setattr(supplement.os, "scandir", record_scan)
    plan = supplement.scan_supplement_plan(extra, init, mapping, hierarchy)
    assert [group.target_relative.as_posix() for group in plan.groups] == ["G/S/A_manual"]
    assert plan.unbacked_init_folders == ("G/S/manual2", "G/S/manual10")
    assert any(path == groups for path in visited)
    assert all(groups not in path.parents for path in visited)


def test_full_init_stages_supplements_and_records_results(init_paths):
    spec, state, state_path, _, selection = init_paths
    report = run_full_init_fixture(init_paths)
    assert report["count"]["file_count"] == 2
    assert report["supplement"]["file_count"] == 1
    assert report["test_files"] == 1
    assert len(list(spec.init_dir.rglob("*.arc_data"))) == 2
    assert len(list(spec.init_test_dir.rglob("*.arc_data"))) == 1
    assert load_pipeline_state(state_path).dataset_builds["full_init"] == report
    assert state.dataset_builds["full_init"] == report
    lineage = load_folder_lineage(spec.cosmic_dir)
    assert Path(lineage["folders"][selection[0]]["outputs"]["test"]).is_dir()


@pytest.mark.parametrize("failure_step", ["supplement", "second_target", "state"])
def test_full_init_failure_keeps_previous_directories_and_database(init_paths, monkeypatch, failure_step):
    spec, state, state_path, _, _ = init_paths
    original_database = state_path.read_bytes()

    if failure_step == "supplement":
        def fail_supplement(*args):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(init_builder, "build_spectrum_folder", fail_supplement)
    elif failure_step == "second_target":
        original_rename = file_ops.rename_directory
        failed = False

        def fail_second_publication(source, target):
            nonlocal failed
            if target == spec.init_test_dir and not failed:
                failed = True
                raise RuntimeError("forced failure")
            original_rename(source, target)

        monkeypatch.setattr(file_ops, "rename_directory", fail_second_publication)
    else:
        def fail_state_write(path, new_state):
            write_pipeline_state(path, new_state)
            raise RuntimeError("forced failure")

        monkeypatch.setattr(supplement, "write_pipeline_state", fail_state_write)

    with pytest.raises(RuntimeError, match="forced failure"):
        run_full_init_fixture(init_paths)
    assert (spec.init_dir / "previous.txt").read_text(encoding="utf-8") == "init"
    assert (spec.init_test_dir / "previous.txt").read_text(encoding="utf-8") == "init_test"
    assert state_path.read_bytes() == original_database
    assert state.dataset_builds["full_init"] == {"previous": True}
    assert not list(spec.dataset_dir.glob(".*.tmp"))


@pytest.fixture
def legacy_db(init_paths):
    spec, _, state_path, _, _ = init_paths
    with db.connect(state_path) as connection:
        connection.execute("CREATE TABLE files (source_id TEXT, file_name TEXT)")
        connection.execute("INSERT INTO files VALUES ('20240101/A1', 'cell1.arc_data')")
        connection.execute("INSERT INTO test_defaults VALUES ('20240101/B1', 2, 'selected', 'today')")
        connection.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    return spec, state_path


def test_database_upgrade_preserves_v1_and_records(legacy_db):
    spec, path = legacy_db
    with pytest.raises(ValueError, match="upgrade-db"):
        load_pipeline_state(path)
    report = db.upgrade_database(path)
    assert report["status"] == "upgraded"
    assert db.upgrade_database(path)["status"] == "already_current"
    assert load_pipeline_state(path).dataset_builds["full_init"] == {"previous": True}
    assert db.load_test_defaults(path) == ["20240101/B1"]
    lineage = load_folder_lineage(spec.cosmic_dir)
    assert "cell1.arc_data" in lineage["folders"]["20240101/A1"]["files"]
    with db.connect(path) as current:
        assert current.execute("SELECT name FROM sqlite_master WHERE name = 'files'").fetchone() is None
    with closing(sqlite3.connect(report["backup"])) as backup:
        assert backup.execute("SELECT count(*) FROM files").fetchone() == (1,)
        assert backup.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone() == ("1",)


def test_database_upgrade_publication_failure_retains_source_and_backup(legacy_db, monkeypatch):
    _, path = legacy_db
    original = path.read_bytes()

    def fail_publish(staged, target):
        if target == path:
            raise OSError("forced failure")
        return file_ops.publish_file(staged, target)

    monkeypatch.setattr(db, "publish_file", fail_publish)
    with pytest.raises(OSError, match="forced failure"):
        db.upgrade_database(path)
    assert path.read_bytes() == original
    assert path.with_suffix(".v1.bak").is_file()
    assert not list(path.parent.glob(".*.tmp"))
