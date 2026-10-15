"""标签版本和类别空间在旧实验读取、续训与合并前保持严格一致。"""

from copy import deepcopy
import json

import pytest

from ramanv2.core.config import build_config
from ramanv2.core.config_file import save_shared_config, write_yaml_dict
from ramanv2.core.experiment_reader import load_run_snapshot
from ramanv2.core.hierarchy_meta import build_hierarchy_meta, load_hierarchy_meta, merge_hierarchy_meta
from ramanv2.core.run_context import open_experiment_context
from ramanv2.evaluation.context import load_evaluation_context
from ramanv2.inference.predictor import load_predictor


LABEL_SCHEMA = "genus_species_prefix_v1"


def _build_meta():
    return build_hierarchy_meta(
        ["level_1", "level_2", "leaf"],
        {"level_1": ["Genus"], "level_2": ["Genus/AB", "Genus/AJO"],
         "leaf": ["Genus/AB20231212_01", "Genus/AJO20231212_01"]},
        {"level_1": {}, "level_2": {"0": [0, 1]}},
        {"level_1": None, "level_2": "level_1", "leaf": "level_2"},
        "level_1", {}, {}, {},
    )


def test_label_schema_is_fixed_in_shared_resolved_and_hierarchy_snapshots():
    config = build_config()
    assert config.to_shared_dict()["label_schema"] == LABEL_SCHEMA
    assert config.to_dict()["label_schema"] == LABEL_SCHEMA
    assert _build_meta()["label_schema"] == LABEL_SCHEMA


@pytest.mark.parametrize("schema", [None, "directory_v1"])
def test_legacy_meta_and_inference_are_rejected_without_rewriting_files(tmp_path, schema):
    meta = _build_meta()
    meta.pop("label_schema", None)
    if schema is not None:
        meta["label_schema"] = schema
    path = tmp_path / "hierarchy_meta.json"
    original = json.dumps(meta).encode("utf-8")
    path.write_bytes(original)
    with pytest.raises(ValueError, match="label_schema"):
        load_hierarchy_meta(path)
    with pytest.raises(ValueError, match="label_schema"):
        load_predictor(tmp_path, "cpu", "level_1")
    assert path.read_bytes() == original


@pytest.mark.parametrize("resolved_only", [False, True])
def test_legacy_run_snapshot_and_evaluation_are_rejected(tmp_path, resolved_only):
    config = build_config().to_dict()
    config.pop("label_schema", None)
    write_yaml_dict(tmp_path / "shared_config.yaml", config)
    run = tmp_path / "level_1/run_old"
    run.mkdir(parents=True)
    filename = "resolved_config.yaml" if resolved_only else "model_config.yaml"
    write_yaml_dict(run / filename, config)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="label_schema"):
        load_run_snapshot(run)
    with pytest.raises(ValueError, match="label_schema"):
        load_evaluation_context(tmp_path)
    with pytest.raises(ValueError, match="label_schema"):
        open_experiment_context(tmp_path, build_config())
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("key", ["class_names_by_level", "parent_to_children", "parent_level_name", "head_names"])
def test_hierarchy_merge_rejects_changed_class_order_or_relations(key):
    existing = _build_meta()
    current = deepcopy(existing)
    if key == "class_names_by_level":
        current[key]["level_2"].reverse()
    elif key == "parent_to_children":
        current[key]["level_2"]["0"] = [0]
    elif key == "parent_level_name":
        current[key]["level_2"] = None
    else:
        current[key].reverse()
    with pytest.raises(ValueError, match=key):
        merge_hierarchy_meta(existing, current)


def test_current_run_snapshot_can_be_read(tmp_path):
    config = build_config()
    save_shared_config(tmp_path / "shared_config.yaml", config)
    run = tmp_path / "level_1/run_new"
    run.mkdir(parents=True)
    write_yaml_dict(run / "model_config.yaml", config.to_model_dict())
    assert load_run_snapshot(run).config == config
