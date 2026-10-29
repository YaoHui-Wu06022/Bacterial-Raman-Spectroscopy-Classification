"""标签版本和类别空间在旧实验读取、续训与合并前保持严格一致。"""



from ramanv2.core.config import build_config
from ramanv2.core.hierarchy_meta import build_hierarchy_meta, load_hierarchy_meta, merge_hierarchy_meta


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








