from __future__ import annotations

from raman_temp.core.hierarchy_meta import (
    build_hierarchy_entry,
    compute_split_hash,
    merge_hierarchy_meta,
)


def test_hierarchy_meta_merges_entries_and_run_history(tmp_path) -> None:
    train_split_path = tmp_path / "train_split.json"
    validation_split_path = tmp_path / "val_split.json"
    train_split_path.write_text("[\"A.arc_data\"]", encoding="utf-8")
    validation_split_path.write_text("[\"B.arc_data\"]", encoding="utf-8")
    first_entry = build_hierarchy_entry(
        "level_1/run_one",
        "level_1/run_one/level_1_model.pt",
        train_split_path="train_split.json",
        val_split_path="val_split.json",
        split_hash=compute_split_hash(train_split_path, validation_split_path),
    )
    second_entry = build_hierarchy_entry(
        "level_2/level_2_0/run_two",
        "level_2/level_2_0/run_two/level_2_0_model.pt",
        child_ids=[0, 1],
    )
    existing_meta = {
        "level_models": {"level_1": first_entry},
        "parent_models": {},
        "runs": {"level_1": [first_entry]},
    }
    current_meta = {
        "head_names": ["level_1", "level_2", "leaf"],
        "level_models": {},
        "parent_models": {"level_2": {"0": second_entry}},
        "runs": {"level_2_0": [second_entry]},
    }

    merged = merge_hierarchy_meta(existing_meta, current_meta)

    assert merged["level_models"]["level_1"] == first_entry
    assert merged["parent_models"]["level_2"]["0"] == second_entry
    assert merged["runs"] == {
        "level_1": [first_entry],
        "level_2_0": [second_entry],
    }
