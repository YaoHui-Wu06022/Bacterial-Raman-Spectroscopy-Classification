from pathlib import Path

from ramanv2.audit.plots import build_changed_plot_groups, build_plot_state


def test_deleted_folder_marks_previous_prefix_group() -> None:
    current = build_plot_state(
        [Path("Genus/A20240101_01"), Path("Genus/B20240101_01")]
    )
    prior = {
        ("Genus", "A20240101_01"): "A",
        ("Genus", "A20240101_02"): "A",
        ("Genus", "B20240101_01"): "B",
    }

    assert build_changed_plot_groups(current, prior) == {("Genus", "A")}


def test_folder_prefix_change_marks_both_groups() -> None:
    current = [{"genus": "Genus", "folder": "B20240101_01", "prefix": "B"}]
    prior = {("Genus", "B20240101_01"): "A"}

    assert build_changed_plot_groups(current, prior) == {
        ("Genus", "A"),
        ("Genus", "B"),
    }
