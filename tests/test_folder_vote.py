from ramanv2.inference.test_labels import build_folder_summary


def _predictions(label: str, count: int) -> list[dict[str, str]]:
    return [{"top1_label": label} for _ in range(count)]


def test_folder_requires_seven_of_ten_votes() -> None:
    summary = build_folder_summary(
        "A20240101_01",
        "A",
        ["A", "B"],
        _predictions("A", 7) + _predictions("B", 3),
    )
    assert summary["required_majority_count"] == 7
    assert summary["high_confidence"] is True
    assert summary["predicted_label"] == "A"
    assert summary["folder_correct"] is True


def test_folder_without_high_confidence_has_no_output() -> None:
    summary = build_folder_summary(
        "A20240101_01",
        "A",
        ["A", "B"],
        _predictions("A", 6) + _predictions("B", 4),
    )
    assert summary["high_confidence"] is False
    assert summary["predicted_label"] == "unknown"
    assert summary["folder_correct"] is False


def test_folder_with_confident_wrong_label_fails() -> None:
    summary = build_folder_summary(
        "A20240101_01",
        "A",
        ["A", "B"],
        _predictions("B", 7) + _predictions("A", 3),
    )
    assert summary["high_confidence"] is True
    assert summary["predicted_label"] == "B"
    assert summary["folder_correct"] is False


def test_fewer_samples_use_seventy_percent_ceiling() -> None:
    confident = build_folder_summary(
        "A20240101_01",
        "A",
        ["A", "B"],
        _predictions("A", 4) + _predictions("B", 1),
    )
    uncertain = build_folder_summary(
        "A20240101_01",
        "A",
        ["A", "B"],
        _predictions("A", 3) + _predictions("B", 2),
    )
    assert confident["required_majority_count"] == 4
    assert confident["folder_correct"] is True
    assert uncertain["required_majority_count"] == 4
    assert uncertain["predicted_label"] == "unknown"
