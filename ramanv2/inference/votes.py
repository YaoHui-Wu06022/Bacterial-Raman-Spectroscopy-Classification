"""按文件夹统计预测票数。"""

from collections.abc import Sequence


def aggregate_top1_votes(
    labels: Sequence[str],
    probabilities: Sequence[float] | None = None,
) -> list[tuple[str, int, float]]:
    """按首次出现顺序返回 (label, count, mean_probability)。"""
    order: list[str] = []
    stats: dict[str, list[float]] = {}
    for index, label in enumerate(labels):
        key = str(label)
        if key not in stats:
            order.append(key)
            stats[key] = [0, 0.0]
        stats[key][0] += 1
        if probabilities is not None:
            stats[key][1] += float(probabilities[index])
    return [
        (
            key,
            int(stats[key][0]),
            float(stats[key][1]) / int(stats[key][0]) if probabilities is not None else 0.0,
        )
        for key in order
    ]
