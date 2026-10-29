"""父类子模型路由预测。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ramanv2.inference.predictor import Predictor


def predict_routed_labels(
    sample_indices: Sequence[int],
    true_labels: Sequence[int],
    parent_labels: Sequence[int],
    parent_to_children: Mapping[str, Sequence[int]],
    predictors: dict[int, Predictor],
    load_predictor_for_parent: Callable[[int], Predictor],
    predict_for_parent: Callable[[Predictor, int], int],
) -> list[tuple[int, int, int]]:
    """按真实父类路由样本，返回 (样本索引, 真值, 预测值)。

    单子类父节点直接取该子类；多子类时按父类加载并复用子模型。
    ``parent_to_children`` 以字符串父类索引为键，与层级元数据保持一致。
    """
    results: list[tuple[int, int, int]] = []
    for sample_index, true_label, parent_id in zip(sample_indices, true_labels, parent_labels):
        true_label = int(true_label)
        parent_id = int(parent_id)
        if true_label < 0 or parent_id < 0:
            continue
        child_ids = [int(item) for item in parent_to_children.get(str(parent_id), [])]
        if true_label not in child_ids:
            continue
        if len(child_ids) == 1:
            predicted_label = child_ids[0]
        else:
            predictor = predictors.get(parent_id)
            if predictor is None:
                predictor = load_predictor_for_parent(parent_id)
                predictors[parent_id] = predictor
            predicted_label = predict_for_parent(predictor, sample_index)
        results.append((int(sample_index), true_label, predicted_label))
    return results


__all__ = ["predict_routed_labels"]
