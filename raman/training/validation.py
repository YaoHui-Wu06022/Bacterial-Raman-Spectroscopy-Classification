import numpy as np
import torch

from raman.eval.common import (
    compute_classification_metrics,
    mask_logits_by_parent,
    select_level_targets,
)
from raman.tool.model import select_logits
from raman.training.se_stats import (
    accumulate_se_stats,
    attach_se_scale_hooks,
    finalize_se_stats,
    init_se_stats_accumulator,
)


def evaluate_validation_loader(
    model,
    loader,
    device,
    head_index,
    head_name=None,
    label_map_tensor=None,
    parent_index=None,
    parent_to_children=None,
):
    """训练期验证入口：统一处理层级标签、局部映射与父类遮罩"""
    model.eval()
    criterion_eval = torch.nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    all_targets = []
    all_preds = []
    num_classes = None
    se_accumulators = init_se_stats_accumulator(model)
    batch_scales = {}
    se_hooks = attach_se_scale_hooks(model, batch_scales)

    try:
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device)

                batch_scales.clear()
                logits = select_logits(model(x), head_name=head_name)
                y_level = select_level_targets(y, head_index)

                if label_map_tensor is not None:
                    invalid = y_level < 0
                    y_level = label_map_tensor[y_level.clamp_min(0)]
                    y_level[invalid] = -1

                if parent_index is not None and parent_to_children is not None:
                    if y.ndim != 2:
                        raise ValueError("parent_index 需要完整的多层标签输入")
                    parent_labels = y[:, parent_index]
                    logits, valid_parent = mask_logits_by_parent(
                        logits,
                        parent_labels,
                        parent_to_children,
                    )
                else:
                    valid_parent = torch.ones_like(y_level, dtype=torch.bool)

                valid = (y_level >= 0) & valid_parent
                accumulate_se_stats(se_accumulators, batch_scales, valid)
                if not valid.any():
                    continue

                logits = logits[valid]
                y_valid = y_level[valid]

                if num_classes is None:
                    num_classes = logits.size(1)

                loss = criterion_eval(logits, y_valid)
                batch_size = y_valid.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size

                all_preds.append(logits.argmax(1).detach().cpu().numpy())
                all_targets.append(y_valid.detach().cpu().numpy())
    finally:
        for hook in se_hooks:
            hook.remove()

    if num_classes is None or not all_targets:
        metrics = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_recall": 0.0,
        }
        return 0.0, 0.0, metrics, finalize_se_stats(se_accumulators)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=range(num_classes),
    )
    return total_loss / max(total, 1), metrics["accuracy"], metrics, finalize_se_stats(se_accumulators)

