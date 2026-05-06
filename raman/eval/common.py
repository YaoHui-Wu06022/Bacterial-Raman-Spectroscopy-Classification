import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


def select_logits(pred, head_name=None):
    """从模型输出中取出当前要使用的 logits

    兼容三种输出形式：
    - dict：多头输出，按 head_name 取指定层
    - tuple/list：默认取第一个元素
    - tensor：直接返回
    """
    if isinstance(pred, dict):
        if head_name is None:
            head_name = list(pred.keys())[-1]
        return pred[head_name]
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred


def mask_logits_by_parent(logits, parent_labels, parent_to_children):
    """按父类约束对子类 logits 做遮罩

    对 batch 中每个样本，只保留该父类允许出现的 child logits，
    其它位置填成 `-inf`，这样 softmax 后概率只会落在允许子类上
    `valid` 用来标记哪些样本成功找到了父类对应的子类集合
    """
    if parent_labels is None or parent_to_children is None:
        valid = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, valid

    device = logits.device
    batch = logits.size(0)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    valid = torch.zeros(batch, dtype=torch.bool, device=device)

    for i, parent_id in enumerate(parent_labels.tolist()):
        if parent_id < 0:
            continue

        child_idx = parent_to_children.get(parent_id)
        if child_idx is None:
            child_idx = parent_to_children.get(str(parent_id))
        if not child_idx:
            continue

        num_classes = logits.size(1)
        child_list = list(child_idx)
        invalid = [c for c in child_list if c < 0 or c >= num_classes]
        if invalid:
            raise ValueError(
                f"parent_to_children index out of range: parent={parent_id}, "
                f"invalid={invalid}, num_classes={num_classes}"
            )

        mask[i, child_list] = True
        valid[i] = True

    masked_logits = logits.masked_fill(~mask, float("-inf"))
    if (~valid).any():
        masked_logits[~valid] = 0.0

    return masked_logits, valid


def mask_logits_by_allowed(logits, allowed_indices):
    """按显式允许的类别索引集合做遮罩"""
    if not allowed_indices:
        return logits, None

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, allowed_indices] = True
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    valid = mask.any(dim=1)
    if (~valid).any():
        masked_logits[~valid] = 0.0
    return masked_logits, valid


def resolve_allowed_indices(class_names, allowed):
    """把类别名或类别索引形式的限制统一成索引列表"""
    if not allowed:
        return []

    items = list(allowed) if isinstance(allowed, (list, tuple, set)) else [allowed]
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    indices = []
    for item in items:
        if isinstance(item, int):
            indices.append(int(item))
            continue
        idx = name_to_idx.get(str(item))
        if idx is not None:
            indices.append(int(idx))
    return sorted(set(indices))


def select_level_targets(y, head_index=None):
    """从多层标签矩阵中取出当前层的目标标签"""
    if y.ndim != 2:
        return y
    if head_index is None:
        head_index = y.size(1) - 1
    return y[:, head_index]


def compute_classification_metrics(y_true, y_pred, labels):
    """统一计算 accuracy、macro_f1 和 macro_recall"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)

    if y_true.size == 0 or y_pred.size == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_recall": 0.0,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
    }


def forward_level_with_probs(
    model,
    x,
    *,
    head_name=None,
    parent_labels=None,
    parent_to_children=None,
    allowed_indices=None,
):
    """执行单层前向，并在需要时统一施加父类遮罩与允许类别遮罩

    返回：
    - logits：遮罩后的 logits
    - probs：softmax 概率
    - valid：当前样本是否仍有合法类别可选；没有约束时为 None
    """
    logits = select_logits(model(x), head_name=head_name)
    valid_parts = []

    if parent_labels is not None and parent_to_children is not None:
        # 父类约束只在“全局模型但需要限制到某个父类子集”时使用
        logits, valid_parent = mask_logits_by_parent(logits, parent_labels, parent_to_children)
        valid_parts.append(valid_parent)

    if allowed_indices:
        # 额外的 allowed_indices 常用于预测时的人工 parent_mask
        logits, valid_allowed = mask_logits_by_allowed(logits, allowed_indices)
        if valid_allowed is not None:
            valid_parts.append(valid_allowed)

    probs = torch.softmax(logits, dim=1)
    valid = None
    for part in valid_parts:
        valid = part if valid is None else (valid & part)
    return logits, probs, valid


def run_cascade_inference(
    runtime,
    x,
    *,
    level_order,
    target_level,
    num_classes_by_level,
    class_names_by_level,
    parent_to_children,
    allowed_names_by_level=None,
    fallback_to_previous=False,
):
    """执行从顶层到目标层的共享级联推理"""
    parent_pred = None
    last_result = None
    allowed_names_by_level = allowed_names_by_level or {}
    device = x.device

    for level_name in level_order:
        level_class_names = class_names_by_level.get(level_name, [])
        allowed_global = resolve_allowed_indices(
            level_class_names,
            allowed_names_by_level.get(level_name),
        )
        step = runtime.prepare_cascade_step(
            level_name,
            parent_pred,
            num_classes=num_classes_by_level[level_name],
            level_class_names=level_class_names,
            parent_to_children=parent_to_children,
        )
        if step is None:
            return last_result if fallback_to_previous else None

        if step["mode"] == "direct":
            pred_global = int(step["pred_global"])
            current = {
                "resolved_level": level_name,
                "probs": torch.ones((1, 1), device=device, dtype=torch.float32),
                "class_names": step["class_names"],
                "child_ids": list(step["child_ids"]),
                "pred_global": pred_global,
            }
        else:
            allowed_indices = allowed_global
            child_ids = step.get("child_ids")
            if child_ids is not None and allowed_global:
                allowed_set = set(allowed_global)
                allowed_indices = [
                    local_idx
                    for local_idx, child_id in enumerate(child_ids)
                    if int(child_id) in allowed_set
                ]

            parent_label_value = step.get("parent_labels")
            parent_labels = None
            if parent_label_value is not None:
                parent_labels = torch.tensor([parent_label_value], device=device)

            _, probs, valid = forward_level_with_probs(
                step["model"],
                x,
                parent_labels=parent_labels,
                parent_to_children=step.get("parent_to_children"),
                allowed_indices=allowed_indices,
            )
            if valid is not None and not valid.any():
                return last_result if fallback_to_previous else None

            pred_local = int(probs.argmax(1).item())
            pred_global = int(child_ids[pred_local]) if child_ids is not None else pred_local
            current = {
                "resolved_level": level_name,
                "probs": probs,
                "class_names": step["class_names"],
                "child_ids": list(child_ids) if child_ids is not None else None,
                "pred_global": pred_global,
            }

        if level_name == target_level:
            return current
        last_result = current
        parent_pred = pred_global

    return last_result if fallback_to_previous else None
